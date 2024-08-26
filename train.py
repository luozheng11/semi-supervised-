import argparse
import os
import logging
import random
import sys

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from tensorboardX import SummaryWriter

import losses
from network import UNet_Dual
from two_stream_dataloader import MYDataset, TwoStreamBatchSampler
from metrics import test_single_volume_dual
import ramps

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,
                    default='ACDC/Dual_learning', help='experiment_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=3,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=1580,
                    help='labeled data')
# costs

parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency_rampup', type=float,
                    default=181.0, help='consistency_rampup')
parser.add_argument('--consistency-scale', default=1.0, type=float, metavar='WEIGHT',
                    help='use consistency loss with given weight (default: None)')

parser.add_argument('--stabilization-rampup', default=181.0, type=float, metavar='EPOCHS',
                    help='length of the stabilization loss ramp-up')
parser.add_argument('--stable-threshold', default=0.4, type=float, metavar='THRESHOLD',
                    help='threshold for stable sample')
parser.add_argument('--stable-threshold-teacher', default=0.5, type=float, metavar='THRESHOLD',
                    help='threshold for stable sample')
parser.add_argument('--stabilization-scale', default=1.0, type=float, metavar='WEIGHT',
                    help='use stabilization loss with given weight (default: None)')                    

parser.add_argument('--logit-distance-cost', default=0.05, type=float, metavar='WEIGHT',
                    help='let the student model have two outputs and use an MSE loss '
                    'the logits with the given weight (default: only have one output)')
parser.add_argument('--model', default='train1.pth', type=str)


args = parser.parse_args()
device = torch.device('cuda:0')

#预训练权重路径
pretrain_path = r'D:\project\quanzhong\train1.pth_best_model.pth'

#调节学习率
def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, total_epoch):
    lr = args.base_lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.base_lr - args.initial_lr) + args.initial_lr

    # decline lr
    lr *= ramps.zero_cosine_rampdown(epoch, total_epoch)

    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr
    return lr

#调节损失权重
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


#训练函数
def train(args, snapshot_path):
    #分割网络的学习率
    base_lr = args.base_lr
    #需要分割的类别数
    num_classes = args.num_classes
    batch_size = args.batch_size
    #最大迭代次数
    max_iterations = args.max_iterations
    def create_model(ema=False):
        # Network definition
        model = UNet_Dual(in_chns=1,class_num=num_classes)
        return model

    stu1_model = create_model()  # student1 model
    stu2_model = create_model()  # student2 model

    #加载预训练权重
    stu1_model.load_state_dict(torch.load(pretrain_path))
    stu2_model.load_state_dict(torch.load(pretrain_path))

    stu1_model = stu1_model.to(device)
    stu2_model = stu2_model.to(device)


    #伪代码，这里编写数据的数据集类
    data_train = MYDataset(split='train')
    data_val = MYDataset(split='val')

    total_number = len(data_train)
    labeled_number = args.labeled_num

    #有标签图像的序号是（0，labeled_number-1），无标签图像的序号是（labeled_number，total_number）
    labeled_idxs = list(range(0, labeled_number))
    unlabeled_idxs = list(range(labeled_number, total_number))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    
    #这里是数据加载器，dataloader
    trainloader = DataLoader(data_train,batch_sampler=batch_sampler)
    valloader = DataLoader(data_val,batch_size=1,shuffle=False)

    #进入模型训练模式
    # stu1_model.train(base_dir='D/project/', split='train', num=None, transform=None)
    # stu2_model.train(base_dir='D/project/', split='val', num=None, transform=None)
    stu1_model.train()
    stu2_model.train()

    #两个分割网络的优化器
    stu1_optimizer = optim.SGD(stu1_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    stu2_optimizer = optim.SGD(stu2_model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    
    #损失函数
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    residual_logit_criterion = losses.symmetric_mse_loss
    eucliden_distance = losses.softmax_mse_loss

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
        stabilization_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
        stabilization_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    iter_num = 0
    val_num = 0
    #最大训练epoch数
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch))
    #噪声系数
    noise_r=0.2
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume1_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            print(type(volume1_batch))
            #加入噪声
            noise = torch.clamp(torch.randn_like(volume1_batch.float()) * 0.1, -noise_r, noise_r)
            volume2_batch = volume1_batch + noise

            #stu1_out1,stu2_out2是两个模型的输出结果,stu1e_out2,stu2e_out1是两个模型对加噪声图像的输出结果
            stu1_out1 = stu1_model(volume1_batch)
            stu1e_out2 = stu1_model(volume2_batch)
            stu2_out2 = stu2_model(volume1_batch)
            stu2e_out1 = stu2_model(volume2_batch)
            
            #两个学生模型均为UNet_Dual，所以输出为两个大小相同的分割结果P和伴随变量Q，故长度为2
            assert len(stu1_out1) == 2
            #stu1_seg_logit，stu1_cons_logit分别为分割结果和伴随变量
            stu1_seg_logit, stu1_cons_logit = stu1_out1
            stu2_seg_logit, stu2_cons_logit = stu2_out2
            stu1e_seg_logit, stu1e_cons_logit = stu1e_out2
            stu2e_seg_logit, stu2e_cons_logit = stu2e_out1
            
            '''
            1.有监督学习下的图像分割损失，以下代码采用交叉熵损失作为分割损失函数
            '''
            #针对有标签图像

            label_batch = label_batch.squeeze()
            #模型1：stu1----真实标签Yl被直接用于监督有标注分割结果Pl
            stu1_loss_ce = ce_loss(stu1_seg_logit[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) / args.labeled_bs
            #计算分割损失
            stu1_seg_loss = stu1_loss_ce
           
            #模型2：stu2----真实标签Yl被直接用于监督有标注分割结果Pl
            stu2_loss_ce = ce_loss(stu2_seg_logit[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) / args.labeled_bs
            stu2_seg_loss = stu2_loss_ce
            
            #stu1_seg_loss,stu2_seg_loss是最终目标损失函数的第一部分
            stu1_loss = stu1_seg_loss 
            stu2_loss = stu2_seg_loss
            
            '''
            2.计算一致性约束部分的损失函数
            '''

            #针对无标签图像

            #一致性损失，args.consistency_scale是一个超参数,默认值是1，args.consistenct_rampup是一个超参数，默认值是181
            # consistency loss
            consistency_weight = args.consistency_scale * sigmoid_rampup(epoch_num, args.consistency_rampup)

            #伴随变量，consistency_criterion是一个损失函数，这里是均方误差损失函数
            stu1_cons = Variable(stu1_cons_logit.detach().data, requires_grad=False)
            
            #对图像加入噪声，计算 无噪声图像输入监督学习的模型stu1模型的分割结果 和 加噪声图像的分割结果 的一致性损失
            stu1_consistency_loss1 = consistency_weight * consistency_criterion(stu1e_seg_logit[args.labeled_bs:], stu1_seg_logit[args.labeled_bs:])
            stu1_consistency_loss2 = consistency_weight * consistency_criterion(stu1_seg_logit[args.labeled_bs:], stu1_cons[args.labeled_bs:])
           
            #计算一致性损失的均值
            stu1_consistency_loss_1 = torch.mean(stu1_consistency_loss1)  
            stu1_consistency_loss_2 = torch.mean(stu1_consistency_loss2)
            stu1_consistency_loss = stu1_consistency_loss_1 + stu1_consistency_loss_2

            #stu1_consistency_loss是最终目标损失函数的第二部分
            stu1_loss += stu1_consistency_loss
        
            #与上同理
            stu2_cons = Variable(stu2_cons_logit.detach().data, requires_grad=False)
            stu2_consistency_loss1 = consistency_weight * consistency_criterion(stu2e_seg_logit[args.labeled_bs:], stu2_seg_logit[args.labeled_bs:])
            stu2_consistency_loss2 = consistency_weight * consistency_criterion(stu2_seg_logit[args.labeled_bs:], stu2_cons[args.labeled_bs:])
            stu2_consistency_loss_1 = torch.mean(stu2_consistency_loss1)
            stu2_consistency_loss_2 = torch.mean(stu2_consistency_loss2)
            stu2_consistency_loss = stu2_consistency_loss_1 + stu2_consistency_loss_2
            stu2_loss += stu2_consistency_loss
            
            '''
            3.计算基于像素稳定性的损失函数
            '''
            #计算分割结果的概率值，以及对应的索引值，即分割结果中概率最大的像素点的索引值
            # stu1_seg_logit的形状大小为(batch_size,channel,w,h),stu1_seg_v，stu1_seg_i的形状大小相同，都是（batch_size,w,h）
            stu1_seg_v, stu1_seg_i = torch.max(F.softmax(stu1_seg_logit[args.labeled_bs:], dim=1), dim=1)
            stu2_seg_v, stu2_seg_i = torch.max(F.softmax(stu2_seg_logit[args.labeled_bs:], dim=1), dim=1)
            stu1e_seg_v, stu1e_seg_i = torch.max(F.softmax(stu1e_seg_logit[args.labeled_bs:], dim=1), dim=1)
            stu2e_seg_v, stu2e_seg_i = torch.max(F.softmax(stu2e_seg_logit[args.labeled_bs:], dim=1), dim=1)
            
            #（batch_size,w,h) -->（batch_size,1,w,h）在第二维处增加一个维度
            stu1_seg_v, stu1_seg_i = stu1_seg_v.unsqueeze(1), stu1_seg_i.unsqueeze(1)       
            stu2_seg_v, stu2_seg_i = stu2_seg_v.unsqueeze(1), stu2_seg_i.unsqueeze(1)
            stu1e_seg_v, stu1e_seg_i = stu1e_seg_v.unsqueeze(1), stu1e_seg_i.unsqueeze(1)
            stu2e_seg_v, stu2e_seg_i = stu2e_seg_v.unsqueeze(1), stu2e_seg_i.unsqueeze(1)
            
            #将有标签和无标签图像分割结果都从计算图中分离，不参与梯度计算
            in_stu2_cons_logit = Variable(stu2_seg_logit[args.labeled_bs:].detach().data, requires_grad=False)
            tar_stu1_seg_logit = Variable(stu1_seg_logit[args.labeled_bs:].clone().detach().data, requires_grad=False)

            in_stu1_cons_logit = Variable(stu1_seg_logit[args.labeled_bs:].detach().data, requires_grad=False)
            tar_stu2_seg_logit = Variable(stu2_seg_logit[args.labeled_bs:].clone().detach().data, requires_grad=False)
            
            # stu1的像素稳定性判断
            # 1st condition
            #稳定性判断条件1：无噪声图像的分割结果和加噪声图像的分割结果相同(论文原文：原图像素的预测标签类别和扰动后像素的预测标签类别一致)
            stu1_mask_1 = (stu1_seg_i == stu1e_seg_i)
            # 2nd condition
            #稳定性判断条件2：无噪声图像的分割结果和加噪声图像的分割结果的概率值大于阈值(论文原文：像素预测标签为 c 类时，对应地在c类上预测的概率值大于阈值)
            stu1_mask_2 = stu1_mask_1 *  stu1_seg_v
            stu1e_mask_2 = stu1_mask_1 * stu1e_seg_v
            stu1_mask_3 = (stu1_mask_2 > args.stable_threshold)
            stu1e_mask_3 = (stu1e_mask_2 > args.stable_threshold)
            # finally mask
            stu1_mask = stu1_mask_3 + stu1e_mask_3 
            stu1_mask_inv = (stu1_mask == False)
            
            #stu2的像素稳定性判断，与上同理             
            # 1st condition
            stu2_mask_1 = (stu2_seg_i == stu2e_seg_i)
            # 2nd condition
            stu2_mask_2 = stu2_mask_1 *  stu2_seg_v
            stu2e_mask_2 = stu2_mask_1 * stu2e_seg_v
            stu2_mask_3 = (stu2_mask_2 > args.stable_threshold)
            stu2e_mask_3 = (stu2e_mask_2 > args.stable_threshold)
            # finally mask
            stu2_mask = stu2_mask_3 + stu2e_mask_3
            stu2_mask_inv = (stu2_mask == False) 

            '''
            论文2.3节中所述,当满足情况1)或者情况2)时候，利用一个模型中稳定/更稳定的像素信息去监督另一个模型中不稳定/稳定的像素信息的学习
            '''

            #情况1：两个模型都稳定，stu1的像素稳定性大于stu2的像素稳定性
            #情况2：一个模型稳定，在另一个模型中不稳定

            # stu1 and stu2 are stable
            #mask_1表示stu1和stu2都稳定的像素点
            mask_1 = stu1_mask * stu2_mask 
            # stu1 and stu2  eucliden_distance
            #   eucliden_distance: losses.softmax_mse_loss
            #分别计算stu1和stu2中加噪声图像得到的分割结果与伪标签之间的欧式距离
            stu1_dis = eucliden_distance(stu1e_seg_logit[args.labeled_bs:], stu1_seg_logit[args.labeled_bs:])
            stu2_dis = eucliden_distance(stu2e_seg_logit[args.labeled_bs:], stu2_seg_logit[args.labeled_bs:])
            #mask_stu1_dis表示的是表示stu1 的欧氏距离大于 stu2，且 stu1 和 stu2 都稳定的位置
            mask_stu1_dis = (stu1_dis > stu2_dis) * mask_1
            #mask_stu1_dis_inv是mask_stu1_dis的逆
            mask_stu1_dis_inv = (mask_stu1_dis == False)
            mask_stu2_dis = (stu2_dis > stu1_dis) * mask_1
            #mask_stu2_dis_inv是mask_stu2_dis的逆
            mask_stu2_dis_inv = (mask_stu2_dis == False)

            #stu2 to supervised stu1
            #mask1_stu2表示 stu2 对 stu1 进行监督训练的位置。
            mask1_stu2 = mask_stu1_dis + stu2_mask * stu1_mask_inv
            mask1_stu2_inv = (mask1_stu2 == False)
            #tar_stu2_seg_logit是stu2的分割结果，in_stu1_cons_logit是stu1的分割结果，最终得到目标预测值，用于stu1模型的训练
            tar_stu2_seg_logit = tar_stu2_seg_logit * mask1_stu2 + in_stu1_cons_logit * mask1_stu2_inv
            
            # stu1 to supervised stu2
            #mask1_stu1表示 stu1 对 stu2 进行监督训练的位置。
            mask1_stu1 = mask_stu2_dis + stu1_mask * stu2_mask_inv
            mask1_stu1_inv = (mask1_stu1 == False)
            tar_stu1_seg_logit = tar_stu1_seg_logit * mask1_stu1 + in_stu2_cons_logit * mask1_stu1_inv         
            
            #计算稳定性权重
            stabilization_weight = args.stabilization_scale * ramps.sigmoid_rampup(epoch_num, args.stabilization_rampup)
            stabilization_weight = (1-(args.labeled_bs / batch_size)) * stabilization_weight
            
            # stabilization loss for stu2 model
            stu2_stabilization_loss1 = stabilization_weight * stabilization_criterion(stu2_cons_logit[args.labeled_bs:], 
                                                                                      tar_stu1_seg_logit)
            stu2_stabilization_loss = torch.mean(stu2_stabilization_loss1)           
            stu2_loss += stu2_stabilization_loss
            
            # stabilization loss for stu1 model
            # stabilization_criterion : losses.softmax_mse_loss
            stu1_stabilization_loss1 = stabilization_weight * stabilization_criterion(stu1_cons_logit[args.labeled_bs:], 
                                                                                     tar_stu2_seg_logit)
            stu1_stabilization_loss = torch.mean(stu1_stabilization_loss1)            
            stu1_loss += stu1_stabilization_loss
            
            #反向传播
            stu1_optimizer.zero_grad()
            stu1_loss.backward()
            stu1_optimizer.step()

            stu2_optimizer.zero_grad()
            stu2_loss.backward()
            stu2_optimizer.step()
            
            # 动态调整学习率
            lr_stu1 = adjust_learning_rate(stu1_optimizer, epoch_num, i_batch, len(trainloader), max_epoch)
            lr_stu2 = adjust_learning_rate(stu2_optimizer, epoch_num, i_batch, len(trainloader), max_epoch)
            
            #将训练过程中的值写入tensorboard日志文件
            writer = SummaryWriter(snapshot_path + '/log')
            logging.info("{} iterations per epoch".format(len(trainloader)))
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_stu1, iter_num)
            writer.add_scalar('info/stu1_loss', stu1_loss, iter_num)
            writer.add_scalar('info/stu2_loss', stu2_loss, iter_num)
            writer.add_scalar('info/stu1_loss_ce', stu1_seg_loss, iter_num)
            writer.add_scalar('info/stu2_loss_ce', stu2_seg_loss, iter_num)
            writer.add_scalar('info/stu1_stabilization_loss',
                              stu1_stabilization_loss, iter_num)
            writer.add_scalar('info/stu2_stabilization_loss',
                              stu2_stabilization_loss, iter_num)
            writer.add_scalar('info/stu1_consistency_loss',
                              stu1_consistency_loss, iter_num)
            writer.add_scalar('info/stu2_consistency_loss',
                              stu2_consistency_loss, iter_num)
                              
            logging.info(
                'iteration %d : stu1_loss : %f, stu2_loss: %f' %
                (iter_num, stu1_loss.item(), stu2_loss.item()))
           
            #每20次迭代，将训练过程中的图像写入tensorboard日志文件
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs= torch.argmax(torch.softmax(
                    stu1_seg_logit, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
            if iter_num > 0 and iter_num % 20 == 0:
                stu1_model.eval()               
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    val_num = i_batch + (iter_num / 5) * len(data_val)
                    metric_i, writer = test_single_volume_dual(
                        sampled_batch["image"], sampled_batch["label"], stu1_model, val_num, writer, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(data_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(stu1_model.state_dict(), save_mode_path)
                    torch.save(stu1_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                stu1_model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(stu1_model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #模型文件保存路径，可以自己设定自己的文件保存路径
    snapshot_path = "D:/project/model_trained".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    #生成日志文件
    log_path = snapshot_path+"/log.txt"
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)