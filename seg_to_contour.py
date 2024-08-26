import os
import cv2
import numpy as np
from PIL import Image

# 定义处理图像并绘制轮廓的函数
# def process_and_draw_contours(image_path, binary_image_path, save_dir):
#     # 读取原始图像
#     original_image = cv2.imread(image_path)
#     # 读取二值图像
#     binary_array = np.array(Image.open(binary_image_path))
#
#     # 找到二值图像中的连通域
#     contours, _ = cv2.findContours(binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 复制原始图像，因为我们将在其上绘制轮廓
#     drawing = cv2.imread(image_path)
#     drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB)  # 转换为RGB格式以便于显示
#
#     # 为每个连通域绘制轮廓
#     for i, contour in enumerate(contours):
#         # 使用绿色绘制轮廓
#         cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 2)
#
#     # 保存带有轮廓的图像
#     save_path = os.path.join(save_dir, os.path.basename(image_path))
#     cv2.imwrite(save_path, drawing)
def process_and_draw_contours(image_path, binary_image_path, save_dir):
    # 读取原始图像
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error: Unable to open image at {image_path}")
        return

    # 读取二值图像
    binary_array = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image,(binary_array.shape[0] ,binary_array.shape[1] ))
    if binary_array is None:
        print(f"Error: Unable to open binary image at {binary_image_path}")
        return

    # 确保二值图像是单通道的
    if binary_array.ndim != 2:
        binary_array = cv2.cvtColor(binary_array, cv2.COLOR_BGR2GRAY)

    # 找到二值图像中的连通域

    contours, _ = cv2.findContours(binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 复制原始图像，因为我们将在其上绘制轮廓
    drawing = cv2.imread(image_path)
    drawing = cv2.resize(drawing, (binary_array.shape[0], binary_array.shape[1]))
    drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB)  # 转换为RGB格式以便于显示

    # 为每个连通域绘制轮廓
    for i, contour in enumerate(contours):
        if not contours:
            print("not found")
        else:
            # 使用绿色绘制轮廓，减小thickness的值可以调整线条粗细
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 1)
            # 使用红色绘制分割结果的外接矩形
           # x,y,w,h = cv2.boundingRect(contour)
           # cv2.rectangle(drawing, (x, y), (x+w, y+h), (0, 0, 255), 1)

    # 保存带有轮廓的图像
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, drawing)

# 图像文件夹路径
images_dir =  r'D:/project/test'  # 替换为实际的图像文件夹路径
binary_dir = r'D:/project/output'

# 保存处理后的图像的目录
save_dir = r'D:\project\contour'  # 替换为保存目录的路径

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 遍历文件夹中的所有JPG文件
for image_filename in os.listdir(images_dir):
    if image_filename.lower().endswith('.jpg'):
        # 构建完整的文件路径
        image_path = os.path.join(images_dir, image_filename)
        binary_image_path = os.path.join(binary_dir,image_filename.replace('.jpg','.png'))  # 假设二值图像的文件名是在原始文件名后添加 '_binary'

        # 处理并绘制轮廓
        process_and_draw_contours(image_path, binary_image_path, save_dir)

print('All images have been processed and contours have been drawn.')