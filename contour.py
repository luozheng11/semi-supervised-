import cv2
import json
import numpy as np

'''
这个文件是用于将json格式的标签转换为二值灰度图像，用于模型的训练，需要自己修改路径
'''

#改成自己图像和对应的标注json文件的路径
image_path = "0001_cropped.jpg"
annotation_path = r"t-set\t-set\0001.json"

with open(annotation_path, 'r') as f:
    annotation_data = json.load(f)

image = cv2.imread(image_path)

segmentation_mask = np.zeros_like(image[:,:,0])

for shape in annotation_data['shapes']:
    label = shape['label']
    points = shape['points']
    polygon = np.array(points, np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    cv2.fillPoly(segmentation_mask, [polygon], 255)

# segmentation_mask = cv2.bitwise_not(segmentation_mask)

# cv2.imshow('Segmentation Mask', segmentation_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 或者保存二值图像
cv2.imwrite('0001_mask.png', segmentation_mask)