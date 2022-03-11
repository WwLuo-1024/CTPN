import os

icdar_img_dir = 'dataset/train_data/image'
icdar_text_dir = 'dataset/train_data/text'
num_workers = 1
pretrained_weights = ' '

anchor_scale = 16
#Threshold
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

checkpoints_dir = 'checkpoints'
outputs = 'logs'