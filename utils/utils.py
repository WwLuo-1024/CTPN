import numpy as np
import cv2
from config import *

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    #initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    #if both of the width and height are None, then return the original image
    if width is None and height is None:
        return image

    #to check the if the width is None
    if width is None:
        #calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        #calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    #resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    #return the image
    return resized

def gen_anchor(featuresize, scale):
    """
    generate base anchor from feature map [HXW][9][4]
    reshape [HXW][9][4] TO [HXWX9][4]

    """
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    #gen k = 9 anchor size (h, w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchor = np.array([0, 0, 15, 15])
    #center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    #x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    #apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])
    return np.array(anchor).reshape((-1, 4))

def iou(box1, box1_area, box2, box2_area):
    """
    box1 [x1, y1, x2, y2] anchor
    box2 [x1, y1, x2, y2] ground-truth box
    """

    x1 = np.maximum(box1[0], box2[:, 0])
    x2 = np.minimum(box1[2], box2[:, 2])
    y1 = np.maximum(box1[1], box2[:, 1])
    y2 = np.minimum(box1[3], box2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + box2_area[:] - intersection[:])
    return iou

def cal_overlaps(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0])) #initial overlaps

    for i in range(boxes1.shape[0]):
        overlaps[i][:] = iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps

def bbox_transform(anchors, gtboxes):
    """
    compute relative predicted vertical coordinates Vc, Vh with respect to the
    bouding box location of an anchor
    """

    cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    cya = (anchors[:, 1] + anchors[:, 3]) * 0.5
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = anchors[:, 3] - anchors[:, 1] + 1.0

    vc = (cy - cya) / ha
    vh = np.log(h / ha)

    return np.vstack((vc,vh)).transpose()


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    imgh, imgw = imgsize

    #generate base anchor
    base_anchor = gen_anchor(featuresize, scale)

    #calculate iou
    overlaps = cal_overlaps(base_anchor, gtboxes)

    #init labels -1 ignore(越界） 0 is negative 1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)

    #for each ground truth box corresponds to an anchor which hase highest IOU
    gr_argmax_overlaps = overlaps.argmax(axis = 0) #

    #the anchor with the highest IOU overlap with a ground truth box
    anchor_argmax_overlaps = overlaps.argmax(axis = 1) #If axis=1, the index of the maximum value in each row is compared by row
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # IOU > IOU_POSITIVE
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1

    # IOU < IOU_NEGATIVR
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0

    # ensure that every ground truth box has at least one postive RPN region
    labels[gr_argmax_overlaps] = 1

    #only keep anchors inside the image
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgw) |
        (base_anchor[:, 3] >= imgh)
    )[0]

    labels[outside_anchor] = -1 #判断是否越界

    #subsample postive labels, if greater than RPN_POSTIVE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    #print (len(fg_index))
    if(len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace = False)] = -1

    #subsample negative labels
    if not OHEM:
        bg_index = np.where(labels == 0)[0]
        num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
        if (len(bg_index) > num_bg):
            labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace = False)] = -1

    bbox_targets = bbox_transform(base_anchor, gtboxes[anchor_argmax_overlaps, :])

if __name__ == '__main__':
    pass
    # heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    # heights = np.array(heights).reshape(len(heights), 1)
    # print(heights)