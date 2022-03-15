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

def get_anchor(featuresize, scale):
    """
    gen base anchor from feature map [HXW][9][4]
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

def cal_rpn():
    pass

if __name__ == '__main__':
    pass
    # heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    # heights = np.array(heights).reshape(len(heights), 1)
    # print(heights)