import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
from config import IMAGE_MEAN
from utils.utils import cal_rpn

class ICDARDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labeldir = labelsdir


    def __len__(self):
        return len(self.img_names)

    def box_transfer(self, coor_lists, rescale_fac = 1.0):
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)] #x1 x2 x3 x4
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)] #y1 y2 y3 y4
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16 * i - 0.5
                gtboxes.append(prev, ymin, next, ymax)
                prev = next
            gtboxes.append(prev, ymin, xmax, ymax)
        return np.array(gtboxes)

    def parse_gtfile(self, gt_path, rescale_fac = 1.0):
        coor_lists = list()
        with open(gt_path, encoding = 'utf-8') as f:
            content = f.readlines()
            for line in content:
                coor_list = line.split(',')[:8]
                if len(coor_list) == 8:
                    coor_lists.append(coor_list)
        return self.box_transfer(coor_lists, rescale_fac)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        img = cv2.imread(img_path)

        assert img is None, 'img not exists'
        h, w, c = img.shape
        rescale_fac = max(h, w) / 1600
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
            img = cv2.resize(img, (w,h))
        gt_path = os.path.join()
        gtbox = self.parse_gtfile(gt_path, rescale_fac)

        [cls, regr], base_anchors = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
        m_img = img - IMAGE_MEAN
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr]) #Element arrays are stacked horizontally
        cls = np.expand_dims(cls, axis = 0)

        #transform to torch tensor
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr