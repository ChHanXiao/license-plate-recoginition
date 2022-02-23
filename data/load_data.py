import os
import random

import cv2
import numpy as np
import torch
from imutils import paths
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from .chinese.PlateCommon import *

SMU_PATH=os.path.join(os.path.dirname(__file__), "sum_imgs")
BG_PATH=os.path.join(os.path.dirname(__file__), "env_imgs")

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    colors = []
    for _, sample in enumerate(batch):
        img, label, length, color = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
        colors.append(color)

    labels = np.asarray(labels).flatten().astype(np.int)
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths, colors)


class LPRDataLoader(DataLoader):
    def __init__(self, cfg, is_train=True,logger=None):
        self.img_size = cfg.input_size
        self.lpr_max_len = cfg.lpr_max_len
        self.CHARS = cfg.CHARS
        self.CHARS_DICT = {char:i for i, char in enumerate(self.CHARS)}
        self.img_dir = cfg.data.train.img_dir if is_train else cfg.data.val.img_dir
        self.img_paths = []
        for i in range(len(self.img_dir)):
            self.img_paths += [el for el in paths.list_images(self.img_dir[i])]
        lp_cate = cfg.LP_CATE
        lp_cate_count = {}
        for cate_ in lp_cate:
            lp_cate_count[cate_]=0
        for filename in self.img_paths:
            basename = os.path.basename(filename)
            imgname, suffix = os.path.splitext(basename)
            file_cate = imgname.split("-")[1]
            if file_cate in lp_cate_count:
                lp_cate_count[file_cate] += 1
            else:
                logger.info("{} not in LP_CATE".format(file_cate))

        for key,val in lp_cate_count.items():
            logger.info("CATE:{:10s} NUM:{:<7}".format(key,val))
        random.shuffle(self.img_paths)
        print(len(self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        if False:
            Image, lab, color = self.gen.gen_plate() #RGB
            height, width, _ = Image.shape
            if height != self.img_size[1] or width != self.img_size[0]:
                Image = cv2.resize(Image, (self.img_size[0],self.img_size[1]))
            Image = self.PreprocFun(Image[:,:,::-1]) #RGB-->BGR
            label = list()
            for c in lab:
                label.append(self.CHARS_DICT[c])
        else:
            filename = self.img_paths[index]
            Image = cv2.imread(filename)
            height, width, _ = Image.shape
            if height != self.img_size[1] or width != self.img_size[0]:
                Image = cv2.resize(Image, (self.img_size[0], self.img_size[1]))
            Image = self.transform(Image)  #BGR
            basename = os.path.basename(filename)
            imgname, suffix = os.path.splitext(basename)
            lpnum = imgname.split("-")[2]
            color = imgname.split("-")[1]

            label = list()
            for c in lpnum:
                label.append(self.CHARS_DICT[c])

        # if len(label) == 8:
        #     if self.check(label) == False:
        #         print(imgname)
        #         assert 0, "Error label ^~^!!!"
        return Image, label, len(label), color

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

    def check(self, label):
        if label[2] != self.CHARS_DICT['D'] and label[2] != self.CHARS_DICT['F'] \
                and label[-1] != self.CHARS_DICT['D'] and label[-1] != self.CHARS_DICT['F'] \
                and label[0] != self.CHARS_DICT['W'] \
                and label[-1] != self.CHARS_DICT['急']:
            print("Error label, Please check!")
            return False
        else:
            return True

