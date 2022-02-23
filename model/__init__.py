'''
Date: 2022-01-05 22:53:35
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-01-06 23:30:02
FilePath: /license-plate-recoginition/model/__init__.py
'''
import copy
from .LPRNet import *

def build_model(cfg):
    name = cfg.model.name
    if name == "PlateNet_multi":
        class_num = len(cfg.CHARS)
        model = PlateNet_multi(class_num)
    elif name == "PlateNet_multi_1":
        class_num = len(cfg.CHARS)
        model = PlateNet_multi_1(class_num)
    else:
        raise NotImplementedError
    return model