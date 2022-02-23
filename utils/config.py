'''
Date: 2022-01-05 22:53:35
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-01-06 23:11:40
FilePath: /license-plate-recoginition/utils/config.py
'''
from .yacs import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.save_dir = "./"
# common params for NETWORK
cfg.model = CfgNode(new_allowed=True)
cfg.input_size = [96,48]
cfg.lpr_max_len = 8
cfg.T_length = 18
# DATASET related params
cfg.data = CfgNode(new_allowed=True)
cfg.data.train = CfgNode(new_allowed=True)
cfg.data.val = CfgNode(new_allowed=True)
cfg.device = CfgNode(new_allowed=True)
# train
cfg.schedule = CfgNode(new_allowed=True)

# logger
cfg.log = CfgNode()
cfg.log.interval = 50

# testing
cfg.test = CfgNode()
# size of images for each device


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(cfg, file=f)
