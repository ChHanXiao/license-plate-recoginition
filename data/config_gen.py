'''
Date: 2022-01-05 22:53:34
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-01-10 18:53:29
FilePath: /license-plate-recoginition/data/config_gen.py
'''
import os
import sys
sys.path.insert(0,os.getcwd())
from utils.yacs import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.p = [0.,0.,0.,1,0.,0.,0.] #[black_plate,blue_plate,yellow_plate,green_plate,white_plate,farm_plate,airport]

# cfg.p = [0.1,0.25,0.2,0.2,0.1,0.1,0.05] #[black_plate,blue_plate,yellow_plate,green_plate,white_plate,farm_plate]
cfg.black_plate = CfgNode(new_allowed=True)
cfg.black_plate.p = [0.4,0.3,0.3] #[COMMON,SHI,LIN]

cfg.blue_plate = CfgNode(new_allowed=True)
cfg.blue_plate.p = [1]

cfg.yellow_plate = CfgNode(new_allowed=True)
cfg.yellow_plate.p = [0.5,0.5] #[SINGLE, MULTI]

cfg.green_plate = CfgNode(new_allowed=True)
cfg.green_plate.p = [0.5,0.5] #[SMALL, LARGE]

cfg.farm_plate = CfgNode(new_allowed=True)
cfg.farm_plate.p = [0.5,0.5] #[COMMON, XUE]

cfg.white_plate = CfgNode(new_allowed=True)
cfg.white_plate.p = [0.0,0.2,0.2,0.2,0.1,0.1,0.2] #[COMMON,JINGCHA,JUNDUI,JUNDUI_M,WUJIN,WUJIN_M,YINGJI]
cfg.white_plate.front = 0.2

cfg.airport_plate = CfgNode(new_allowed=True)
cfg.airport_plate.p = [1]

def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "w") as f:
        print(cfg, file=f)