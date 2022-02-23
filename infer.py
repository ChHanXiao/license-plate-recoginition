'''
Date: 2022-01-11 21:45:55
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-01-21 20:48:52
FilePath: /license-plate-recoginition/infer.py
'''

import argparse
import os
import sys
sys.path.insert(0,os.getcwd())
import pytorch_lightning as pl
import torch
import cv2
from pytorch_lightning.callbacks import ProgressBar
from data.load_data import LPRDataLoader, collate_fn
from task import TrainingTask
from utils import (
    cfg,
    load_config,
    mkdir,
    LPLightningLogger,
    load_model_weight
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="val", help="task to run, test or val")
    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    parser.add_argument("--path", type=str, help="img path")

    args = parser.parse_args()
    return args

class Perdictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        logger.info("Creating model...")
        self.task = TrainingTask(cfg)
        ckpt = torch.load(model_path)
        self.task.model.load_state_dict(ckpt["state_dict"])

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)

        height, width, _ = img.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        img = self.transform(img)  #BGR

def main(args):
    load_config(cfg, args.config)
    local_rank = -1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)
    logger = LPLightningLogger(cfg.save_dir)
    logger.dump_cfg(cfg)

    logger.info("Setting up data...")
    val_dataset = LPRDataLoader(cfg, False,logger=logger)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    logger.info("Creating model...")
    task = TrainingTask(cfg)
    ckpt = torch.load(args.model)
    task.model.load_state_dict(ckpt["state_dict"])
    # task.load_state_dict(ckpt["state_dict"])

    # task.model.load_state_dict(ckpt["state_dict"])
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        gpus=cfg.device.gpu_ids,
        accelerator="ddp",
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        logger=logger,
    )
    logger.info("Starting testing...")
    trainer.test(task, val_dataloader)



if __name__ == "__main__":
    args = parse_args()
    main(args)