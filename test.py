'''
Date: 2022-01-05 22:53:35
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-02-14 12:18:46
FilePath: /license-plate-recoginition/test.py
'''
import argparse
import os
import sys

sys.path.insert(0,os.getcwd())
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ProgressBar

from data.load_data import LPRDataLoader, collate_fn
from task import TrainingTask
from utils import LPLightningLogger, cfg, load_config, load_model_weight, mkdir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="val", help="task to run, test or val")
    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    args = parser.parse_args()
    return args

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
