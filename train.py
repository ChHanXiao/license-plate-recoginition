'''
Date: 2022-01-05 22:53:35
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-01-10 19:33:20
FilePath: /license-plate-recoginition/train.py
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
from utils import (
    cfg,
    load_config,
    mkdir,
    LPLightningLogger,
    load_model_weight
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args

def main(args):
    load_config(cfg, args.config)
    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)
    logger = LPLightningLogger(cfg.save_dir)
    logger.dump_cfg(cfg)
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    train_dataset = LPRDataLoader(cfg,logger=logger)
    val_dataset = LPRDataLoader(cfg, False,logger=logger)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
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
    if "load_model" in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model)
        load_model_weight(task.model, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))
    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg.schedule
        else None
    )

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        gpus=cfg.device.gpu_ids,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator="ddp",
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        resume_from_checkpoint=model_resume_path,
        callbacks=[ProgressBar(refresh_rate=0)],  # disable tqdm bar
        logger=logger,
        benchmark=True,
        gradient_clip_val=cfg.get("grad_clip", 0.0),
    )

    trainer.fit(task, train_dataloader, val_dataloader)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)