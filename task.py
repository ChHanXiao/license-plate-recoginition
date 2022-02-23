# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import warnings
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from model import build_model
from utils import gather_results, mkdir


class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.CHARS = cfg.CHARS
        self.LP_CATE = cfg.LP_CATE
        self.LP_CATE_SUB = cfg.LP_CATE_SUB
        self.model = build_model(cfg)
        self.T_length = cfg.T_length
        self.evaluator = evaluator
        self.save_flag = -10

    def forward(self, x):
        x = self.model(x)
        return x
    @torch.no_grad()
    def inference(self, meta):
        prebs = self.forward(meta['img'])
        preb_labels = list()
        for i in range(prebs.shape[0]):
            lpnum=''
            preb = prebs[i, :, :]
            preb_label = torch.argmax(preb,dim=1)
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(self.CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(self.CHARS) - 1):
                    if c == len(self.CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            for idx in no_repeat_blank_label:
                lpnum += self.CHARS[int(idx.item())]
            preb_labels.append(lpnum)
        return preb_labels


    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        images, labels, lengths, colors = batch
        device = labels.device
        prebs = self.forward(images)
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = torch.argmax(preb,dim=1)
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(self.CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(self.CHARS) - 1):
                    if c == len(self.CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            no_repeat_blank_label=torch.Tensor(no_repeat_blank_label).to(device)
            preb_labels.append(no_repeat_blank_label)
        preb={
            'preb_labels': preb_labels,
            'target_labels':  labels,
            'target_lengths': lengths,
            'target_colors': colors
        }
        return preb

    def training_step(self, batch, batch_idx):
        images, labels, lengths, colors = batch
        logits = self.forward(images)
        log_probs = logits.permute(1, 0, 2)  # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = self.model.sparse_tuple(self.T_length, lengths)
        loss = self.model.ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        loss_states={'ctc loss':loss}
        # log train losses
        if self.global_step % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({})| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )
            self.logger.info(log_msg)
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))
        self.save_model_state(os.path.join(self.cfg.save_dir, "model_last.pth"))
        self.lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        images, labels, lengths, colors = batch
        device = labels.device
        prebs = self.forward(images)
        log_probs = prebs.permute(1, 0, 2)  # for ctc loss: T x N x C
        log_probs = log_probs.log_softmax(2).requires_grad_()
        input_lengths, target_lengths = self.model.sparse_tuple(self.T_length, lengths)
        loss = self.model.ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        loss_states={'ctc loss':loss}

        if batch_idx % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({})| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx,
                lr,
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = torch.argmax(preb,dim=1)

            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(self.CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:  # dropout repeate label and blank label
                if (pre_c == c) or (c == len(self.CHARS) - 1):
                    if c == len(self.CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            no_repeat_blank_label=torch.Tensor(no_repeat_blank_label).to(device)
            preb_labels.append(no_repeat_blank_label)
        preb={
            'preb_labels': preb_labels,
            'target_labels':  labels,
            'target_lengths': lengths,
            'target_colors': colors
        }
        return preb

    def validation_epoch_end(self, validation_step_outputs):
        """
        Called at the end of the validation epoch with the
        outputs of all validation steps.Evaluating results
        and save best model.
        Args:
            validation_step_outputs: A list of val outputs

        """
        Tp = 0
        Tn_1 = 0
        Tn_2 = 0
        result_TP={}
        result_TN1={}
        result_TN2={}
        for lpcate in self.LP_CATE:
            result_TP[lpcate] = 0
            result_TN1[lpcate] = 0
            result_TN2[lpcate] = 0
        # results = {}
        # for res in validation_step_outputs:
        #     results.update(res)
        # all_results = (
        #     gather_results(results)
        #     if dist.is_available() and dist.is_initialized()
        #     else results
        # )
        for i, validation_step_output in enumerate(validation_step_outputs):
            prebs = validation_step_output['preb_labels']
            targets_labels = validation_step_output['target_labels']
            target_lengths = validation_step_output['target_lengths']
            colors = validation_step_output['target_colors']
            start = 0
            targets = []
            for length in target_lengths:
                targets.append(targets_labels[start:start + length])
                start += length
            for preb,target,color in zip(prebs,targets,colors):
                if len(preb) != len(target):
                    result_TN1[color] = result_TN1[color] + 1
                    Tn_1 += 1
                    continue
                if (target == preb).all():
                    result_TP[color] = result_TP[color] + 1
                    Tp += 1
                else:
                    result_TN2[color] = result_TN2[color] + 1
                    Tn_2 += 1

        Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2+0.00000001)
        if Acc > self.save_flag:
            self.save_flag = Acc
            best_save_path = os.path.join(self.cfg.save_dir, "model_best")
            mkdir(self.local_rank, best_save_path)
            self.trainer.save_checkpoint(
                os.path.join(best_save_path, "lp_model_best.ckpt")
            )
            self.save_model_state(
                os.path.join(best_save_path, "lp_model_best.pth")
            )

        log_msg = "Val Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2))
        self.logger.info(log_msg)
        self.scalar_summary("Val Accuracy", "Val", Acc, self.global_step)

        for lpcate in self.LP_CATE:
            Tp = result_TP[lpcate]
            Tn_1 = result_TN1[lpcate]
            Tn_2 = result_TN2[lpcate]
            Acc_p =  Tp * 1.0 / (Tp + Tn_1 + Tn_2+0.00000001)
            log_msg = "Val Accuracy {}: {} [{}:{}:{}:{}]".format(lpcate, Acc_p, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2))
            self.logger.info(log_msg)
            self.scalar_summary("Val Accuracy "+lpcate, "Val", Acc_p, self.global_step)


    def test_step(self, batch, batch_idx):
        prebs = self.predict(batch, batch_idx)
        return prebs

    def test_epoch_end(self, test_step_outputs):
        Tp = 0
        Tn_1 = 0
        Tn_2 = 0
        result_TP={}
        result_TN1={}
        result_TN2={}
        for lpcate in self.LP_CATE:
            result_TP[lpcate] = 0
            result_TN1[lpcate] = 0
            result_TN2[lpcate] = 0

        # results = {}
        # for res in test_step_outputs:
        #     results.update(res)
        # all_results = (
        #     gather_results(results)
        #     if dist.is_available() and dist.is_initialized()
        #     else results
        # )
        for i, validation_step_output in enumerate(test_step_outputs):
            prebs = validation_step_output['preb_labels']
            targets_labels = validation_step_output['target_labels']
            target_lengths = validation_step_output['target_lengths']
            colors = validation_step_output['target_colors']
            start = 0
            targets = []
            for length in target_lengths:
                targets.append(targets_labels[start:start + length])
                start += length
            for preb,target,color in zip(prebs,targets,colors):
                if len(preb) != len(target):
                    result_TN1[color] = result_TN1[color] + 1
                    Tn_1 += 1
                    continue
                if (target == preb).all():
                    result_TP[color] = result_TP[color] + 1
                    Tp += 1
                else:
                    result_TN2[color] = result_TN2[color] + 1
                    Tn_2 += 1

        Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2+0.00000001)
        self.logger.info("Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
        for lpcate in self.LP_CATE:
            Tp = result_TP[lpcate]
            Tn_1 = result_TN1[lpcate]
            Tn_2 = result_TN2[lpcate]
            Acc =  Tp * 1.0 / (Tp + Tn_1 + Tn_2+0.00000001)
            self.logger.info("Test Accuracy {}: {} [{}:{}:{}:{}]".format(lpcate, Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.

        Returns:
            optimizer
        """
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        name = optimizer_cfg.pop("name")
        build_optimizer = getattr(torch.optim, name)
        optimizer = build_optimizer(params=self.parameters(), **optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)

        return optimizer

    def optimizer_step(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        optimizer_idx=None,
        optimizer_closure=None,
        on_tpu=None,
        using_native_amp=None,
        using_lbfgs=None,
    ):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
            if self.cfg.schedule.warmup.name == "constant":
                warmup_lr = (
                    self.cfg.schedule.optimizer.lr * self.cfg.schedule.warmup.ratio
                )
            elif self.cfg.schedule.warmup.name == "linear":
                k = (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps) * (
                    1 - self.cfg.schedule.warmup.ratio
                )
                warmup_lr = self.cfg.schedule.optimizer.lr * (1 - k)
            elif self.cfg.schedule.warmup.name == "exp":
                k = self.cfg.schedule.warmup.ratio ** (
                    1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                )
                warmup_lr = self.cfg.schedule.optimizer.lr * k
            else:
                raise Exception("Unsupported warm up type!")
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        torch.save({"state_dict": self.model.state_dict()}, path)

    # ------------Hooks-----------------
    def on_train_start(self) -> None:
        if self.current_epoch > 0:
            self.lr_scheduler.last_epoch = self.current_epoch - 1

