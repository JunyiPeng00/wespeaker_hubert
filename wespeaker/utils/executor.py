# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp

import torch
import torchnet as tnt
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug
from wespeaker.utils.prune_utils import pruning_loss

def run_epoch(dataloader, epoch_iter, model, criterion, optimizers, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.train()
    
    optimizer, optimizer_reg = optimizers

    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    # only used when pruning
    cls_loss_meter = tnt.meter.AverageValueMeter() 
    pruning_loss_meter = tnt.meter.AverageValueMeter() 

    use_pruning      = configs.get('use_pruning_loss', False)
    if use_pruning:
        target_sp        = configs.get('target_sparsity', 0.5)
        l1, l2           = configs.get('lambda_pair', (1.0, 5.0))
        orig_params      = float(configs.get('original_ssl_num_params', 1.0))
        warmup_epochs    = configs.get('sparsity_warmup_epochs', 5)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        if use_pruning:
            warmup_iters = warmup_epochs * epoch_iter # warmup epochs
            if cur_iter < warmup_iters:
                # linearly increase the target sparsity
                target_sp_cur = target_sp * cur_iter / warmup_iters
            else:
                target_sp_cur = target_sp  

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B)
        if frontend_type == 'fbank':
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                features, _ = model.module.frontend(wavs, wavs_len)

        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.module.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, cls_loss = outputs
            else:
                cls_loss = criterion(outputs, targets)
        
        # ==== pruning regularization ---------------------------------------
        if use_pruning:
            cur_params = model.module.frontend.get_num_params()
            prune_loss, exp_sp = pruning_loss(
                cur_params, orig_params, target_sp_cur, l1, l2
            )   
            total_loss = cls_loss + prune_loss  
        else:
            prune_loss, exp_sp = 0.0, None
            total_loss = cls_loss

        # loss, acc
        loss_meter.add(total_loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

        if use_pruning:
            cls_loss_meter.add(cls_loss.item())
            pruning_loss_meter.add(prune_loss.item())

        # updata the model
        optimizer.zero_grad()
        if use_pruning:
            optimizer_reg.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(total_loss).backward()
        if use_pruning:
            scaler.step(optimizer_reg)
            with torch.no_grad():
                l1.clamp_(min=0.0)  # clip l1
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            row_extra = [
                round(cls_loss_meter.value()[0], 4),
                round(pruning_loss_meter.value()[0], 4),
                f"{target_sp_cur:.4f}",
                f"{exp_sp:.4f}",
            ] if use_pruning else []
            row = [epoch, i + 1,
                   scheduler.get_lr(),
                   margin_scheduler.get_margin(),
                   round(loss_meter.value()[0], 4),
                   round(acc_meter.value()[0], 2)] + row_extra
            logger.info(tp.row(row, width=10, style='grid'))

        if (i + 1) == epoch_iter:
            break

    # ========= epoch-end summary =========
    summary = [epoch, i + 1,
               scheduler.get_lr(),
               margin_scheduler.get_margin(),
               round(loss_meter.value()[0], 4),
               round(acc_meter.value()[0], 2)]
    if use_pruning:
        summary += [
            round(cls_loss_meter.value()[0], 4),
            round(pruning_loss_meter.value()[0], 4),
            f"{target_sp_cur:.4f}",
            f"{exp_sp:.4f}"
        ]    
    logger.info(tp.row(tuple(summary), width=10, style='grid'))
