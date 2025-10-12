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
from wespeaker.utils.prune_utils import pruning_loss, get_progressive_sparsity, get_learning_rate_with_plateau_decay
import torch.nn.utils as nn_utils

def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.train()
    # Accept either a single optimizer or a tuple (optimizer, optimizer_reg)
    if isinstance(optimizer, tuple):
        optimizer, optimizer_reg = optimizer
    else:
        optimizer_reg = None

    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    # Additional meters for pruning
    cls_loss_meter = tnt.meter.AverageValueMeter()
    pruning_loss_meter = tnt.meter.AverageValueMeter()

    # Pruning configuration
    use_pruning = configs.get('use_pruning_loss', False)
    if use_pruning:
        target_sp = configs.get('target_sparsity', 0.5)
        l1, l2 = configs.get('lambda_pair', (1.0, 5.0))
        orig_params = float(configs.get('original_ssl_num_params', 1.0))
        warmup_epochs = configs.get('sparsity_warmup_epochs', 5)
        sparsity_schedule = configs.get('sparsity_schedule', 'cosine')
        min_sparsity = configs.get('min_sparsity', 0.0)
        total_epochs = configs.get('num_epochs', 100)
        total_iters = total_epochs * epoch_iter
        plateau_start_ratio = configs.get('plateau_start_ratio', 0.9)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    # Optional two-phase LSQ controller from configs
    lsq_controller = configs.get('lsq_controller', None)
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B)
        if frontend_type == 'fbank' or str(frontend_type).startswith('lfcc'):
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

        # Prepare pruning target sparsity for current iteration
        if use_pruning:
            cur_iter = (epoch - 1) * epoch_iter + i
            warmup_iters = warmup_epochs * epoch_iter
            if cur_iter < warmup_iters:
                target_sp_cur = get_progressive_sparsity(
                    current_iter=cur_iter,
                    total_warmup_iters=warmup_iters,
                    target_sparsity=target_sp,
                    schedule_type=sparsity_schedule,
                    min_sparsity=min_sparsity,
                    total_iters=total_iters,
                    plateau_start_ratio=plateau_start_ratio,
                )
            else:
                target_sp_cur = get_progressive_sparsity(
                    current_iter=cur_iter,
                    total_warmup_iters=warmup_iters,
                    target_sparsity=target_sp,
                    schedule_type=sparsity_schedule,
                    min_sparsity=min_sparsity,
                    total_iters=total_iters,
                    plateau_start_ratio=plateau_start_ratio,
                )

            # Expected current params; rely on frontend for pruning stats when available
            try:
                cur_params = model.module.frontend.get_num_params()
            except Exception:
                try:
                    cur_params = model.frontend.get_num_params()
                except Exception:
                    cur_params = orig_params

            prune_reg, exp_sp = pruning_loss(cur_params, orig_params, target_sp_cur, l1, l2)
            total_loss = cls_loss + prune_reg
        else:
            prune_reg, exp_sp, target_sp_cur = 0.0, None, None
            total_loss = cls_loss

        # loss, acc
        loss_meter.add(total_loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())
        if use_pruning:
            cls_loss_meter.add(cls_loss.item())
            pruning_loss_meter.add(prune_reg.item())

        # updata the model
        optimizer.zero_grad()
        if optimizer_reg is not None:
            optimizer_reg.zero_grad()

        # Two-phase LSQ: unfreeze and gradient clipping if controller provided
        if lsq_controller is not None:
            try:
                lsq_controller.maybe_unfreeze((epoch - 1) * epoch_iter + i)
            except Exception:
                pass

        # scaler does nothing here if enable_amp=False
        scaler.scale(total_loss).backward()
        nn_utils.clip_grad_norm_(model.parameters(), 1.0)


        if lsq_controller is not None:
            try:
                # Zero out LSQ gradients if still in frozen phase (for static graph compatibility)
                lsq_controller.zero_lsq_gradients_if_frozen()
                # Apply gradient clipping
                lsq_controller.clip_gradients()
            except Exception:
                pass

        if optimizer_reg is not None:
            scaler.step(optimizer_reg)
            try:
                with torch.no_grad():
                    l1.clamp_(min=0.0)  # Ensure lambda1 >= 0 if present
            except Exception:
                pass
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate with plateau decay if enabled
        if configs.get('use_plateau_lr_decay', False):
            cur_iter = (epoch - 1) * epoch_iter + i
            new_lr = get_learning_rate_with_plateau_decay(
                current_iter=cur_iter,
                total_iters=total_iters,
                initial_lr=configs.get('lr', 2e-4),
                plateau_start_ratio=plateau_start_ratio,
                decay_factor=configs.get('lr_decay_factor', 0.1),
                min_lr_ratio=configs.get('min_lr_ratio', 0.01),
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            row = [epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin(),
                   loss_meter.value()[0], acc_meter.value()[0]]
            if use_pruning:
                row += [
                    round(cls_loss_meter.value()[0], 4),
                    round(pruning_loss_meter.value()[0], 4),
                    f"{target_sp_cur:.4f}",
                    f"{exp_sp:.4f}",
                ]
            logger.info(tp.row(tuple(row), width=10, style='grid'))

        if (i + 1) == epoch_iter:
            break

    summary = [epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin(),
               loss_meter.value()[0], acc_meter.value()[0]]
    if use_pruning:
        summary += [
            round(cls_loss_meter.value()[0], 4),
            round(pruning_loss_meter.value()[0], 4),
            f"{target_sp_cur:.4f}",
            f"{exp_sp:.4f}",
        ]
    logger.info(tp.row(tuple(summary), width=10, style='grid'))
