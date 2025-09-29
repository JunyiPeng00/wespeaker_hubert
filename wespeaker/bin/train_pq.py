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

import os
import re
from pprint import pformat

import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
import time

import wespeaker.utils.schedulers as schedulers
from wespeaker.dataset.dataset import Dataset
from wespeaker.frontend import *
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.utils.executor import run_epoch
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    spk2id
from wespeaker.utils.prune_utils import make_pruning_param_groups, StochasticWeightAveraging


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)

    # pruning related hyper-parameters
    use_pruning = configs.get("use_pruning_loss", False)
    if use_pruning:
        prune_defaults = {
            'use_pruning_loss': False,
            'target_sparsity': 0.5,
            'sparsity_warmup_epochs': 7,
            'sparsity_schedule': 'cosine',
            'min_sparsity': 0.0,
            'plateau_start_ratio': 0.9,
            'use_plateau_lr_decay': False,
            'lr_decay_factor': 0.1,
            'min_lr_ratio': 0.01,
            'use_swa': False,
            'swa_start_ratio': 0.9,
            'swa_update_freq': 10,
            'min_temperature': 0.1,
            'temperature_decay': 0.95,
            'temperature_decay_freq': 100,
        }
        for k, v in prune_defaults.items():
            configs.setdefault(k, v)

    # quantization related hyper-parameters
    use_quantization = configs.get("use_quantization", False)
    if use_quantization:
        quant_defaults = {
            'use_quantization': False,
            'quantization_config': '8bit_symmetric',
            'quantize_weights': True,
            'quantize_activations': False,
            'preserve_hp_gating': True,
            'freeze_lsq_steps': 2000,
            'lsq_step_lr': 1e-5,
            'grad_clip_norm': 1.0,
            'per_channel_weights': True,
            'per_channel_activations': False,
            'quantize_bias': False,
        }
        for k, v in quant_defaults.items():
            configs.setdefault(k, v)
    checkpoint = configs.get('checkpoint', None)
    # dist configs
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(configs['gpus'][local_rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir, exist_ok=True)
        except IOError:
            print("[warning] " + model_dir + " already exists !!!")
            if checkpoint is None:
                print("[error] checkpoint is null !")
                exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)

    # train data
    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)
    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}, spk num: {}".format(
            len(train_utt_spk_list), len(spk2id_dict)))

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            spk2id_dict,
                            train_lmdb_file=configs.get('train_lmdb', None),
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None))
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args'])
    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_spk_list)
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('epoch iteration number: {}'.format(epoch_iter))

    # model: frontend (optional) => speaker model => projection layer
    logger.info("<== Model ==>")
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    if frontend_type != "fbank":
        frontend_args = frontend_type + "_args"
        frontend = frontend_class_dict[frontend_type](
            **configs['dataset_args'][frontend_args],
            sample_rate=configs['dataset_args']['resample_rate'])
        configs['model_args']['feat_dim'] = frontend.output_size()
        model = get_speaker_model(configs['model'])(**configs['model_args'])
        model.add_module("frontend", frontend)
    else:
        model = get_speaker_model(configs['model'])(**configs['model_args'])
    if rank == 0:
        num_params = sum(param.numel() for param in model.parameters())
        logger.info('speaker_model size: {}'.format(num_params))

    # Apply quantization (guarded) before loading checkpoint
    if use_quantization:
        if rank == 0:
            logger.info("<== Quantization ==>\nApplying quantization to model (guarded)...")
        try:
            # Prefer wespeaker's quantization if exists
            from wespeaker.frontend.wav2vec2 import apply_quantization_with_hp_integration, get_quantization_config  # type: ignore
            quant_src = 'wespeaker.frontend.wav2vec2'
        except Exception:
            try:
                # Fallback: allow reuse of wedefense implementation if available in PYTHONPATH
                from wedefense.frontend.wav2vec2 import apply_quantization_with_hp_integration, get_quantization_config  # type: ignore
                quant_src = 'wedefense.frontend.wav2vec2'
            except Exception as e:  # no quantization utils
                if rank == 0:
                    logger.warning(f"Quantization utilities not found: {e}. Continue without quantization.")
                use_quantization = False
                configs['use_quantization'] = False
        if use_quantization:
            try:
                quant_config = get_quantization_config(configs['quantization_config'])
                if configs.get('quantize_weights', True):
                    quant_config.quantize_weights = True
                if configs.get('quantize_activations', False):
                    quant_config.quantize_activations = True
                quant_config.per_channel_weights = configs.get('per_channel_weights', True)
                quant_config.per_channel_activations = configs.get('per_channel_activations', False)
                if 'quantize_bias' in configs:
                    quant_config.quantize_bias = configs['quantize_bias']

                pruning_config = {
                    'target_sparsity': configs.get('target_sparsity', 0.5),
                    'sparsity_warmup_epochs': configs.get('sparsity_warmup_epochs', 7),
                    'sparsity_schedule': configs.get('sparsity_schedule', 'cosine'),
                    'min_sparsity': configs.get('min_sparsity', 0.0),
                    'plateau_start_ratio': configs.get('plateau_start_ratio', 0.9),
                    'min_temperature': configs.get('min_temperature', 0.1),
                    'temperature_decay': configs.get('temperature_decay', 0.95),
                    'temperature_decay_freq': configs.get('temperature_decay_freq', 100),
                }
                if checkpoint is not None:
                    load_checkpoint(model, checkpoint)
                    start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                                checkpoint)[0]) + 1
                    logger.info('Load checkpoint: {}'.format(checkpoint))
                else:
                    start_epoch = 1

                model = apply_quantization_with_hp_integration(
                    model,
                    config=quant_config,
                    enable_pruning=use_pruning,
                    pruning_config=pruning_config,
                )
                if rank == 0:
                    quantized_params = sum(param.numel() for param in model.parameters())
                    logger.info(f"Quantization applied via {quant_src}. Quantized model size: {quantized_params}")
            except Exception as e:
                if rank == 0:
                    logger.error(f"Failed to apply quantization: {e}. Continue without quantization.")
                use_quantization = False
                configs['use_quantization'] = False
    # For model_init, only frontend and speaker model are needed !!!
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        load_checkpoint(model, configs['model_init'])
    elif checkpoint is None:
        logger.info('Train model from scratch ...')
    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args'][
        'embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if configs['data_type'] != 'feat' and configs['dataset_args'][
            'speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info(
                'No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)
    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        if frontend_type == 'fbank':
            script_model = torch.jit.script(model)
            script_model.save(os.path.join(model_dir, 'init.zip'))

    # If specify checkpoint, load some info from checkpoint.
    # For checkpoint, frontend, speaker model, and projection layer
    # are all needed !!!
    if checkpoint is not None and not use_quantization:
        load_checkpoint(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('Load checkpoint: {}'.format(checkpoint))
    elif checkpoint is not None and use_quantization:
        all_state_dict = torch.load(checkpoint, map_location='cpu')
        
        model_state_dict = model.state_dict()
        loaded_keys = []
        
        for key in model_state_dict.keys():
            if key == 'projection.weight':  # 使用 'in' 支持更灵活的键名匹配
                if key in all_state_dict:
                    # 检查维度是否匹配
                    if model_state_dict[key].shape == all_state_dict[key].shape:
                        # 直接修改参数tensor的数据
                        model_state_dict[key].data.copy_(all_state_dict[key])
                        loaded_keys.append(key)
                        logger.info('Load projection weight: {} (shape: {})'.format(
                            key, model_state_dict[key].shape))
                    else:
                        logger.warning('Shape mismatch for {}: model {} vs checkpoint {}'.format(
                            key, model_state_dict[key].shape, all_state_dict[key].shape))
                else:
                    logger.warning('Key {} not found in checkpoint'.format(key))
        
        if not loaded_keys:
            logger.warning('No projection.weight keys found in model or checkpoint')

        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('Load checkpoint: {} (loaded {} projection weights)'.format(
            checkpoint, len(loaded_keys)))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    # Freeze pretraining-specific params; keep LSQ step_size trainable
    try:
        for name, param in model.named_parameters():
            if any(k in name for k in ["project_q", "final_proj"]) or ("quantizer" in name and "step_size" not in name):
                param.requires_grad = False
    except Exception:
        pass

    # ddp_model
    model.cuda()
    
    # Convert BatchNorm2d to SyncBatchNorm for better distributed training
    if rank == 0:
        logger.info("Converting BatchNorm2d to SyncBatchNorm...")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    # NOTE:
    # Static graph optimization is incompatible with training regimes where
    # the autograd graph changes over time (e.g., LSQ step_size unfreezing
    # after freeze_lsq_steps, or dynamic gating/pruning). When enabled, it can
    # trigger: "Your training graph has changed ... not compatible with static_graph set to True".
    #
    # Therefore, only opt-in when explicitly requested AND when neither
    # quantization-with-unfreeze nor pruning is used.
    use_pruning_flag = configs.get('use_pruning_loss', False)
    use_quant_flag = configs.get('use_quantization', False)
    freeze_steps = int(configs.get('freeze_lsq_steps', 0)) if use_quant_flag else 0
    allow_static_graph = bool(configs.get('ddp_static_graph', False)) and (not use_pruning_flag) and (not use_quant_flag or freeze_steps == 0)
    if allow_static_graph:
        ddp_model._set_static_graph()
    device = torch.device("cuda")

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    configs['optimizer_args']['lr'] = configs['scheduler_args']['initial_lr']
    optimizer_reg = None
    # Build pruning param groups when enabled, else fallback to quantization-aware or normal optimizer
    if use_pruning:
        # Track original frontend params for sparsity stats if possible
        try:
            configs['original_ssl_num_params'] = sum(param.numel() for param in model.module.frontend.parameters())
        except Exception:
            try:
                configs['original_ssl_num_params'] = sum(param.numel() for param in model.frontend.parameters())
            except Exception:
                configs['original_ssl_num_params'] = 1.0
        reg_lr = configs.get('initial_reg_lr', 2e-2)
        p_groups, lambda_pair = make_pruning_param_groups(
            ddp_model,
            cls_lr=configs['optimizer_args']['lr'],
            reg_lr=reg_lr,
        )
        pg_main   = [pg for pg in p_groups if pg.get('name') == 'main']
        pg_others = [pg for pg in p_groups if pg.get('name') != 'main']
        opt_kwargs = {k: v for k, v in configs['optimizer_args'].items() if k not in ('lr', 'reg_lr')}
        optimizer = getattr(torch.optim, configs['optimizer'])(pg_main, **opt_kwargs)
        optimizer_reg = getattr(torch.optim, configs['optimizer'])(pg_others, **opt_kwargs)
        configs['reg_lr'] = reg_lr
        configs['lambda_pair'] = lambda_pair
    else:
        # Build optimizer with optional two-phase LSQ support (guarded)
        optimizer = None
        if use_quantization:
            try:
                from wespeaker.frontend.wav2vec2.quantization_utils import build_two_phase_lsq_optimizer, collect_lsq_step_size_params  # type: ignore
                base_opt_cls = getattr(torch.optim, configs['optimizer'])
                main_lr = configs['optimizer_args']['lr']
                step_lr = configs.get('lsq_step_lr', main_lr)
                optimizer = build_two_phase_lsq_optimizer(ddp_model, base_opt_cls, main_lr=main_lr, step_lr=step_lr, **{k: v for k, v in configs['optimizer_args'].items() if k != 'lr'})
                if rank == 0:
                    lsq_params = collect_lsq_step_size_params(ddp_model)
                    logger.info(f"Enabled two-phase LSQ optimizer: {len(lsq_params)} step_size params, step_lr={step_lr}")
            except Exception:
                try:
                    from wedefense.frontend.wav2vec2.quantization_utils import build_two_phase_lsq_optimizer, collect_lsq_step_size_params  # type: ignore
                    base_opt_cls = getattr(torch.optim, configs['optimizer'])
                    main_lr = configs['optimizer_args']['lr']
                    step_lr = configs.get('lsq_step_lr', main_lr)
                    optimizer = build_two_phase_lsq_optimizer(ddp_model, base_opt_cls, main_lr=main_lr, step_lr=step_lr, **{k: v for k, v in configs['optimizer_args'].items() if k != 'lr'})
                    if rank == 0:
                        lsq_params = collect_lsq_step_size_params(ddp_model)
                        logger.info(f"Enabled two-phase LSQ optimizer (wedefense utils): {len(lsq_params)} step_size params, step_lr={step_lr}")
                except Exception as e:
                    if rank == 0:
                        logger.warning(f"Two-phase LSQ optimizer unavailable: {e}. Falling back to single optimizer.")
                    optimizer = getattr(torch.optim, configs['optimizer'])(ddp_model.parameters(), **configs['optimizer_args'])
        else:
            optimizer = getattr(torch.optim, configs['optimizer'])(ddp_model.parameters(), **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # Optional two-phase LSQ controller
    lsq_controller = None
    if use_quantization:
        try:
            from wespeaker.frontend.wav2vec2.quantization_utils import TwoPhaseLSQController  # type: ignore
            lsq_controller = TwoPhaseLSQController(
                model=ddp_model,
                optimizer=optimizer,
                freeze_steps=int(configs.get('freeze_lsq_steps', 2000)),
                clip_norm=float(configs.get('grad_clip_norm', 1.0)),
            )
            configs['lsq_controller'] = lsq_controller
        except Exception:
            try:
                from wedefense.frontend.wav2vec2.quantization_utils import TwoPhaseLSQController  # type: ignore
                lsq_controller = TwoPhaseLSQController(
                    model=ddp_model,
                    optimizer=optimizer,
                    freeze_steps=int(configs.get('freeze_lsq_steps', 2000)),
                    clip_norm=float(configs.get('grad_clip_norm', 1.0)),
                )
                configs['lsq_controller'] = lsq_controller
            except Exception:
                pass

    # Initialize SWA if enabled
    swa = None
    if configs.get('use_swa', False):
        total_iters = configs['num_epochs'] * epoch_iter
        swa_start_iter = int(total_iters * configs.get('swa_start_ratio', 0.9))
        swa_update_freq = configs.get('swa_update_freq', 10)
        swa = StochasticWeightAveraging(
            model=ddp_model,
            start_iter=swa_start_iter,
            update_freq=swa_update_freq
        )
        if rank == 0:
            logger.info(f"SWA enabled: start_iter={swa_start_iter}, update_freq={swa_update_freq}")

    # save config.yaml
    if rank == 0:  
        cfg_to_save = {k: v for k, v in configs.items() if k != "lambda_pair"}
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(cfg_to_save)
            fout.write(data)
        logger.info(f"Configuration saved to {saved_config_path}")

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        if use_pruning:
            header += ["loss_cls", "loss_reg", "spa_tgt", "spa_cur"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        run_epoch(train_dataloader,
                  epoch_iter,
                  ddp_model,
                  criterion,
                  (optimizer, optimizer_reg) if use_pruning else optimizer,
                  scheduler,
                  margin_scheduler,
                  epoch,
                  logger,
                  scaler,
                  device=device,
                  configs=configs,
                  swa=swa)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch > configs[
                    'num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))

    # Apply SWA at the end of training
    if swa is not None and rank == 0:
        logger.info("Applying SWA to final model...")
        swa.apply_swa()
        save_checkpoint(model, os.path.join(model_dir, 'model_swa.pt'))
        logger.info("SWA model saved as model_swa.pt")

    if rank == 0:
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   os.path.join(model_dir, 'final_model.pt'))
        logger.info(tp.bottom(len(header), width=10, style='grid'))


if __name__ == '__main__':
    fire.Fire(train)
