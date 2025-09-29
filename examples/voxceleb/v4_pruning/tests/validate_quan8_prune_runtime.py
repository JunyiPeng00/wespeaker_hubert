#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8-bit 量化 + 剪枝 联合验证脚本（内存冒烟）：
- 构建极小线性网络模拟前端（或直接使用 WavLM SelfAttention/FFN）
- 通过 wespeaker.frontend.wav2vec2.quantization_utils.apply_quantization_with_hp_integration 应用 8-bit 量化
- 启用 HardConcrete 剪枝门控并运行温度退火、稀疏度调度、学习率退火、SWA
- 打印关键日志以确认二者联合可正常训练
运行：
  source /opt/miniconda3/bin/activate wedefense
  PYTHONPATH=/Users/pengjy/wespeaker_hubert python examples/voxceleb/v4_pruning/tests/validate_quan8_prune_runtime.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from wespeaker.frontend.wav2vec2.components import WavLMSelfAttention, FeedForward
from wespeaker.frontend.wav2vec2.hardconcrete import HardConcrete
from wespeaker.frontend.wav2vec2.quantization_utils import (
    get_quantization_config,
    apply_quantization_with_hp_integration,
)
from wespeaker.utils.prune_utils import (
    get_progressive_sparsity,
    get_learning_rate_with_plateau_decay,
    StochasticWeightAveraging,
)


def build_stack(embed_dim: int, heads: int) -> nn.Module:
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = WavLMSelfAttention(embed_dim, heads, remaining_heads=list(range(heads)), prune_heads=True, prune_layer=True)
            self.ffn = FeedForward(embed_dim, 4 * embed_dim, 0.0, 0.0, prune_intermediate=True, prune_layer=True)
        def forward(self, x):
            x, _ = self.attn(x)
            x = self.ffn(x)
            return x
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.stack = nn.ModuleList([Block(), Block()])
            self.proj = nn.Linear(embed_dim, 10)
        def forward(self, x):
            for m in self.stack:
                x = m(x)
            x = x.mean(dim=1)
            return self.proj(x)
    return Tiny()


def main() -> None:
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_dim = 256
    heads = 8
    model = build_stack(embed_dim, heads)

    # 8-bit 量化配置
    qcfg = get_quantization_config('8bit_symmetric')
    qcfg.quantize_weights = True
    qcfg.quantize_activations = False
    qcfg.per_channel_weights = True
    qcfg.per_channel_activations = False

    pruning_config = {
        'target_sparsity': 0.6,
        'sparsity_warmup_epochs': 6,
        'sparsity_schedule': 'cosine_plateau',
        'min_sparsity': 0.0,
        'plateau_start_ratio': 0.9,
        'min_temperature': 0.1,
        'temperature_decay': 0.95,
        'temperature_decay_freq': 50,
    }

    # 应用量化 + HP 剪枝集成（保留 HardConcrete 门控）
    model = apply_quantization_with_hp_integration(
        model,
        config=qcfg,
        enable_pruning=True,
        pruning_config=pruning_config,
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-7)

    total_iters = 500
    warmup_iters = 150
    target_sparsity = pruning_config['target_sparsity']

    swa = StochasticWeightAveraging(model, start_iter=int(total_iters*0.9), update_freq=10)

    print('== Validate 8-bit Quantization + Pruning ==')
    print(f'device={device}, embed_dim={embed_dim}, heads={heads}')

    for it in range(total_iters):
        model.train()

        # 温度退火
        for m in model.modules():
            if isinstance(m, HardConcrete):
                m.update_temperature(it)

        # 稀疏度目标
        if it < warmup_iters:
            tgt_sp = get_progressive_sparsity(
                current_iter=it,
                total_warmup_iters=warmup_iters,
                target_sparsity=target_sparsity,
                schedule_type=pruning_config['sparsity_schedule'],
                min_sparsity=pruning_config['min_sparsity'],
                total_iters=total_iters,
                plateau_start_ratio=pruning_config['plateau_start_ratio'],
            )
        else:
            tgt_sp = target_sparsity

        # 假输入
        x = torch.randn(8, 32, embed_dim, device=device)
        y = torch.randint(0, 10, (8,), device=device)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        # 末段学习率退火
        new_lr = get_learning_rate_with_plateau_decay(
            current_iter=it,
            total_iters=total_iters,
            initial_lr=2e-4,
            plateau_start_ratio=pruning_config['plateau_start_ratio'],
            decay_factor=0.1,
            min_lr_ratio=0.01,
        )
        for pg in optimizer.param_groups:
            pg['lr'] = new_lr

        # SWA
        swa.update(it)

        if (it+1) % 100 == 0 or it == 0:
            print(f'it={it:04d} loss={float(loss):.4f} tgt_sp={tgt_sp:.3f} lr={optimizer.param_groups[0]["lr"]:.2e}')

    swa.apply_swa()
    print('Applied SWA. ✓ 8-bit quantization + pruning validation passed.')


if __name__ == '__main__':
    main()
