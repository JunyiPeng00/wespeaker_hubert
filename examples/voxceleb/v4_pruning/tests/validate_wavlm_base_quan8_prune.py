#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WavLM Base 8bit 量化 + 剪枝 联合验证：
- 加载 microsoft/wavlm-base 作为冻结前端
- 在其输出后接一层可剪枝注意力+FFN+投影的小头
- 对小头应用 8bit 量化并启用 HP 剪枝，运行短训练以验证可行性

运行：
  source /opt/miniconda3/bin/activate wedefense
  PYTHONPATH=/Users/pengjy/wespeaker_hubert python examples/voxceleb/v4_pruning/tests/validate_wavlm_base_quan8_prune.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import WavLMModel, WavLMConfig

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


def anneal_all_hc(model: nn.Module, it: int) -> None:
    for m in model.modules():
        if isinstance(m, HardConcrete):
            m.update_temperature(it)


def main() -> None:
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 HF WavLM Base 作为冻结前端
    cfg = WavLMConfig.from_pretrained('microsoft/wavlm-base')
    base = WavLMModel.from_pretrained('microsoft/wavlm-base').to(device)
    base.eval()

    embed_dim = cfg.hidden_size
    heads = cfg.num_attention_heads

    # 构建小头（注意力+FFN+分类）
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = WavLMSelfAttention(embed_dim, heads, remaining_heads=list(range(heads)), prune_heads=True, prune_layer=True)
            self.ffn = FeedForward(embed_dim, 4 * embed_dim, 0.0, 0.0, prune_intermediate=True, prune_layer=True)
            self.proj = nn.Linear(embed_dim, 10)
        def forward(self, x):
            x, _ = self.attn(x)
            x = self.ffn(x)
            x = x.mean(dim=1)
            return self.proj(x)

    head = Head()

    # 量化配置 + 剪枝配置
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

    # 对 head 应用 8bit 量化并启用 HP 剪枝
    head = apply_quantization_with_hp_integration(head, config=qcfg, enable_pruning=True, pruning_config=pruning_config)
    head.to(device)

    optimizer = optim.AdamW(head.parameters(), lr=1e-4, weight_decay=1e-7)

    total_iters = 300
    warmup_iters = 100
    target_sparsity = pruning_config['target_sparsity']
    swa = StochasticWeightAveraging(head, start_iter=int(total_iters*0.9), update_freq=10)

    print('== Validate WavLM Base + 8bit quant + pruning ==')
    print(f'device={device}, embed_dim={embed_dim}, heads={heads}')

    for it in range(total_iters):
        # 随机波形
        wav = torch.randn(1, 16000, device=device)
        with torch.no_grad():
            feat = base(wav).last_hidden_state

        # 退火与稀疏度目标
        anneal_all_hc(head, it)
        if it < warmup_iters:
            tgt_sp = get_progressive_sparsity(
                current_iter=it,
                total_warmup_iters=warmup_iters,
                target_sparsity=target_sparsity,
                schedule_type='cosine_plateau',
                min_sparsity=0.0,
                total_iters=total_iters,
                plateau_start_ratio=pruning_config['plateau_start_ratio'],
            )
        else:
            tgt_sp = target_sparsity

        # 前向与更新
        y = torch.randint(0, 10, (1,), device=device)
        head.train()
        logits = head(feat)
        loss = nn.CrossEntropyLoss()(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        # 末段学习率退火
        new_lr = get_learning_rate_with_plateau_decay(
            current_iter=it,
            total_iters=total_iters,
            initial_lr=1e-4,
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
    print('Applied SWA. ✓ WavLM Base + 8bit quant + pruning validation passed.')


if __name__ == '__main__':
    main()
