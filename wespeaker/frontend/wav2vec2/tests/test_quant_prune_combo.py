import torch
import torch.nn as nn

from wespeaker.frontend.wav2vec2.quantization_utils import (
    QuantizationConfig,
    apply_quantization_with_hp_integration,
    get_quantization_stats,
)
from wespeaker.utils.prune_utils import get_progressive_sparsity, pruning_loss


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(8, 16, 3, padding=1)
        self.norm = nn.GroupNorm(4, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.lin = nn.Linear(16, 8)

    def forward(self, x):
        # x: (B, 8, T)
        y = torch.relu(self.conv(x))
        y = self.norm(y)
        y = self.pool(y).squeeze(-1)
        y = self.lin(y)
        return y


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    # Build tiny model and dummy classifier head
    backbone = TinyBackbone().to(device)
    head = nn.Linear(8, 4).to(device)
    model = nn.Sequential(backbone, head).to(device)

    # Quantization config: 8-bit weights only
    qcfg = QuantizationConfig(
        weight_bits=8,
        activation_bits=8,
        quantize_weights=True,
        quantize_bias=False,
        quantize_activations=False,
        per_channel_weights=True,
    )

    # Apply quantization with HP integration (pruning enabled)
    pruning_cfg = {
        'target_sparsity': 0.5,
        'sparsity_warmup_epochs': 1,
        'sparsity_schedule': 'cosine',
        'min_sparsity': 0.0,
    }
    qmodel = apply_quantization_with_hp_integration(model, qcfg, enable_pruning=True, pruning_config=pruning_cfg).to(device)

    # Dummy data
    x = torch.randn(6, 8, 24, device=device)
    y_true = torch.randint(0, 4, (6,), device=device)

    # Optimizer
    opt = torch.optim.AdamW(qmodel.parameters(), lr=1e-4)
    ce = nn.CrossEntropyLoss()

    # Simulate one epoch with pruning regularization
    total_params = float(sum(p.numel() for p in qmodel.parameters()))
    warmup_iters = 5
    for step in range(10):
        logits = qmodel(x)
        cls_loss = ce(logits, y_true)

        # Progressive sparsity target
        tgt_sp = get_progressive_sparsity(step, warmup_iters, target_sparsity=0.5, schedule_type='cosine', min_sparsity=0.0)
        # Use total params as a proxy for current params in this smoke test
        reg, exp_sp = pruning_loss(current_params=total_params, original_params=total_params, target_sparsity=tgt_sp, lambda1=torch.tensor(1.0, device=logits.device), lambda2=torch.tensor(5.0, device=logits.device))

        loss = cls_loss + reg
        opt.zero_grad()
        loss.backward()
        opt.step()

    stats = get_quantization_stats(qmodel)
    print('device=', device)
    print('loss=', float(loss.item()))
    print('quantized_modules=', len(stats.get('quantized_modules', [])))


if __name__ == '__main__':
    main()



