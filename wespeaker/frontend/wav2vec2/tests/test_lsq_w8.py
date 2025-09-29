import torch
import torch.nn as nn

from wespeaker.frontend.wav2vec2.quantization_utils import (
    QuantizationConfig,
    apply_quantization,
    get_quantization_stats,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple MLP branch
        self.lin1 = nn.Linear(32, 64, bias=True)
        self.norm = nn.LayerNorm(64)
        self.lin2 = nn.Linear(64, 16, bias=True)
        # Simple Conv1d branch
        self.conv1 = nn.Conv1d(8, 8, 3, padding=1, bias=True)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=1, bias=True)

    def forward(self, x_feat, x_seq):
        # x_feat: (B, 32)
        # x_seq:  (B, 8, T)
        y = self.lin1(x_feat)
        y = torch.relu(self.norm(y))
        y = self.lin2(y)
        z = torch.relu(self.conv1(x_seq))
        z = self.conv2(z)
        return y, z


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    # Build float model and inputs
    net = TinyNet().to(device)
    x_feat = torch.randn(8, 32, device=device)
    x_seq = torch.randn(8, 8, 40, device=device)

    # Configure 8-bit symmetric, weights-only quantization to verify core path
    cfg = QuantizationConfig(
        weight_bits=8,
        bias_bits=8,
        activation_bits=8,
        weight_symmetric=True,
        bias_symmetric=True,
        activation_symmetric=True,
        quantize_weights=True,
        quantize_bias=False,
        quantize_activations=False,
        per_channel_weights=True,
        per_channel_activations=False,
    )

    # Apply quantization (out-of-place)
    qnet = apply_quantization(net, cfg, inplace=False).to(device)

    # Forward + backward one step to ensure gradients flow and optimizer updates
    opt = torch.optim.AdamW(qnet.parameters(), lr=1e-4)
    y, z = qnet(x_feat, x_seq)
    loss = (y.pow(2).mean() + z.pow(2).mean())
    loss.backward()
    opt.step()

    # Collect stats for sanity check
    stats = get_quantization_stats(qnet)
    assert 'total_parameters' in stats and stats['total_parameters'] > 0
    assert 'quantized_parameters' in stats and stats['quantized_parameters'] > 0

    # Print minimal info for CI/log inspection
    print('device=', device)
    print('loss=', float(loss.item()))
    print('quantized_parameters=', int(stats['quantized_parameters']))


if __name__ == '__main__':
    main()


