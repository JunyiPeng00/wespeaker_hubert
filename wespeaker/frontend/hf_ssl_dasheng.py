# Copyright (c) 2024 Junyi Peng (pengjy@fit.vut.cz)
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

import contextlib
import torch
import torch.nn as nn

from transformers import AutoModel, AutoFeatureExtractor
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[ %(levelname)s : %(asctime)s ] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

class HuggingfaceFrontend_Dasheng(nn.Module):
    """Speech Pretrained Representation Frontend."""

    def __init__(self,
                 upstream_args: dict,
                 download_dir: str = "./convert/wavlm_base_plus_hf/",
                 frozen: bool = True,
                 cnn_frozen: bool = False,
                 frame_shift: int = 10,
                 frame_length: int = 20,
                 sample_rate: int = 16000):
        super().__init__()

        self.upstream_name = upstream_args["name"] # "mispeech/dasheng-0.6B"
        self.frozen = frozen

        # model_name = "mispeech/dasheng-0.6B"
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.upstream_name, cache_dir=download_dir, trust_remote_code=True)
        self.upstream = AutoModel.from_pretrained(self.upstream_name, cache_dir=download_dir, trust_remote_code=True)


        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            for name, param in self.upstream.named_parameters():
                freeze = False
                if cnn_frozen and ('feature_extractor' in name or 'feature_projection' in name):
                    freeze = True
                if "mask_emb" in name:
                    freeze = True
                if freeze:
                    param.requires_grad_(False)

    
    def get_num_params(self):
        total_params = sum(p.numel() for p in self.upstream.parameters())
        return total_params

    def output_size(self):
        if "0.6" in self.upstream_name:
            return 1280
        elif "base" in self.upstream_name:
            return 768
        elif "1.2" in self.upstream_name:
            return 1536
        else: #TODO for other models, 
            raise ValueError(f"Unknown model size for: {self.upstream_name}")


    def forward(self, input_wav, input_lengths):
        max_input_length = 120 * 16000
        hidden_states = []

        def hook_fn(module, input, output):
            hidden_states.append(output)

        handles = [
            blk.register_forward_hook(hook_fn)
            for blk in self.upstream.encoder.blocks
        ]

        feats = self.feature_extractor(
            input_wav[:, :max_input_length].cpu(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        device = next(self.upstream.parameters()).device
        inputs = {k: v.to(device) for k, v in feats.items()}

        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            _ = self.upstream(**inputs)

        for handle in handles:
            handle.remove()

        layer_reps = torch.stack(hidden_states).permute(1, 3, 2, 0)  # [B, D, T, L]
        return layer_reps, None


if __name__ == "__main__":
    upstream_args= {
        "name": "mispeech/dasheng-0.6B",
        "path_or_url": "/scratch/project_465001402/junyi/sv/ICASSP26_Pruning/wespeaker_hubert/examples/voxceleb/v4_ssl_pruning/exp/MHFA_WavLM_Base_Plus-ft-s50-2/pruned_model/pytorch_model.bin",
    }
    net = HuggingfaceFrontend_Dasheng(upstream_args)
    x = torch.randn(4, 32000)
    # print(net.get_num_params())
    print(net)
    output,_ = net(x,None)
    print(output.shape)

