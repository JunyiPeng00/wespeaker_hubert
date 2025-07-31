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

from wespeaker.frontend.wav2vec2.model import wav2vec2_model
from wespeaker.frontend.wav2vec2.utils.import_huggingface_wavlm import import_huggingface_model

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[ %(levelname)s : %(asctime)s ] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

class HuggingfaceFrontend(nn.Module):
    """Speech Pretrained Representation Frontend."""

    def __init__(self,
                 upstream_args: dict,
                 download_dir: str = "./convert/wavlm_base_plus_hf/",
                 frozen: bool = False,
                 frame_shift: int = 20,
                 frame_length: int = 20,
                 sample_rate: int = 16000):
        super().__init__()

        self.upstream_name = upstream_args["name"].lower()
        self.frozen = frozen
        
        self.upstream, self.upstream_config = self.build_upstream(
            upstream_args.get("path_or_url", None),
            upstream_args.get("pruning_units", "conv,head,attlayer,interm,ffnlayer")
        )

        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            for name, param in self.upstream.named_parameters():
                if "mask_emb" in name:
                    param.requires_grad_(False)
    
    def get_num_params(self):
        return self.upstream.get_num_params()

    
    def build_upstream(self, upstream_ckpt, pruning_units):
        """Builds the upstream model from a checkpoint."""
        upstream_ckpt = torch.load(upstream_ckpt, map_location="cpu")
        upstream_config = upstream_ckpt['config']
        pruning_units = pruning_units.split(",")
        logger.info(f"Pruning units: {pruning_units}")
        upstream_config.update(
            dict(
                extractor_prune_conv_channels = "conv" in pruning_units,
                encoder_prune_attention_heads = "head" in pruning_units,
                encoder_prune_attention_layer = "attlayer" in pruning_units,
                encoder_prune_feed_forward_intermediate = "interm" in pruning_units,
                encoder_prune_feed_forward_layer = "ffnlayer" in pruning_units,
            )
        )
        upstream = wav2vec2_model(**upstream_config)
        upstream_result = upstream.load_state_dict(upstream_ckpt['state_dict'], strict=False)
        logger.info(f"Loaded pretrained ckpt to upstream: missing {upstream_result.missing_keys}, unexpected {upstream_result.unexpected_keys}") 
        return upstream, upstream_config

    def prune(self):
        return self.upstream.prune()


    def output_size(self):
        if "large" in self.upstream_name or "xlsr" in self.upstream_name:
            return 1024
        elif "base" in self.upstream_name:
            return 768
        elif self.upstream_name == "xls_r_300m" or self.upstream_name == "xls_r_1b":
            return 1024
        elif self.upstream_name == "xls_r_2b":
            return 1920
        else: #TODO for other models, 
            raise ValueError(f"Unknown model size for: {self.upstream_name}")


    def forward(self, input_wav: torch.Tensor, input_lengths: torch.LongTensor):
        max_input_length = 120 * 16000 
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            input_ = input_wav[:, :max_input_length]
            ssl_hiddens, _ = self.upstream.extract_features(input_)
            
        layer_reps = [x for x in ssl_hiddens]
        layer_reps = torch.stack(layer_reps).permute(1, 3, 2, 0)
        # print(f"ssl_hiddens: {layer_reps.shape}")
        return layer_reps, None


if __name__ == "__main__":
    upstream_args= {
        "name": "wavlm_base",
        "path_or_url": "/scratch/project_465001402/junyi/sv/ICASSP26_Pruning/wespeaker_hubert/examples/voxceleb/v4_ssl_pruning/exp/MHFA_WavLM_Base_Plus-ft-s50-2/pruned_model/pytorch_model.bin",
        "pruning_units": ""
    }
    net = HuggingfaceFrontend(upstream_args)
    x = torch.randn(4, 32000)
    # print(net.get_num_params())
    print(net)
