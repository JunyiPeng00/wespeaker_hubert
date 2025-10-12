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

"""Frontend for speech feature extraction using Hugging Face pretrained models."""

import contextlib
import os
from typing import Mapping, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from wespeaker.frontend.wav2vec2.convert_hf_ssl_models import convert_hf_ssl_model
from wespeaker.frontend.wav2vec2.model import wav2vec2_model, wavlm_model

def is_rank_zero() -> bool:
    """Check if current process is rank 0 in distributed training.
    
    Returns:
        True if current process is rank 0 or not in distributed training.
    """
    # First check environment variable (for manual testing)
    rank_env = os.environ.get('RANK')
    if rank_env is not None:
        return int(rank_env) == 0
    
    # Then check distributed training
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True  # If not distributed, always log
    return torch.distributed.get_rank() == 0


class HuggingfaceFrontend(nn.Module):
    """Wraps a Hugging Face pretrained model for speech feature extraction.

    This module handles the download, conversion, and loading of a specified
    Hugging Face WavLM/XLS-R model. It serves as a feature extractor frontend
    and supports optional freezing of weights and model pruning.

    Attributes:
        upstream_name: The name of the upstream model (e.g., 'wavlm_large').
        frozen: A boolean indicating if the upstream model weights are frozen.
        upstream: The loaded pretrained model instance.
        upstream_config: The configuration dictionary of the upstream model.
    """

    # Maximum input length in samples (120 seconds * 16kHz)
    _MAX_INPUT_SAMPLES = 120 * 16000

    def __init__(
        self,
        upstream_args: Mapping[str, Any],
        download_dir: str = './hf_models/',
        frozen: bool = False,
        frame_shift: int = 20,
        frame_length: int = 20,
        sample_rate: int = 16000,
    ):
        """Initializes the HuggingfaceFrontend.

        Args:
            upstream_args: A dictionary containing configuration for the upstream
                model. Must include 'name' (e.g., 'wavlm_base_plus'). It can
                also include 'pruning_units'.
            download_dir: The directory to save downloaded and converted
                Hugging Face models.
            frozen: If True, the model parameters are frozen and not trained.
        """
        super().__init__()

        self.upstream_name = upstream_args['name'].lower()
        download_dir = upstream_args['path_or_url']
        self.frozen = frozen

        # 1. If download_dir ends with '.bin' or '.pth', treat it as a pre-converted
        #    model path. Otherwise, download and convert the model.
        if download_dir.endswith(('.bin', '.pth')):
            if is_rank_zero():
                print(f"Path ends with {download_dir.split('.')[-1]}, assuming it is a pre-converted model: {download_dir}")
            converted_model_path = download_dir
        else:
            if is_rank_zero():
                print(f"Starting model download and conversion for {self.upstream_name}.")
            # This function returns the path to the converted checkpoint and config.
            converted_model_path, _ = convert_hf_ssl_model(
                model_name=self.upstream_name,
                exp_dir=download_dir,
                hf_cache_dir=download_dir,
                local_files_only=False,
            )
            # print(f"converted_model_path: {converted_model_path}")

        # 2. Build the upstream model from the newly converted checkpoint.
        pruning_units = upstream_args.get('pruning_units', '')
        self.upstream, self.upstream_config = self._build_upstream(
            upstream_ckpt_path=converted_model_path,
            pruning_units=pruning_units,
        )

        # 3. Freeze weights if required.
        if self.frozen:
            for param in self.upstream.parameters():
                param.requires_grad_(False)
        else:
            # By default, do not train the codebook embeddings.
            for name, param in self.upstream.named_parameters():
                if 'mask_emb' in name:
                    param.requires_grad_(False)
            self.upstream.train()

    def _build_upstream(
        self, upstream_ckpt_path: str, pruning_units: str
    ) -> Tuple[nn.Module, Mapping[str, Any]]:
        """Builds the upstream model from a WeSpeaker format checkpoint.

        Args:
            upstream_ckpt_path: Path to the WeSpeaker format checkpoint (.pth).
            pruning_units: A comma-separated string specifying parts of the
                model to prune (e.g., "head,ffnlayer").

        Returns:
            A tuple containing:
                - The loaded upstream model instance.
                - The configuration dictionary for the model.
        """
        ckpt = torch.load(upstream_ckpt_path, map_location='cpu')
        config = ckpt['config']
        pruning_set = set(p.strip() for p in pruning_units.split(',') if p)
        if is_rank_zero():
            print(f'Enabled pruning units: {pruning_set}')

        config.update({
            'extractor_prune_conv_channels': 'conv' in pruning_set,
            'encoder_prune_attention_heads': 'head' in pruning_set,
            'encoder_prune_attention_layer': 'attlayer' in pruning_set,
            'encoder_prune_feed_forward_intermediate': 'interm' in pruning_set,
            'encoder_prune_feed_forward_layer': 'ffnlayer' in pruning_set,
        })

        # Ensure required parameters are present
        if 'aux_num_out' not in config:
            config['aux_num_out'] = None
        if 'normalize_waveform' not in config:
            config['normalize_waveform'] = False

        # Determine which model class to use based on the model name
        if "wavlm" in self.upstream_name.lower():
            model = wavlm_model(**config)
        else:
            model = wav2vec2_model(**config)
        
        result = model.load_state_dict(ckpt['state_dict'], strict=False)
        if is_rank_zero():
            print(f'[{self.upstream_name}] Loaded pretrained ckpt to upstream: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')
        return model, config

    def output_size(self) -> int:
        """Returns the output feature dimension of the model.

        Raises:
            ValueError: If the model name is unknown.

        Returns:
            The integer size of the output dimension.
        """
        # Check if we have config information from the loaded model
        if hasattr(self, 'upstream_config') and 'encoder_embed_dim' in self.upstream_config:
            return self.upstream_config['encoder_embed_dim']
        
        # Fallback to model name-based logic
        if 'large' in self.upstream_name or 'xlsr' in self.upstream_name:
            return 1024
        if 'base' in self.upstream_name:
            return 768
        if self.upstream_name in ('xls_r_300m', 'xls_r_1b'):
            return 1024
        if self.upstream_name == 'xls_r_2b':
            return 1920
        
        # For data2vec models
        if 'data2vec' in self.upstream_name:
            if 'large' in self.upstream_name:
                return 1024
            elif 'base' in self.upstream_name:
                return 768
                
        raise ValueError(f'Unknown output size for model: {self.upstream_name}')

    def forward(
        self, input_wav: torch.Tensor, input_lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """Performs the forward pass to extract features.

        Args:
            input_wav: A batch of input waveforms, shape (B, T).
            input_lengths: A batch of waveform lengths, shape (B,). This
                argument is unused but maintained for API compatibility.

        Returns:
            A tuple containing:
                - The extracted features, shape (B, D, F, L), where D is the
                  feature dimension, F is the number of frames, and L is the
                  number of layers.
                - None, for API compatibility.
        """
        # Ensure model is not running on excessively long inputs
        input_tensor = input_wav[:, : self._MAX_INPUT_SAMPLES]

        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            # ssl_hiddens is a tuple of tensors, one for each layer
            ssl_hiddens, _ = self.upstream.extract_features(input_tensor)

        # Stack layer representations and reorder dimensions
        # Original: (L, B, F, D) -> Stacked: (L, B, F, D)
        # Permuted: (B, D, F, L) for downstream convenience
        layer_reps = torch.stack(ssl_hiddens, dim=0).permute(1, 3, 2, 0)
        return layer_reps, None

    def get_num_params(self) -> int:
        """Returns the total number of parameters in the upstream model."""
        return self.upstream.get_num_params()

    def prune(self) -> nn.Module:
        """Applies pruning to the upstream model."""
        return self.upstream.prune()


def main():
    """Instantiates and tests the HuggingfaceFrontend."""
    # Check if we're in distributed training
    rank = 0
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    # This example config demonstrates support for various SSL models.
    # The conversion logic will handle downloading and converting the model.
    upstream_config = {
        'name': 'wavlm_base_plus',  # Can be: hubert_base, wav2vec2_base, wavlm_base, etc.
        # 'path_or_url': '/scratch/project_465002053/junyi/sv/wespeaker_dev/wespeaker_hubert/examples/voxceleb/v4_pruning/convert/wavlm_base.hf.pth',  # Directory for model cache
        # 'path_or_url': '/scratch/project_465002053/junyi/sv/wespeaker_dev/wespeaker_hubert/examples/voxceleb/v4_pruning/exp/pruning/mhfa_WavLMBasePlus_p70_e/pruned_model/pytorch_model.bin',  # Directory for model cache
        'path_or_url':'/scratch/project_465002053/junyi/sv/wespeaker_dev/wespeaker_hubert/examples/voxceleb/v4_pruning/exp/qua2pruning/mhfa_WavLMBasePlus_p70_qua/pruned_model/pytorch_model.bin',
        'pruning_units': '',  # No pruning in this example
    }
    
    if is_rank_zero():
        print(f'Initializing HuggingfaceFrontend with config: {upstream_config}')
        if world_size > 1:
            print(f'Running in distributed mode: rank={rank}, world_size={world_size}')

    # Note: This will trigger the download and conversion of the specified model
    # if it's not already present in './hf_models/'.
    net = HuggingfaceFrontend(
        upstream_args=upstream_config, download_dir='./hf_models/'
    )

    dummy_input = torch.randn(4, 32000)  # Batch of 4, 2 seconds of audio
    dummy_lengths = torch.LongTensor([32000] * 4)

    if is_rank_zero():
        print('Model initialized successfully.')
        print(f'Number of parameters: {net.get_num_params():,}')
        # print(net)  # Uncomment to see model architecture

    if is_rank_zero():
        print('Testing forward pass...')
    
    output, _ = net(dummy_input, dummy_lengths)
    
    from deepspeed.profiling.flops_profiler import get_model_profile
    
    flops, macs, params  = get_model_profile(
        model=net.eval(),
        input_shape=(1, 16000*4),  # 根据你模型的输入更改
        print_profile=True,
        detailed=True,                 # 打印每一层
        module_depth=-1,               # 控制打印深度
    )
    print(f"flops: {flops}, macs; {macs}, Params: {params}")

    if is_rank_zero():
        print(f'Output shape: {output.shape}')


if __name__ == '__main__':
    main()