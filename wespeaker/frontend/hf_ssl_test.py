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

from wav2vec2.model import wav2vec2_model
from wav2vec2.utils.import_huggingface_wavlm import import_huggingface_model

import logging
import time

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


    def forward(self, input_wav: torch.Tensor):
        max_input_length = 120 * 16000 
        with torch.no_grad() if self.frozen else contextlib.nullcontext():
            input_ = input_wav[:, :max_input_length]
            ssl_hiddens, _ = self.upstream.extract_features(input_)
            
        layer_reps = [x for x in ssl_hiddens]
        layer_reps = torch.stack(layer_reps).permute(1, 3, 2, 0)
        # print(f"ssl_hiddens: {layer_reps.shape}")
        return layer_reps, None


if __name__ == "__main__":
    from deepspeed.profiling.flops_profiler import get_model_profile

    upstream_args= {
        "name": "wavlm_base",
        # "path_or_url": "/scratch/project_465001402/junyi/sv/ICASSP26_Pruning/wespeaker_hubert/examples/voxceleb/v4_ssl_pruning/exp/MHFA_WavLM_Base_Plus-ft-s30-2/pruned_model/pytorch_model.bin",
        # "path_or_url": "/scratch/project_465001402/junyi/sv/ICASSP26_Pruning/wespeaker_hubert/examples/voxceleb/v4_ssl_pruning/exp/CNCeleb/MHFA_WavLM_Base_Plus-ft-s30-2/pruned_model/pytorch_model.bin",  # Use local path or URL to the model
        # "path_or_url": "/scratch/project_465001402/junyi/sv/DiariZen_dev_sv/recipes/diar_ssl_pruning/exp/baseline_librispeech/s50_base_lumi_ls_further_distill/avg_student_ckpt/pytorch_model.bin",
        # "path_or_url": "/scratch/project_465001402/junyi/sv/DiariZen_dev_sv/recipes/diar_ssl_pruning/exp/baseline_asv5/s80_base_lumi_further_distill/avg_student_ckpt/pytorch_model.bin",
        # "path_or_url": "/scratch/project_465001402/junyi/data/hf_models/others/pytorch_model.bin",
        # "path_or_url": "/scratch/project_465001402/junyi/sv_anti/wedefense/egs/detection/spoofceleb/v15_mhfa_pruning/exp/pruning_v0/MHFA_wavlm_s30/pruned_model/pytorch_model.bin",
        "path_or_url": "/scratch/project_465001402/junyi/sv/DiariZen_dev_sv/recipes/diar_ssl_pruning/convert/wavlm-base/wavlm-base.hf.pth",
        # "path_or_url": "/scratch/project_465001402/junyi/sv_anti/wedefense/egs/detection/spoofceleb/v15_mhfa_pruning/exp/pruning_v0/MHFA_wavlm_s50/pruned_model/pytorch_model.bin",
        # "path_or_url": "/scratch/project_465001402/junyi/sv_anti/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa_pruning/exp/pruning_v0_t1/MHFA_wavlm_s50/pruned_model/pytorch_model.bin",
        "pruning_units": ""
    }
    model = HuggingfaceFrontend(upstream_args)
    

    def count_parameters(module):
        """统计一个子模块的可训练参数；若 module 为 None 返回 0。"""
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    # ----------------- 3. 统计 feature_extractor -----------------
    fe_params = count_parameters(model.upstream.feature_extractor)
    print(f"feature_extractor: {fe_params:,} 参数")

    # ----------------- 4. 统计每个 EncoderLayer 内部 -----------------
    attn_counts, ff_counts = [], []
    for idx, layer in enumerate(model.upstream.encoder.transformer.layers):
        attn = layer.attention                # WavLMSelfAttention
        ff   = getattr(layer, "feed_forward", None)

        attn_cnt = count_parameters(attn)
        ff_cnt   = count_parameters(ff) if ff is not None else 0

        attn_counts.append(attn_cnt)
        ff_counts.append(ff_cnt)

        print(f"Layer {idx:02d}  |  "
            f"SelfAttention: {attn_cnt:,}  |  FeedForward: {ff_cnt:,}")

    # ----------------- 5. 汇总总参数量 -----------------
    total_params = count_parameters(model)
    print(f"\n=== 总参数量：{total_params:,} ===")

    model.eval().cuda()
    flops, macs, params  = get_model_profile(
        model=model,
        input_shape=(1, 16000*4),  # 根据你模型的输入更改
        print_profile=True,
        detailed=True,                 # 打印每一层
        module_depth=-1,               # 控制打印深度
    )
    print(f"flops: {flops}, macs; {macs}, Params: {params}")

    # --- GPU Inference Performance ---
    if torch.cuda.is_available():
        print("\n--- Measuring GPU Inference Time ---")
        model.eval().cuda()
        # Using a smaller batch size for the dummy input to prevent potential OOM errors
        dummy_input_gpu = torch.randn(16, 16000 * 20).cuda()
        with torch.no_grad():
            # Warm-up iterations
            for _ in range(10):
                _ = model(dummy_input_gpu)

            torch.cuda.synchronize() # Wait for warm-up to finish
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(dummy_input_gpu)
            end_event.record()

            torch.cuda.synchronize() # Wait for the forward pass to complete
            latency_ms_gpu = start_event.elapsed_time(end_event)
            print(f"Inference Latency (GPU): {latency_ms_gpu:.2f} ms")
        print("-" * 30)
    else:
        print("\nCUDA not available, skipping GPU performance measurement.")


    # --- CPU Inference Performance ---
    print("\n--- Measuring CPU Inference Time ---")
    model.eval().cpu()
    # Using a smaller batch size consistent with GPU test
    dummy_input_cpu = torch.randn(16, 16000 * 20)

    with torch.no_grad():
        # Warm-up iterations
        for _ in range(5):
            _ = model(dummy_input_cpu)

        start_time = time.perf_counter()
        _ = model(dummy_input_cpu)
        end_time = time.perf_counter()

        latency_ms_cpu = (end_time - start_time) * 1000
        print(f"Inference Latency (CPU): {latency_ms_cpu:.2f} ms")
    print("-" * 30)
    