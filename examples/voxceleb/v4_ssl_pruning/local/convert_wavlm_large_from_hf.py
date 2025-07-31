"""Convert Hugging Face's WavLM to our format."""

import os

import torch
from transformers import WavLMModel

from wespeaker.frontend.wav2vec2.model import wav2vec2_model
from wespeaker.frontend.wav2vec2.utils.import_huggingface_wavlm import import_huggingface_model
import os

os.environ["HF_HOME"] = "/scratch/project_465001402/junyi/data/hf_models/" 

wavlm_dir = r'/scratch/project_465001402/junyi/data/hf_models/wavlm-large'
exp_dir = r'/scratch/project_465001402/junyi/sv/ICASSP26_Pruning/wespeaker_hubert/examples/voxceleb/v4_ssl_pruning/convert'

if __name__ == "__main__":
    os.makedirs(exp_dir, exist_ok=True)
    out_name = exp_dir + "/wavlm-large.hf.pth"

    original = WavLMModel.from_pretrained(wavlm_dir)
    imported, config = import_huggingface_model(original)
    imported.eval()
    print(imported)

    aux_dict = dict(
        aux_num_out=None,
        extractor_prune_conv_channels=False,
        encoder_prune_attention_heads=False,
        encoder_prune_attention_layer=False,
        encoder_prune_feed_forward_intermediate=False,
        encoder_prune_feed_forward_layer=False,
    )
    config.update(aux_dict)

    torch.save(
        {
            'state_dict': imported.state_dict(),
            'config': config,
        }, 
        out_name
    )

    # verify the saved ckpt
    ckpt = torch.load(out_name, map_location="cpu")
    model = wav2vec2_model(**ckpt['config'])
    print(model.load_state_dict(ckpt['state_dict'], strict=False))