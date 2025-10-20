"""Wav2Vec2 frontend with quantization support.

This module provides wav2vec2 models with integrated LSQ (Learnable Step Size Quantization)
for efficient inference and training.
"""

from .model import Wav2Vec2Model, wav2vec2_model, wavlm_base, wavlm_large, wavlm_model
from .components import (
    FeatureExtractor,
    FeatureProjection,
    SelfAttention,
    WavLMSelfAttention,
    FeedForward,
    EncoderLayer,
    Transformer,
    Encoder
)
from .hardconcrete import HardConcrete
from .pruning_utils import (
    prune_linear_layer,
    prune_conv1d_layer,
    prune_layer_norm
)

# Quantization modules removed - LSQ quantization is disabled

__all__ = [
    # Core wav2vec2 components
    'Wav2Vec2Model',
    'wav2vec2_model',
    'wavlm_base',
    'wavlm_large', 
    'wavlm_model',
    'FeatureExtractor',
    'FeatureProjection',
    'SelfAttention',
    'WavLMSelfAttention',
    'FeedForward',
    'EncoderLayer',
    'Transformer',
    'Encoder',
    
    # Pruning components
    'HardConcrete',
    'prune_linear_layer',
    'prune_conv1d_layer',
    'prune_layer_norm',
    
    # Quantization components removed - LSQ quantization is disabled
]
