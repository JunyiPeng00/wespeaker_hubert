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

# Quantization modules
from .lsq_quantizer import (
    LSQQuantizer,
    LSQQuantizerFunction,
    create_lsq_quantizer,
    LSQ_8BIT_SYMMETRIC,
    LSQ_8BIT_ASYMMETRIC,
    LSQ_4BIT_SYMMETRIC,
    LSQ_4BIT_ASYMMETRIC
)
from .quantized_layers import (
    QuantizedLinear,
    QuantizedConv1d,
    QuantizedLayerNorm,
    QuantizedGroupNorm,
    convert_linear_to_quantized,
    convert_conv1d_to_quantized
)
from .quantization_utils import (
    QuantizationConfig,
    quantize_model,
    get_quantization_stats,
    compare_model_sizes,
    get_quantization_config,
    apply_quantization,
    apply_quantization_with_hp_integration,
    QUANTIZATION_CONFIGS
)

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
    
    # Quantization components
    'LSQQuantizer',
    'LSQQuantizerFunction',
    'create_lsq_quantizer',
    'LSQ_8BIT_SYMMETRIC',
    'LSQ_8BIT_ASYMMETRIC',
    'LSQ_4BIT_SYMMETRIC',
    'LSQ_4BIT_ASYMMETRIC',
    
    # Quantized layers
    'QuantizedLinear',
    'QuantizedConv1d',
    'QuantizedLayerNorm',
    'QuantizedGroupNorm',
    'convert_linear_to_quantized',
    'convert_conv1d_to_quantized',
    
    # Quantization utilities
    'QuantizationConfig',
    'quantize_model',
    'get_quantization_stats',
    'compare_model_sizes',
    'get_quantization_config',
    'apply_quantization',
    'apply_quantization_with_hp_integration',
    'QUANTIZATION_CONFIGS'
]
