
"""Utility functions for structured pruning operations.

This module provides in-place pruning functions for different layer types,
including linear layers, convolutional layers, and normalization layers.
"""

from typing import Union

import torch
import torch.nn as nn


def prune_linear_layer(layer, index: torch.LongTensor, dim: str) -> None:
    """Prune a linear layer in place by removing specified dimensions.
    
    Args:
        layer: The linear layer to prune (nn.Linear or QuantizedLinear).
        index: Indices of dimensions to keep.
        dim: Dimension to prune ("input" or "output").
        
    Raises:
        ValueError: If dim is not "input" or "output".
    """
    # Handle QuantizedLinear layers
    if hasattr(layer, 'linear') and hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
        # This is a QuantizedLinear layer
        actual_layer = layer.linear
        layer.in_features = len(index) if dim == "input" else layer.in_features
        layer.out_features = len(index) if dim == "output" else layer.out_features
        
        # Quantization support removed - LSQ quantization is disabled
    else:
        # This is a regular nn.Linear layer
        actual_layer = layer
        if dim == "input":
            layer.in_features = len(index)
        elif dim == "output":
            layer.out_features = len(index)
    
    # NOTE: weight shape is (out_features, in_features), bias shape is (out_features,)
    if dim == "input":
        dim = 1
    elif dim == "output":
        dim = 0
    else:
        raise ValueError(f"Invalid dimension '{dim}'. Must be 'input' or 'output'.")

    # Prune weights and bias
    actual_layer.weight = nn.Parameter(actual_layer.weight.index_select(dim, index).clone().detach())
    if actual_layer.bias is not None and dim == 0:
        actual_layer.bias = nn.Parameter(actual_layer.bias.index_select(0, index).clone().detach())


def prune_conv1d_layer(layer, index: torch.LongTensor, dim: str) -> None:
    """Prune a 1D convolutional layer in place by removing specified channels.
    
    Args:
        layer: The 1D convolutional layer to prune (nn.Conv1d or QuantizedConv1d).
        index: Indices of channels to keep.
        dim: Dimension to prune ("input" or "output").
        
    Raises:
        ValueError: If dim is not "input" or "output".
    """
    # Handle QuantizedConv1d layers
    if hasattr(layer, 'conv1d') and hasattr(layer, 'in_channels') and hasattr(layer, 'out_channels'):
        # This is a QuantizedConv1d layer
        actual_layer = layer.conv1d
        layer.in_channels = len(index) if dim == "input" else layer.in_channels
        layer.out_channels = len(index) if dim == "output" else layer.out_channels
        
        # Quantization support removed - LSQ quantization is disabled
    else:
        # This is a regular nn.Conv1d layer
        actual_layer = layer
        if dim == "input":
            layer.in_channels = len(index)
        elif dim == "output":
            layer.out_channels = len(index)
    
    # NOTE: weight shape is (out_channels, in_channels, kernel_size), bias shape is (out_channels,)
    if dim == "input":
        dim = 1
    elif dim == "output":
        dim = 0
    else:
        raise ValueError(f"Invalid dimension '{dim}'. Must be 'input' or 'output'.")

    # Prune weights and bias
    actual_layer.weight = nn.Parameter(actual_layer.weight.index_select(dim, index).clone().detach())
    if actual_layer.bias is not None and dim == 0:
        actual_layer.bias = nn.Parameter(actual_layer.bias.index_select(0, index).clone().detach())


def prune_layer_norm(
    layernorm, 
    index: torch.LongTensor
) -> None:
    """Prune a layer normalization or group normalization layer in place.
    
    Args:
        layernorm: The normalization layer to prune.
        index: Indices of features to keep.
    """
    # Handle QuantizedLayerNorm layers
    if hasattr(layernorm, 'layer_norm'):
        # This is a QuantizedLayerNorm layer
        actual_layernorm = layernorm.layer_norm
        layernorm.normalized_shape = (len(index),)
    # Handle QuantizedGroupNorm layers
    elif hasattr(layernorm, 'group_norm'):
        # This is a QuantizedGroupNorm layer
        actual_layernorm = layernorm.group_norm
        layernorm.num_channels = len(index)
    else:
        # This is a regular normalization layer
        actual_layernorm = layernorm
    
    # Prune weight and bias parameters
    actual_layernorm.weight = nn.Parameter(actual_layernorm.weight.index_select(0, index).clone().detach())
    actual_layernorm.bias = nn.Parameter(actual_layernorm.bias.index_select(0, index).clone().detach())
    
    # Update layer-specific attributes
    if isinstance(actual_layernorm, nn.LayerNorm):
        actual_layernorm.normalized_shape = (len(index),)
    elif isinstance(actual_layernorm, nn.GroupNorm):
        actual_layernorm.num_groups = len(index)
        actual_layernorm.num_channels = len(index)
