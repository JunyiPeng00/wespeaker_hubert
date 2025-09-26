"""Quantized layer implementations for WeDefense.

This module provides quantized versions of common neural network layers
using LSQ (Learnable Step Size Quantization) for efficient inference.

Supported layers:
- QuantizedLinear: Quantized linear layer
- QuantizedConv1d: Quantized 1D convolutional layer
- QuantizedLayerNorm: Quantized layer normalization
- QuantizedGroupNorm: Quantized group normalization
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lsq_quantizer import LSQQuantizer, create_lsq_quantizer


class QuantizedLinear(nn.Module):
    """Quantized linear layer with LSQ quantization.
    
    This layer applies LSQ quantization to the weight matrix and bias vector
    before performing the linear transformation.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias term
        weight_quantizer: LSQ quantizer for weights (optional)
        bias_quantizer: LSQ quantizer for bias (optional)
        activation_quantizer: LSQ quantizer for activations (optional)
        quantize_weights: Whether to quantize weights (default: True)
        quantize_bias: Whether to quantize bias (default: True)
        quantize_activations: Whether to quantize activations (default: False)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_quantizer: Optional[LSQQuantizer] = None,
        bias_quantizer: Optional[LSQQuantizer] = None,
        activation_quantizer: Optional[LSQQuantizer] = None,
        quantize_weights: bool = True,
        quantize_bias: bool = True,
        quantize_activations: bool = False,
        weight_bits: int = 8,
        bias_bits: int = 8,
        activation_bits: int = 8,
        weight_symmetric: bool = True,
        bias_symmetric: bool = True,
        activation_symmetric: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.quantize_weights = quantize_weights
        self.quantize_bias = quantize_bias and bias  # Only quantize bias if it exists
        self.quantize_activations = quantize_activations
        
        # Create linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Initialize weight quantizer
        if quantize_weights:
            if weight_quantizer is not None:
                self.weight_quantizer = weight_quantizer
            else:
                # Prefer per-channel quantization for weights (better accuracy)
                self.weight_quantizer = create_lsq_quantizer(
                    num_bits=weight_bits,
                    symmetric=weight_symmetric,
                    per_channel=True,
                    channel_axis=0  # output features axis for Linear
                )
            # Data-driven initialization from existing weights
            try:
                self.weight_quantizer.initialize_from_tensor(self.linear.weight)
            except Exception:
                pass
        else:
            self.weight_quantizer = None
        
        # Initialize bias quantizer
        if self.quantize_bias:
            if bias_quantizer is not None:
                self.bias_quantizer = bias_quantizer
            else:
                self.bias_quantizer = create_lsq_quantizer(
                    num_bits=bias_bits,
                    symmetric=bias_symmetric,
                    per_channel=False,  # Scalar quantization for bias
                    channel_axis=0
                )
            try:
                self.bias_quantizer.initialize_from_tensor(self.linear.bias)
            except Exception:
                pass
        else:
            self.bias_quantizer = None
        
        # Initialize activation quantizer
        if quantize_activations:
            if activation_quantizer is not None:
                self.activation_quantizer = activation_quantizer
            else:
                self.activation_quantizer = create_lsq_quantizer(
                    num_bits=activation_bits,
                    symmetric=activation_symmetric,
                    per_channel=False  # Scalar quantization for activations
                )
        else:
            self.activation_quantizer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Quantize weights if enabled
        if self.quantize_weights and self.weight_quantizer is not None:
            quantized_weight = self.weight_quantizer(self.linear.weight)
        else:
            quantized_weight = self.linear.weight
        
        # Quantize bias if enabled and exists
        if self.linear.bias is not None:
            if self.quantize_bias and self.bias_quantizer is not None:
                quantized_bias = self.bias_quantizer(self.linear.bias)
            else:
                quantized_bias = self.linear.bias
            output = F.linear(x, quantized_weight, quantized_bias)
        else:
            output = F.linear(x, quantized_weight)
        
        # Quantize activations if enabled
        if self.quantize_activations and self.activation_quantizer is not None:
            output = self.activation_quantizer(output)
        
        return output
    
    def get_quantization_stats(self) -> dict:
        """Get quantization statistics for monitoring.
        
        Returns:
            Dictionary containing quantization statistics
        """
        stats = {}
        
        if self.quantize_weights and self.weight_quantizer is not None:
            stats['weight_quantization_error'] = self.weight_quantizer.get_quantization_error(
                self.linear.weight
            ).item()
            stats['weight_effective_bits'] = self.weight_quantizer.get_effective_bits(
                self.linear.weight
            )
        
        if self.quantize_bias and self.bias_quantizer is not None and self.linear.bias is not None:
            stats['bias_quantization_error'] = self.bias_quantizer.get_quantization_error(
                self.linear.bias
            ).item()
            stats['bias_effective_bits'] = self.bias_quantizer.get_effective_bits(
                self.linear.bias
            )
        
        return stats


class QuantizedConv1d(nn.Module):
    """Quantized 1D convolutional layer with LSQ quantization.
    
    This layer applies LSQ quantization to the weight tensor and bias vector
    before performing the convolution operation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Stride of the convolution
        padding: Padding added to both sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input to output channels
        bias: Whether to include bias term
        weight_quantizer: LSQ quantizer for weights (optional)
        bias_quantizer: LSQ quantizer for bias (optional)
        activation_quantizer: LSQ quantizer for activations (optional)
        quantize_weights: Whether to quantize weights (default: True)
        quantize_bias: Whether to quantize bias (default: True)
        quantize_activations: Whether to quantize activations (default: False)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        weight_quantizer: Optional[LSQQuantizer] = None,
        bias_quantizer: Optional[LSQQuantizer] = None,
        activation_quantizer: Optional[LSQQuantizer] = None,
        quantize_weights: bool = True,
        quantize_bias: bool = True,
        quantize_activations: bool = False,
        weight_bits: int = 8,
        bias_bits: int = 8,
        activation_bits: int = 8,
        weight_symmetric: bool = True,
        bias_symmetric: bool = True,
        activation_symmetric: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.quantize_weights = quantize_weights
        self.quantize_bias = quantize_bias and bias  # Only quantize bias if it exists
        self.quantize_activations = quantize_activations
        
        # Create conv1d layer
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        
        # Initialize weight quantizer
        if quantize_weights:
            if weight_quantizer is not None:
                self.weight_quantizer = weight_quantizer
            else:
                # Prefer per-channel quantization for weights (better accuracy)
                self.weight_quantizer = create_lsq_quantizer(
                    num_bits=weight_bits,
                    symmetric=weight_symmetric,
                    per_channel=True,
                    channel_axis=0  # output channels axis for Conv1d
                )
            try:
                self.weight_quantizer.initialize_from_tensor(self.conv1d.weight)
            except Exception:
                pass
        else:
            self.weight_quantizer = None
        
        # Initialize bias quantizer
        if self.quantize_bias:
            if bias_quantizer is not None:
                self.bias_quantizer = bias_quantizer
            else:
                self.bias_quantizer = create_lsq_quantizer(
                    num_bits=bias_bits,
                    symmetric=bias_symmetric,
                    per_channel=False,  # Scalar quantization for bias
                    channel_axis=0
                )
            try:
                self.bias_quantizer.initialize_from_tensor(self.conv1d.bias)
            except Exception:
                pass
        else:
            self.bias_quantizer = None
        
        # Initialize activation quantizer
        if quantize_activations:
            if activation_quantizer is not None:
                self.activation_quantizer = activation_quantizer
            else:
                self.activation_quantizer = create_lsq_quantizer(
                    num_bits=activation_bits,
                    symmetric=activation_symmetric,
                    per_channel=False  # Scalar quantization for activations
                )
        else:
            self.activation_quantizer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, output_length)
        """
        # Quantize weights if enabled
        if self.quantize_weights and self.weight_quantizer is not None:
            quantized_weight = self.weight_quantizer(self.conv1d.weight)
        else:
            quantized_weight = self.conv1d.weight
        
        # Quantize bias if enabled and exists
        if self.conv1d.bias is not None:
            if self.quantize_bias and self.bias_quantizer is not None:
                quantized_bias = self.bias_quantizer(self.conv1d.bias)
            else:
                quantized_bias = self.conv1d.bias
            output = F.conv1d(
                x, quantized_weight, quantized_bias,
                self.stride, self.padding, self.dilation, self.groups
            )
        else:
            output = F.conv1d(
                x, quantized_weight, None,
                self.stride, self.padding, self.dilation, self.groups
            )
        
        # Quantize activations if enabled
        if self.quantize_activations and self.activation_quantizer is not None:
            output = self.activation_quantizer(output)
        
        return output
    
    def get_quantization_stats(self) -> dict:
        """Get quantization statistics for monitoring.
        
        Returns:
            Dictionary containing quantization statistics
        """
        stats = {}
        
        if self.quantize_weights and self.weight_quantizer is not None:
            stats['weight_quantization_error'] = self.weight_quantizer.get_quantization_error(
                self.conv1d.weight
            ).item()
            stats['weight_effective_bits'] = self.weight_quantizer.get_effective_bits(
                self.conv1d.weight
            )
        
        if self.quantize_bias and self.bias_quantizer is not None and self.conv1d.bias is not None:
            stats['bias_quantization_error'] = self.bias_quantizer.get_quantization_error(
                self.conv1d.bias
            ).item()
            stats['bias_effective_bits'] = self.bias_quantizer.get_effective_bits(
                self.conv1d.bias
            )
        
        return stats


class QuantizedLayerNorm(nn.Module):
    """Quantized layer normalization with LSQ quantization.
    
    This layer applies LSQ quantization to the normalized output.
    
    Args:
        normalized_shape: Input shape from an expected input
        eps: A value added to the denominator for numerical stability
        elementwise_affine: Whether to use learnable affine parameters
        activation_quantizer: LSQ quantizer for activations (optional)
        quantize_activations: Whether to quantize activations (default: False)
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        activation_quantizer: Optional[LSQQuantizer] = None,
        quantize_activations: bool = False,
        activation_bits: int = 8,
        activation_symmetric: bool = True
    ):
        super().__init__()
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.quantize_activations = quantize_activations
        
        # Create layer norm
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        
        # Initialize quantizer
        if quantize_activations:
            if activation_quantizer is not None:
                self.activation_quantizer = activation_quantizer
            else:
                self.activation_quantizer = create_lsq_quantizer(
                    num_bits=activation_bits,
                    symmetric=activation_symmetric,
                    per_channel=False
                )
        else:
            self.activation_quantizer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized and optionally quantized tensor
        """
        # Apply layer normalization
        output = self.layer_norm(x)
        
        # Quantize activations if enabled
        if self.quantize_activations and self.activation_quantizer is not None:
            output = self.activation_quantizer(output)
        
        return output


class QuantizedGroupNorm(nn.Module):
    """Quantized group normalization with LSQ quantization.
    
    This layer applies LSQ quantization to the normalized output.
    
    Args:
        num_groups: Number of groups to separate the channels into
        num_channels: Number of channels expected in input
        eps: A value added to the denominator for numerical stability
        affine: Whether to use learnable affine parameters
        activation_quantizer: LSQ quantizer for activations (optional)
        quantize_activations: Whether to quantize activations (default: False)
    """
    
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        activation_quantizer: Optional[LSQQuantizer] = None,
        quantize_activations: bool = False,
        activation_bits: int = 8,
        activation_symmetric: bool = True
    ):
        super().__init__()
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.quantize_activations = quantize_activations
        
        # Create group norm
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps, affine)
        
        # Initialize quantizer
        if quantize_activations:
            if activation_quantizer is not None:
                self.activation_quantizer = activation_quantizer
            else:
                self.activation_quantizer = create_lsq_quantizer(
                    num_bits=activation_bits,
                    symmetric=activation_symmetric,
                    per_channel=False
                )
        else:
            self.activation_quantizer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization.
        
        Args:
            x: Input tensor of shape (N, C, ...)
            
        Returns:
            Normalized and optionally quantized tensor
        """
        # Apply group normalization
        output = self.group_norm(x)
        
        # Quantize activations if enabled
        if self.quantize_activations and self.activation_quantizer is not None:
            output = self.activation_quantizer(output)
        
        return output


def convert_linear_to_quantized(
    linear_layer: nn.Linear,
    weight_bits: int = 8,
    bias_bits: int = 8,
    activation_bits: int = 8,
    quantize_weights: bool = True,
    quantize_bias: bool = True,
    quantize_activations: bool = False
) -> QuantizedLinear:
    """Convert a standard Linear layer to QuantizedLinear.
    
    Args:
        linear_layer: Original linear layer
        weight_bits: Number of bits for weight quantization
        bias_bits: Number of bits for bias quantization
        activation_bits: Number of bits for activation quantization
        quantize_weights: Whether to quantize weights
        quantize_bias: Whether to quantize bias
        quantize_activations: Whether to quantize activations
        
    Returns:
        QuantizedLinear layer with copied parameters
    """
    quantized_layer = QuantizedLinear(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
        quantize_weights=quantize_weights,
        quantize_bias=quantize_bias,
        quantize_activations=quantize_activations,
        weight_bits=weight_bits,
        bias_bits=bias_bits,
        activation_bits=activation_bits
    )
    
    # Copy parameters
    with torch.no_grad():
        quantized_layer.linear.weight.copy_(linear_layer.weight)
        if linear_layer.bias is not None:
            quantized_layer.linear.bias.copy_(linear_layer.bias)
    
    return quantized_layer


def convert_conv1d_to_quantized(
    conv1d_layer: nn.Conv1d,
    weight_bits: int = 8,
    bias_bits: int = 8,
    activation_bits: int = 8,
    quantize_weights: bool = True,
    quantize_bias: bool = True,
    quantize_activations: bool = False
) -> QuantizedConv1d:
    """Convert a standard Conv1d layer to QuantizedConv1d.
    
    Args:
        conv1d_layer: Original conv1d layer
        weight_bits: Number of bits for weight quantization
        bias_bits: Number of bits for bias quantization
        activation_bits: Number of bits for activation quantization
        quantize_weights: Whether to quantize weights
        quantize_bias: Whether to quantize bias
        quantize_activations: Whether to quantize activations
        
    Returns:
        QuantizedConv1d layer with copied parameters
    """
    quantized_layer = QuantizedConv1d(
        in_channels=conv1d_layer.in_channels,
        out_channels=conv1d_layer.out_channels,
        kernel_size=conv1d_layer.kernel_size,
        stride=conv1d_layer.stride,
        padding=conv1d_layer.padding,
        dilation=conv1d_layer.dilation,
        groups=conv1d_layer.groups,
        bias=conv1d_layer.bias is not None,
        quantize_weights=quantize_weights,
        quantize_bias=quantize_bias,
        quantize_activations=quantize_activations,
        weight_bits=weight_bits,
        bias_bits=bias_bits,
        activation_bits=activation_bits
    )
    
    # Copy parameters
    with torch.no_grad():
        quantized_layer.conv1d.weight.copy_(conv1d_layer.weight)
        if conv1d_layer.bias is not None:
            quantized_layer.conv1d.bias.copy_(conv1d_layer.bias)
    
    return quantized_layer


def convert_layernorm_to_quantized(
    layernorm_layer: nn.LayerNorm,
    activation_bits: int = 8,
    quantize_activations: bool = False,
    activation_symmetric: bool = True
) -> QuantizedLayerNorm:
    """Convert a standard LayerNorm layer to QuantizedLayerNorm.
    
    Args:
        layernorm_layer: Original LayerNorm layer
        activation_bits: Number of bits for activation quantization
        quantize_activations: Whether to quantize activations
        activation_symmetric: Whether to use symmetric quantization for activations
        
    Returns:
        QuantizedLayerNorm layer with copied parameters
    """
    quantized_layer = QuantizedLayerNorm(
        normalized_shape=layernorm_layer.normalized_shape,
        eps=layernorm_layer.eps,
        elementwise_affine=layernorm_layer.elementwise_affine,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        activation_symmetric=activation_symmetric
    )
    
    # Copy parameters
    with torch.no_grad():
        if layernorm_layer.elementwise_affine:
            quantized_layer.layer_norm.weight.copy_(layernorm_layer.weight)
            quantized_layer.layer_norm.bias.copy_(layernorm_layer.bias)
    
    return quantized_layer


def convert_groupnorm_to_quantized(
    groupnorm_layer: nn.GroupNorm,
    activation_bits: int = 8,
    quantize_activations: bool = False,
    activation_symmetric: bool = True
) -> QuantizedGroupNorm:
    """Convert a standard GroupNorm layer to QuantizedGroupNorm.
    
    Args:
        groupnorm_layer: Original GroupNorm layer
        activation_bits: Number of bits for activation quantization
        quantize_activations: Whether to quantize activations
        activation_symmetric: Whether to use symmetric quantization for activations
        
    Returns:
        QuantizedGroupNorm layer with copied parameters
    """
    quantized_layer = QuantizedGroupNorm(
        num_groups=groupnorm_layer.num_groups,
        num_channels=groupnorm_layer.num_channels,
        eps=groupnorm_layer.eps,
        affine=groupnorm_layer.affine,
        quantize_activations=quantize_activations,
        activation_bits=activation_bits,
        activation_symmetric=activation_symmetric
    )
    
    # Copy parameters
    with torch.no_grad():
        if groupnorm_layer.affine:
            quantized_layer.group_norm.weight.copy_(groupnorm_layer.weight)
            quantized_layer.group_norm.bias.copy_(groupnorm_layer.bias)
    
    return quantized_layer
