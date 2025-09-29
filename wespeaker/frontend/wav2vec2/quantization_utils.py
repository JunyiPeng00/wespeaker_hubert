"""Quantization utilities for WeDefense.

This module provides utility functions for model quantization, including:
- Model conversion utilities
- Quantization configuration management
- Quantization-aware training helpers
- Model analysis and statistics
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable, Type

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from .lsq_quantizer import LSQQuantizer, create_lsq_quantizer
from .quantized_layers import (
    QuantizedConv1d,
    QuantizedLinear,
    QuantizedLayerNorm,
    QuantizedGroupNorm,
    convert_conv1d_to_quantized,
    convert_linear_to_quantized,
    convert_layernorm_to_quantized,
    convert_groupnorm_to_quantized
)


class QuantizationConfig:
    """Configuration class for model quantization.
    
    This class manages all quantization-related parameters and provides
    a convenient interface for configuring quantization settings.
    """
    
    def __init__(
        self,
        weight_bits: int = 8,
        bias_bits: int = 8,
        activation_bits: int = 8,
        weight_symmetric: bool = True,
        bias_symmetric: bool = True,
        activation_symmetric: bool = True,
        quantize_weights: bool = True,
        quantize_bias: bool = True,
        quantize_activations: bool = False,
        per_channel_weights: bool = True,
        per_channel_activations: bool = False,
        exclude_modules: Optional[List[str]] = None,
        include_modules: Optional[List[str]] = None
    ):
        """Initialize quantization configuration.
        
        Args:
            weight_bits: Number of bits for weight quantization
            bias_bits: Number of bits for bias quantization
            activation_bits: Number of bits for activation quantization
            weight_symmetric: Whether to use symmetric quantization for weights
            bias_symmetric: Whether to use symmetric quantization for bias
            activation_symmetric: Whether to use symmetric quantization for activations
            quantize_weights: Whether to quantize weights
            quantize_bias: Whether to quantize bias
            quantize_activations: Whether to quantize activations
            per_channel_weights: Whether to use per-channel quantization for weights
            per_channel_activations: Whether to use per-channel quantization for activations
            exclude_modules: List of module names to exclude from quantization
            include_modules: List of module names to include in quantization (if None, all are included)
        """
        self.weight_bits = weight_bits
        self.bias_bits = bias_bits
        self.activation_bits = activation_bits
        self.weight_symmetric = weight_symmetric
        self.bias_symmetric = bias_symmetric
        self.activation_symmetric = activation_symmetric
        self.quantize_weights = quantize_weights
        self.quantize_bias = quantize_bias
        self.quantize_activations = quantize_activations
        self.per_channel_weights = per_channel_weights
        self.per_channel_activations = per_channel_activations
        self.exclude_modules = exclude_modules or []
        self.include_modules = include_modules


def collect_lsq_step_size_params(model: nn.Module) -> List[nn.Parameter]:
    """Collect all LSQ step_size parameters from a model.

    This searches submodules for 'weight_quantizer' or 'activation_quantizer'
    that expose a learnable 'step_size' parameter. Returns a de-duplicated list
    of those Parameters that require gradients.

    Args:
        model: The model potentially containing LSQ quantizers

    Returns:
        List of Parameter objects corresponding to LSQ step sizes.
    """
    step_params: List[nn.Parameter] = []
    for module in model.modules():
        # weight quantizer
        if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
            q = module.weight_quantizer
            if hasattr(q, 'step_size') and q.step_size is not None and isinstance(q.step_size, torch.nn.Parameter):
                if q.step_size.requires_grad:
                    step_params.append(q.step_size)
        # activation quantizer
        if hasattr(module, 'activation_quantizer') and module.activation_quantizer is not None:
            q = module.activation_quantizer
            if hasattr(q, 'step_size') and q.step_size is not None and isinstance(q.step_size, torch.nn.Parameter):
                if q.step_size.requires_grad:
                    step_params.append(q.step_size)

    # de-duplicate
    seen = set()
    uniq: List[nn.Parameter] = []
    for p in step_params:
        pid = id(p)
        if pid not in seen:
            uniq.append(p)
            seen.add(pid)
    return uniq


def collect_lsq_step_size_params(model: nn.Module) -> List[nn.Parameter]:
    """Collect all LSQ step_size parameters from a model.

    This searches submodules for 'weight_quantizer', 'bias_quantizer', or 'activation_quantizer'
    that expose a learnable 'step_size' parameter. Returns a de-duplicated list
    of those Parameters that require gradients.

    Args:
        model: The model potentially containing LSQ quantizers

    Returns:
        List of Parameter objects corresponding to LSQ step sizes.
    """
    step_params: List[nn.Parameter] = []
    for module in model.modules():
        # weight quantizer
        if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
            q = module.weight_quantizer
            if hasattr(q, 'step_size') and q.step_size is not None and isinstance(q.step_size, torch.nn.Parameter):
                if q.step_size.requires_grad:
                    step_params.append(q.step_size)
        # bias quantizer
        if hasattr(module, 'bias_quantizer') and module.bias_quantizer is not None:
            q = module.bias_quantizer
            if hasattr(q, 'step_size') and q.step_size is not None and isinstance(q.step_size, torch.nn.Parameter):
                if q.step_size.requires_grad:
                    step_params.append(q.step_size)
        # activation quantizer
        if hasattr(module, 'activation_quantizer') and module.activation_quantizer is not None:
            q = module.activation_quantizer
            if hasattr(q, 'step_size') and q.step_size is not None and isinstance(q.step_size, torch.nn.Parameter):
                if q.step_size.requires_grad:
                    step_params.append(q.step_size)

    # de-duplicate
    seen = set()
    uniq: List[nn.Parameter] = []
    for p in step_params:
        pid = id(p)
        if pid not in seen:
            uniq.append(p)
            seen.add(pid)
    return uniq


def build_two_phase_lsq_optimizer(
    model: nn.Module,
    base_optimizer_cls: Type[torch.optim.Optimizer],
    main_lr: float,
    step_lr: float,
    weight_decay: float = 1e-7,
    **optimizer_kwargs: Any,
) -> torch.optim.Optimizer:
    """Create optimizer with two parameter groups: main params vs. LSQ step_size params.

    This utility separates LSQ ``step_size`` parameters into a dedicated parameter group
    with a typically smaller learning rate to stabilize training under low-bit QAT.

    Args:
        model: Model containing quantized layers.
        base_optimizer_cls: Optimizer class, e.g., ``torch.optim.AdamW``.
        main_lr: Learning rate for regular parameters.
        step_lr: Learning rate for LSQ ``step_size`` parameters.
        weight_decay: Weight decay applied to both groups (if desired).
        optimizer_kwargs: Extra kwargs forwarded to the optimizer.

    Returns:
        Configured optimizer instance with two parameter groups.
    """
    step_params = set(collect_lsq_step_size_params(model))
    main_params: List[nn.Parameter] = []
    lsq_params: List[nn.Parameter] = []

    for param in model.parameters():
        if param in step_params:
            lsq_params.append(param)
        else:
            main_params.append(param)

    param_groups = [
        {"params": main_params, "lr": main_lr, "weight_decay": weight_decay},
        {"params": lsq_params, "lr": step_lr, "weight_decay": weight_decay},
    ]
    optimizer = base_optimizer_cls(param_groups, **optimizer_kwargs)
    return optimizer


class TwoPhaseLSQController:
    """Two-phase LSQ training controller.

    Phase 1 (warmup): Freeze LSQ step sizes and only train regular parameters for ``freeze_steps``.
    Phase 2: Unfreeze LSQ step sizes and continue training with a smaller step-size LR.

    Additionally provides gradient clipping per iteration to stabilize updates.

    Usage pattern per iteration:
        controller.maybe_unfreeze(global_step)
        loss.backward()
        controller.clip_gradients()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        freeze_steps: int = 2000,
        clip_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.freeze_steps = int(max(0, freeze_steps))
        self.clip_norm = float(max(0.0, clip_norm))
        self._unfrozen: bool = self.freeze_steps == 0

        if not self._unfrozen:
            freeze_quantization_parameters(self.model)

    def maybe_unfreeze(self, global_step: int) -> None:
        """Unfreeze LSQ step sizes when reaching the boundary ``freeze_steps``.

        Args:
            global_step: Current global optimization step (starting from 0 or 1, both ok).
        """
        if not self._unfrozen and global_step >= self.freeze_steps:
            unfreeze_quantization_parameters(self.model)
            self._unfrozen = True

    def clip_gradients(self) -> None:
        """Apply global norm gradient clipping if ``clip_norm > 0``."""
        if self.clip_norm > 0.0:
            clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)


class QuantizationConfig:
    """Configuration class for model quantization.
    
    This class manages all quantization-related parameters and provides
    a convenient interface for configuring quantization settings.
    """
    
    def __init__(
        self,
        weight_bits: int = 8,
        bias_bits: int = 8,
        activation_bits: int = 8,
        weight_symmetric: bool = True,
        bias_symmetric: bool = True,
        activation_symmetric: bool = True,
        quantize_weights: bool = True,
        quantize_bias: bool = True,
        quantize_activations: bool = False,
        per_channel_weights: bool = True,
        per_channel_activations: bool = False,
        exclude_modules: Optional[List[str]] = None,
        include_modules: Optional[List[str]] = None
    ):
        """Initialize quantization configuration.
        
        Args:
            weight_bits: Number of bits for weight quantization
            bias_bits: Number of bits for bias quantization
            activation_bits: Number of bits for activation quantization
            weight_symmetric: Whether to use symmetric quantization for weights
            bias_symmetric: Whether to use symmetric quantization for bias
            activation_symmetric: Whether to use symmetric quantization for activations
            quantize_weights: Whether to quantize weights
            quantize_bias: Whether to quantize bias
            quantize_activations: Whether to quantize activations
            per_channel_weights: Whether to use per-channel quantization for weights
            per_channel_activations: Whether to use per-channel quantization for activations
            exclude_modules: List of module names to exclude from quantization
            include_modules: List of module names to include in quantization (if None, all are included)
        """
        self.weight_bits = weight_bits
        self.bias_bits = bias_bits
        self.activation_bits = activation_bits
        self.weight_symmetric = weight_symmetric
        self.bias_symmetric = bias_symmetric
        self.activation_symmetric = activation_symmetric
        self.quantize_weights = quantize_weights
        self.quantize_bias = quantize_bias
        self.quantize_activations = quantize_activations
        self.per_channel_weights = per_channel_weights
        self.per_channel_activations = per_channel_activations
        self.exclude_modules = exclude_modules or []
        self.include_modules = include_modules
    
    def should_quantize_module(self, module_name: str) -> bool:
        """Check if a module should be quantized based on configuration.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if the module should be quantized
        """
        # Check if module is explicitly excluded
        if any(excluded in module_name for excluded in self.exclude_modules):
            return False
        
        # Check if only specific modules should be included
        if self.include_modules is not None:
            return any(included in module_name for included in self.include_modules)
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'weight_bits': self.weight_bits,
            'bias_bits': self.bias_bits,
            'activation_bits': self.activation_bits,
            'weight_symmetric': self.weight_symmetric,
            'bias_symmetric': self.bias_symmetric,
            'activation_symmetric': self.activation_symmetric,
            'quantize_weights': self.quantize_weights,
            'quantize_bias': self.quantize_bias,
            'quantize_activations': self.quantize_activations,
            'per_channel_weights': self.per_channel_weights,
            'per_channel_activations': self.per_channel_activations,
            'exclude_modules': self.exclude_modules,
            'include_modules': self.include_modules
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantizationConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            QuantizationConfig instance
        """
        return cls(**config_dict)


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
    inplace: bool = False
) -> nn.Module:
    """Quantize a model according to the given configuration.
    
    Args:
        model: Model to quantize
        config: Quantization configuration
        inplace: Whether to modify the model in-place
        
    Returns:
        Quantized model
    """
    if not inplace:
        model = copy.deepcopy(model)
    
    # Convert linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and config.should_quantize_module(name):
            # Find the parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
            else:
                parent_module = model
            
            # Convert to quantized layer
            quantized_layer = convert_linear_to_quantized(
                module,
                weight_bits=config.weight_bits,
                bias_bits=config.bias_bits,
                activation_bits=config.activation_bits,
                quantize_weights=config.quantize_weights,
                quantize_bias=config.quantize_bias,
                quantize_activations=config.quantize_activations
            )
            
            setattr(parent_module, attr_name, quantized_layer)
    
    # Convert conv1d layers (including weight_norm wrapped ones)
    for name, module in model.named_modules():
        # Check for both regular Conv1d and weight_norm wrapped Conv1d
        is_conv1d = isinstance(module, nn.Conv1d) or (
            hasattr(module, 'weight') and 
            hasattr(module, 'bias') and 
            hasattr(module, 'in_channels') and 
            hasattr(module, 'out_channels') and
            hasattr(module, 'kernel_size') and
            hasattr(module, 'stride') and
            hasattr(module, 'padding') and
            hasattr(module, 'dilation') and
            hasattr(module, 'groups') and
            hasattr(module, 'forward') and
            callable(getattr(module, 'forward'))
        )
        
        if is_conv1d and config.should_quantize_module(name):
            # Find the parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
            else:
                parent_module = model
            
            # Convert to quantized layer
            # Handle weight_norm wrapped conv1d by extracting the underlying conv layer
            if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
                # This is a weight_norm wrapped conv1d, we need to handle it specially
                # For now, we'll create a regular conv1d with the same parameters
                underlying_conv = nn.Conv1d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None
                )
                # Copy the actual weight (after weight_norm transformation)
                with torch.no_grad():
                    underlying_conv.weight.copy_(module.weight)
                    if module.bias is not None:
                        underlying_conv.bias.copy_(module.bias)
                
                quantized_layer = convert_conv1d_to_quantized(
                    underlying_conv,
                    weight_bits=config.weight_bits,
                    bias_bits=config.bias_bits,
                    activation_bits=config.activation_bits,
                    quantize_weights=config.quantize_weights,
                    quantize_bias=config.quantize_bias,
                    quantize_activations=config.quantize_activations
                )
            else:
                # Regular conv1d
                quantized_layer = convert_conv1d_to_quantized(
                    module,
                    weight_bits=config.weight_bits,
                    bias_bits=config.bias_bits,
                    activation_bits=config.activation_bits,
                    quantize_weights=config.quantize_weights,
                    quantize_bias=config.quantize_bias,
                    quantize_activations=config.quantize_activations
                )
            
            setattr(parent_module, attr_name, quantized_layer)
    
    # Convert LayerNorm layers
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) and config.should_quantize_module(name):
            # Find the parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
            else:
                parent_module = model
            
            # Convert to quantized layer
            quantized_layer = convert_layernorm_to_quantized(
                module,
                activation_bits=config.activation_bits,
                quantize_activations=config.quantize_activations,
                activation_symmetric=config.activation_symmetric
            )
            
            setattr(parent_module, attr_name, quantized_layer)
    
    # Convert GroupNorm layers
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm) and config.should_quantize_module(name):
            # Find the parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            
            if parent_name:
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
            else:
                parent_module = model
            
            # Convert to quantized layer
            quantized_layer = convert_groupnorm_to_quantized(
                module,
                activation_bits=config.activation_bits,
                quantize_activations=config.quantize_activations,
                activation_symmetric=config.activation_symmetric
            )
            
            setattr(parent_module, attr_name, quantized_layer)
    
    return model


def get_quantization_stats(model: nn.Module) -> Dict[str, Any]:
    """Get quantization statistics for a quantized model.
    
    Args:
        model: Quantized model
        
    Returns:
        Dictionary containing quantization statistics
    """
    stats = {
        'total_parameters': 0,
        'quantized_parameters': 0,
        'quantization_errors': {},
        'effective_bits': {},
        'quantized_modules': []
    }
    
    for name, module in model.named_modules():
        if hasattr(module, 'get_quantization_stats'):
            module_stats = module.get_quantization_stats()
            stats['quantization_errors'][name] = module_stats.get('weight_quantization_error', 0.0)
            stats['effective_bits'][name] = module_stats.get('weight_effective_bits', 0.0)
            stats['quantized_modules'].append(name)
        
        # Count parameters
        for name, param in module.named_parameters(recurse=False):
            stats['total_parameters'] += param.numel()
            # Count only the tensors actually quantized: weights and optionally bias
            if name == 'linear.weight' or name == 'conv1d.weight' or name == 'weight':
                if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
                    stats['quantized_parameters'] += param.numel()
            if name == 'linear.bias' or name == 'conv1d.bias' or name == 'bias':
                if hasattr(module, 'bias_quantizer') and module.bias_quantizer is not None:
                    stats['quantized_parameters'] += param.numel()
    
    return stats


def calculate_model_size(model: nn.Module, bits_per_param: int = 32) -> Dict[str, float]:
    """Calculate model size in different units.
    
    Args:
        model: Model to analyze
        bits_per_param: Number of bits per parameter (32 for float32, 8 for int8, etc.)
        
    Returns:
        Dictionary containing size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate sizes
    bits = total_params * bits_per_param
    bytes_size = bits / 8
    kb_size = bytes_size / 1024
    mb_size = kb_size / 1024
    gb_size = mb_size / 1024
    
    return {
        'total_parameters': total_params,
        'bits': bits,
        'bytes': bytes_size,
        'kb': kb_size,
        'mb': mb_size,
        'gb': gb_size,
        'bits_per_param': bits_per_param
    }


def compare_model_sizes(
    original_model: nn.Module,
    quantized_model: nn.Module,
    original_bits: int = 32,
    quantized_bits: int = 8
) -> Dict[str, Any]:
    """Compare sizes between original and quantized models.
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        original_bits: Bits per parameter in original model
        quantized_bits: Bits per parameter in quantized model
        
    Returns:
        Dictionary containing comparison statistics
    """
    original_size = calculate_model_size(original_model, original_bits)
    quantized_size = calculate_model_size(quantized_model, quantized_bits)
    
    compression_ratio = original_size['mb'] / quantized_size['mb']
    size_reduction = (original_size['mb'] - quantized_size['mb']) / original_size['mb']
    
    return {
        'original_size_mb': original_size['mb'],
        'quantized_size_mb': quantized_size['mb'],
        'compression_ratio': compression_ratio,
        'size_reduction_percent': size_reduction * 100,
        'original_parameters': original_size['total_parameters'],
        'quantized_parameters': quantized_size['total_parameters']
    }


def create_quantization_aware_training_hooks(
    model: nn.Module,
    config: QuantizationConfig
) -> List[torch.utils.hooks.RemovableHandle]:
    """Create hooks for quantization-aware training.
    
    Args:
        model: Model to add hooks to
        config: Quantization configuration
        
    Returns:
        List of hook handles that can be removed
    """
    hooks = []
    
    def quantization_hook(module, input, output):
        """Hook to apply quantization during forward pass."""
        if hasattr(module, 'activation_quantizer') and module.activation_quantizer is not None:
            return module.activation_quantizer(output)
        return output
    
    # Add hooks to modules that have activation quantizers
    for name, module in model.named_modules():
        if hasattr(module, 'activation_quantizer') and module.activation_quantizer is not None:
            hook = module.register_forward_hook(quantization_hook)
            hooks.append(hook)
    
    return hooks


def freeze_quantization_parameters(model: nn.Module) -> None:
    """Freeze quantization parameters (step sizes) to prevent further training.
    
    Args:
        model: Model with quantization parameters to freeze
    """
    for module in model.modules():
        if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
            if hasattr(module.weight_quantizer, 'step_size') and module.weight_quantizer.step_size is not None:
                module.weight_quantizer.step_size.requires_grad = False
        
        if hasattr(module, 'activation_quantizer') and module.activation_quantizer is not None:
            if hasattr(module.activation_quantizer, 'step_size') and module.activation_quantizer.step_size is not None:
                module.activation_quantizer.step_size.requires_grad = False


def unfreeze_quantization_parameters(model: nn.Module) -> None:
    """Unfreeze quantization parameters to allow training.
    
    Args:
        model: Model with quantization parameters to unfreeze
    """
    for module in model.modules():
        if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
            if hasattr(module.weight_quantizer, 'step_size') and module.weight_quantizer.step_size is not None:
                module.weight_quantizer.step_size.requires_grad = True
        
        if hasattr(module, 'activation_quantizer') and module.activation_quantizer is not None:
            if hasattr(module.activation_quantizer, 'step_size') and module.activation_quantizer.step_size is not None:
                module.activation_quantizer.step_size.requires_grad = True


def apply_quantization(
    model: nn.Module, 
    config: Optional[QuantizationConfig] = None,
    target_modules: Optional[List[str]] = None,
    preserve_hp_gating: bool = True
) -> nn.Module:
    """Apply quantization to a model, preserving HP framework gating logic.
    
    This function converts standard PyTorch layers to quantized versions while
    preserving the Hard Concrete gating logic used in the HP framework for
    structured pruning.
    
    Args:
        model: The model to quantize
        config: Quantization configuration. If None, uses default 8-bit symmetric
        target_modules: List of module types to quantize. If None, quantizes all supported types
        preserve_hp_gating: Whether to preserve Hard Concrete gating logic
        
    Returns:
        The quantized model with preserved gating logic
        
    Example:
        >>> from wedefense.frontend.wav2vec2 import wavlm_base, apply_quantization
        >>> model = wavlm_base()
        >>> quantized_model = apply_quantization(model, preserve_hp_gating=True)
    """
    if config is None:
        config = QuantizationConfig()
    
    if target_modules is None:
        target_modules = ['Linear', 'Conv1d', 'LayerNorm', 'GroupNorm']
    
    # Create a copy of the model to avoid modifying the original
    quantized_model = copy.deepcopy(model)
    
    # Track conversion statistics
    conversion_stats = {
        'converted_modules': 0,
        'preserved_gating': 0,
        'skipped_modules': 0
    }
    
    def convert_module(module, name=""):
        """Recursively convert modules to quantized versions."""
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this module should be quantized
            should_quantize = (
                child_module.__class__.__name__ in target_modules and
                not isinstance(child_module, (QuantizedLinear, QuantizedConv1d))
            )
            
            if should_quantize:
                try:
                    # Check if module has Hard Concrete gating
                    has_gating = False
                    gating_info = {}
                    
                    if hasattr(child_module, 'hard_concrete') and child_module.hard_concrete is not None:
                        has_gating = True
                        gating_info['hard_concrete'] = child_module.hard_concrete
                    
                    if hasattr(child_module, 'hard_concrete_for_heads') and child_module.hard_concrete_for_heads is not None:
                        has_gating = True
                        gating_info['hard_concrete_for_heads'] = child_module.hard_concrete_for_heads
                    
                    if hasattr(child_module, 'hard_concrete_for_layer') and child_module.hard_concrete_for_layer is not None:
                        has_gating = True
                        gating_info['hard_concrete_for_layer'] = child_module.hard_concrete_for_layer
                    
                    if hasattr(child_module, 'hard_concrete_for_intermediate') and child_module.hard_concrete_for_intermediate is not None:
                        has_gating = True
                        gating_info['hard_concrete_for_intermediate'] = child_module.hard_concrete_for_intermediate
                    
                    # Convert to quantized version
                    if isinstance(child_module, nn.Linear):
                        quantized_child = convert_linear_to_quantized(
                            child_module, 
                            weight_bits=config.weight_bits,
                            bias_bits=config.bias_bits,
                            activation_bits=config.activation_bits,
                            quantize_weights=config.quantize_weights,
                            quantize_bias=config.quantize_bias,
                            quantize_activations=config.quantize_activations
                        )
                    elif isinstance(child_module, nn.Conv1d):
                        quantized_child = convert_conv1d_to_quantized(
                            child_module,
                            weight_bits=config.weight_bits,
                            bias_bits=config.bias_bits,
                            activation_bits=config.activation_bits,
                            quantize_weights=config.quantize_weights,
                            quantize_bias=config.quantize_bias,
                            quantize_activations=config.quantize_activations
                        )
                    elif isinstance(child_module, nn.LayerNorm):
                        quantized_child = convert_layernorm_to_quantized(
                            child_module,
                            activation_bits=config.activation_bits,
                            quantize_activations=config.quantize_activations,
                            activation_symmetric=config.activation_symmetric
                        )
                    elif isinstance(child_module, nn.GroupNorm):
                        quantized_child = convert_groupnorm_to_quantized(
                            child_module,
                            activation_bits=config.activation_bits,
                            quantize_activations=config.quantize_activations,
                            activation_symmetric=config.activation_symmetric
                        )
                    else:
                        # Skip other module types
                        conversion_stats['skipped_modules'] += 1
                        convert_module(child_module, full_name)
                        continue
                    
                    # Preserve gating logic if requested
                    if preserve_hp_gating and has_gating:
                        for gating_name, gating_module in gating_info.items():
                            setattr(quantized_child, gating_name, gating_module)
                        conversion_stats['preserved_gating'] += 1
                    
                    # Replace the module
                    setattr(module, child_name, quantized_child)
                    conversion_stats['converted_modules'] += 1
                    
                except Exception as e:
                    print(f"Warning: Failed to quantize module {full_name}: {e}")
                    conversion_stats['skipped_modules'] += 1
            else:
                # Recursively process child modules
                convert_module(child_module, full_name)
    
    # Apply quantization
    convert_module(quantized_model)
    
    # Print conversion statistics
    print(f"Quantization applied successfully:")
    print(f"  - Converted modules: {conversion_stats['converted_modules']}")
    print(f"  - Preserved gating: {conversion_stats['preserved_gating']}")
    print(f"  - Skipped modules: {conversion_stats['skipped_modules']}")
    
    return quantized_model


def apply_quantization_with_hp_integration(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
    enable_pruning: bool = True,
    pruning_config: Optional[Dict] = None
) -> nn.Module:
    """Apply quantization with full HP framework integration.
    
    This function applies quantization while preserving the complete HP framework
    functionality, including Hard Concrete gating, structured pruning, and
    progressive sparsity scheduling.
    
    Args:
        model: The model to quantize and integrate with HP framework
        config: Quantization configuration
        enable_pruning: Whether to enable structured pruning
        pruning_config: Configuration for pruning (if enabled)
        
    Returns:
        Model with quantization and HP framework integration
    """
    if config is None:
        config = QuantizationConfig()
    
    if pruning_config is None:
        pruning_config = {
            'target_sparsity': 0.5,
            'sparsity_warmup_epochs': 7,
            'sparsity_schedule': 'cosine',
            'min_sparsity': 0.0
        }
    
    # Step 1: Apply quantization first
    print("Step 1: Applying quantization...")
    quantized_model = apply_quantization(model, config, preserve_hp_gating=True)
    
    # Step 2: Enable HP framework gating if pruning is enabled
    if enable_pruning:
        print("Step 2: Enabling HP framework gating...")
        _enable_hp_gating(quantized_model, pruning_config)
    
    # Step 3: Set up quantization-aware training
    print("Step 3: Setting up quantization-aware training...")
    _setup_quantization_training(quantized_model, config)
    
    return quantized_model


def _enable_hp_gating(model: nn.Module, pruning_config: Dict) -> None:
    """Enable Hard Concrete gating for structured pruning."""
    # Extract temperature annealing parameters from pruning_config
    min_temperature = pruning_config.get('min_temperature', 0.1)
    temperature_decay = pruning_config.get('temperature_decay', 0.95)
    temperature_decay_freq = pruning_config.get('temperature_decay_freq', 100)
    from .hardconcrete import HardConcrete
    from .components import ConvolutionalPositionalEmbedding
    from .quantized_layers import QuantizedConv1d, QuantizedLinear
    
    def resolve_underlying_conv1d(module: nn.Module) -> Optional[nn.Conv1d]:
        """Return an nn.Conv1d if this module or its field wraps one.
        Supports both float modules (module.conv: nn.Conv1d) and quantized wrappers
        (module.conv: QuantizedConv1d with .conv1d: nn.Conv1d).
        """
        # Common pattern in extractor blocks: module.conv may be Conv1d or QuantizedConv1d
        conv_attr = getattr(module, 'conv', None)
        if isinstance(conv_attr, nn.Conv1d):
            return conv_attr
        if isinstance(conv_attr, QuantizedConv1d):
            return getattr(conv_attr, 'conv1d', None)
        return None
    
    def resolve_underlying_linear(module: nn.Module, field_name: str) -> Optional[nn.Linear]:
        """Return an nn.Linear from module.<field_name> for float or quantized wrapper.
        Example field_name: 'intermediate_dense'.
        """
        dense = getattr(module, field_name, None)
        if isinstance(dense, nn.Linear):
            return dense
        if isinstance(dense, QuantizedLinear):
            return getattr(dense, 'linear', None)
        return None
    
    for name, module in model.named_modules():
        # Enable gating for convolutional layers (handle float and quantized)
        conv1d = resolve_underlying_conv1d(module)
        if conv1d is not None and not isinstance(module, ConvolutionalPositionalEmbedding):
            if not hasattr(module, 'hard_concrete') or module.hard_concrete is None:
                out_channels = conv1d.out_channels
                module.hard_concrete = HardConcrete(
                    n_in=out_channels, 
                    init_mean=0.01,
                    min_temperature=min_temperature,
                    temperature_decay=temperature_decay,
                    temperature_decay_freq=temperature_decay_freq
                )
        
        # Enable gating for attention layers
        elif hasattr(module, 'remaining_heads') and hasattr(module, 'total_num_heads'):
            if not hasattr(module, 'hard_concrete_for_heads') or module.hard_concrete_for_heads is None:
                num_heads = len(module.remaining_heads)
                module.hard_concrete_for_heads = HardConcrete(
                    n_in=num_heads, 
                    init_mean=0.5,
                    min_temperature=min_temperature,
                    temperature_decay=temperature_decay,
                    temperature_decay_freq=temperature_decay_freq
                )
            
            if not hasattr(module, 'hard_concrete_for_layer') or module.hard_concrete_for_layer is None:
                module.hard_concrete_for_layer = HardConcrete(
                    n_in=1, 
                    init_mean=0.01,
                    min_temperature=min_temperature,
                    temperature_decay=temperature_decay,
                    temperature_decay_freq=temperature_decay_freq
                )
        
        # Enable gating for feed-forward layers (handle float and quantized)
        elif hasattr(module, 'intermediate_dense'):
            underlying_linear = resolve_underlying_linear(module, 'intermediate_dense')
            if underlying_linear is not None:
                if not hasattr(module, 'hard_concrete_for_intermediate') or module.hard_concrete_for_intermediate is None:
                    intermediate_features = underlying_linear.out_features
                    module.hard_concrete_for_intermediate = HardConcrete(
                        n_in=intermediate_features, 
                        init_mean=0.5,
                        min_temperature=min_temperature,
                        temperature_decay=temperature_decay,
                        temperature_decay_freq=temperature_decay_freq
                    )
                
                if not hasattr(module, 'hard_concrete_for_layer') or module.hard_concrete_for_layer is None:
                    module.hard_concrete_for_layer = HardConcrete(
                        n_in=1, 
                        init_mean=0.01,
                        min_temperature=min_temperature,
                        temperature_decay=temperature_decay,
                        temperature_decay_freq=temperature_decay_freq
                    )


def _setup_quantization_training(model: nn.Module, config: QuantizationConfig) -> None:
    """Set up quantization-aware training parameters."""
    # Set quantization parameters to trainable
    unfreeze_quantization_parameters(model)
    
    # Set appropriate training modes
    for module in model.modules():
        if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
            module.weight_quantizer.train()
        
        if hasattr(module, 'activation_quantizer') and module.activation_quantizer is not None:
            module.activation_quantizer.train()


# Predefined quantization configurations
QUANTIZATION_CONFIGS = {
    '8bit_symmetric': QuantizationConfig(
        weight_bits=8,
        bias_bits=8,
        activation_bits=8,
        weight_symmetric=True,
        bias_symmetric=True,
        activation_symmetric=True,
        quantize_weights=True,
        quantize_bias=True,
        quantize_activations=False
    ),
    '6bit_symmetric': QuantizationConfig(
        weight_bits=6,
        bias_bits=6,
        activation_bits=6,
        weight_symmetric=True,
        bias_symmetric=True,
        activation_symmetric=True,
        quantize_weights=True,
        quantize_bias=True,
        quantize_activations=False
    ),
    '8bit_asymmetric': QuantizationConfig(
        weight_bits=8,
        bias_bits=8,
        activation_bits=8,
        weight_symmetric=False,
        bias_symmetric=False,
        activation_symmetric=False,
        quantize_weights=True,
        quantize_bias=True,
        quantize_activations=False
    ),
    '4bit_symmetric': QuantizationConfig(
        weight_bits=4,
        bias_bits=4,
        activation_bits=4,
        weight_symmetric=True,
        bias_symmetric=True,
        activation_symmetric=True,
        quantize_weights=True,
        quantize_bias=True,
        quantize_activations=False
    ),
    '2bit_symmetric': QuantizationConfig(
        weight_bits=2,
        bias_bits=2,
        activation_bits=2,
        weight_symmetric=True,
        bias_symmetric=True,
        activation_symmetric=True,
        quantize_weights=True,
        quantize_bias=True,
        quantize_activations=False
    ),
    '1bit_symmetric': QuantizationConfig(
        weight_bits=1,
        bias_bits=1,
        activation_bits=1,
        weight_symmetric=True,
        bias_symmetric=True,
        activation_symmetric=True,
        quantize_weights=True,
        quantize_bias=True,
        quantize_activations=False
    ),
    'mixed_precision': QuantizationConfig(
        weight_bits=8,
        bias_bits=8,
        activation_bits=16,
        weight_symmetric=True,
        bias_symmetric=True,
        activation_symmetric=True,
        quantize_weights=True,
        quantize_bias=True,
        quantize_activations=True
    ),
    'weights_only': QuantizationConfig(
        weight_bits=8,
        bias_bits=8,
        activation_bits=8,
        weight_symmetric=True,
        bias_symmetric=True,
        activation_symmetric=True,
        quantize_weights=True,
        quantize_bias=False,
        quantize_activations=False
    ),
    'weights_and_bias': QuantizationConfig(
        weight_bits=8,
        bias_bits=8,
        activation_bits=8,
        weight_symmetric=True,
        bias_symmetric=True,
        activation_symmetric=True,
        quantize_weights=True,
        quantize_bias=True,
        quantize_activations=False
    )
}


def get_quantization_config(config_name: str) -> QuantizationConfig:
    """Get a predefined quantization configuration.
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        QuantizationConfig instance
        
    Raises:
        ValueError: If config_name is not found
    """
    if config_name not in QUANTIZATION_CONFIGS:
        available_configs = list(QUANTIZATION_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available configs: {available_configs}")
    
    return QUANTIZATION_CONFIGS[config_name]
