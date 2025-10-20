"""Dynamic structured pruning with input-dependent gating.

This module implements dynamic pruning that adapts to input activations,
extending the existing structured pruning framework with input-dependent
gating mechanisms.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hardconcrete import HardConcrete


class InputPredictor(nn.Module):
    """Lightweight input predictor for dynamic pruning.
    
    This module extracts statistics from input activations and predicts
    dynamic gating values. Uses SE-style architecture for efficiency.
    
    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden dimension for the predictor network.
        output_dim: Output dimension (number of gates).
        reduction_ratio: Channel reduction ratio for SE-style bottleneck.
        use_global_pool: Whether to use global average pooling.
        use_std: Whether to include standard deviation statistics.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: int = 1,
        reduction_ratio: int = 16,
        use_global_pool: bool = True,
        use_std: bool = True,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_global_pool = use_global_pool
        self.use_std = use_std
        
        # Calculate feature dimension including statistics
        self.feature_dim = input_dim
        if use_std:
            self.feature_dim *= 2  # mean + std
        
        # SE-style bottleneck architecture
        if hidden_dim is None:
            hidden_dim = max(self.feature_dim // reduction_ratio, 8)
        
        self.predictor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output gating values in [0, 1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract input statistics and predict gating values.
        
        Args:
            x: Input tensor of shape (B, *, input_dim) or (B, input_dim).
        
        Returns:
            Gating tensor of shape (B, output_dim).
        """
        # Handle different input shapes
        if x.dim() > 2:
            # For 3D tensors (B, T, F), use global pooling
            if self.use_global_pool:
                # Global average pooling
                x_pooled = x.mean(dim=1)  # (B, input_dim)
            else:
                # Use last timestep
                x_pooled = x[:, -1, :]  # (B, input_dim)
        else:
            x_pooled = x  # (B, input_dim)
        
        # Extract statistics
        if self.use_std:
            x_mean = x_pooled
            x_std = torch.std(x_pooled, dim=-1, keepdim=True).expand_as(x_pooled)
            features = torch.cat([x_mean, x_std], dim=-1)  # (B, input_dim * 2)
        else:
            features = x_pooled
        
        # Predict gating values
        gates = self.predictor(features)  # (B, output_dim)
        
        return gates


class DynamicStructuredGate(nn.Module):
    """Dynamic structured pruning gate with input-dependent activation.
    
    This module combines static learnable pruning (via HardConcrete) with
    dynamic input-dependent gating. The final mask is computed as:
    final_mask = static_mask * dynamic_gate
    
    Args:
        n_in: Number of prunable units (channels, heads, etc.).
        input_dim: Input feature dimension for dynamic gating.
        gate_type: Type of static gate ('hardconcrete' or 'sigmoid').
        hard_concrete_config: Configuration for HardConcrete gate.
        predictor_config: Configuration for input predictor.
        dynamic_weight: Weight for dynamic gating (0.0 = static only, 1.0 = dynamic only).
        use_input_gating: Whether to enable input-dependent gating.
    """
    
    def __init__(
        self,
        n_in: int,
        input_dim: int,
        gate_type: str = "hardconcrete",
        hard_concrete_config: Optional[dict] = None,
        predictor_config: Optional[dict] = None,
        dynamic_weight: float = 0.5,
        use_input_gating: bool = True,
    ) -> None:
        super().__init__()
        
        self.n_in = n_in
        self.input_dim = input_dim
        self.gate_type = gate_type
        self.dynamic_weight = dynamic_weight
        self.use_input_gating = use_input_gating
        
        # Initialize static gate
        if gate_type == "hardconcrete":
            hard_concrete_config = hard_concrete_config or {}
            self.static_gate = HardConcrete(n_in=n_in, **hard_concrete_config)
        elif gate_type == "sigmoid":
            self.static_gate = nn.Parameter(torch.zeros(n_in))
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")
        
        # Initialize dynamic predictor
        if use_input_gating:
            predictor_config = predictor_config or {}
            self.input_predictor = InputPredictor(
                input_dim=input_dim,
                output_dim=n_in,
                **predictor_config
            )
        else:
            self.input_predictor = None
        
        # Mode control
        self.dynamic_mode = True  # Enable dynamic execution by default
        self.static_mask = None   # Cached static mask for static mode
        
    def set_mode(self, dynamic: bool = True) -> None:
        """Set execution mode (dynamic or static).
        
        Args:
            dynamic: If True, use input-dependent dynamic gating.
                    If False, use static mask only.
        """
        self.dynamic_mode = dynamic
        if not dynamic:
            # Cache static mask for static mode
            self._cache_static_mask()
    
    def _cache_static_mask(self) -> None:
        """Cache static mask for static execution mode."""
        with torch.no_grad():
            if self.gate_type == "hardconcrete":
                # Force HardConcrete to eval mode for deterministic mask
                was_training = self.static_gate.training
                self.static_gate.eval()
                # Use deterministic approximation from HardConcrete
                self.static_mask = self.static_gate(current_iter=None)
                # Restore original training state
                if was_training:
                    self.static_gate.train()
            else:  # sigmoid
                self.static_mask = torch.sigmoid(self.static_gate)
    
    def forward(
        self, 
        x: Optional[torch.Tensor] = None, 
        current_iter: Optional[int] = None
    ) -> torch.Tensor:
        """Compute dynamic pruning mask.
        
        Args:
            x: Input tensor for dynamic gating (B, *, input_dim).
            current_iter: Current training iteration for temperature annealing.
        
        Returns:
            Pruning mask tensor of shape (n_in,).
        """
        # Apply static mask if not using dynamic gating or in static mode
        if not self.use_input_gating or not self.dynamic_mode or x is None:
            if not self.dynamic_mode and self.static_mask is not None:
                # In static mode, return cached static mask
                return self.static_mask
            else:
                # Dynamic mode or no cached mask, compute static mask
                if self.gate_type == "hardconcrete":
                    static_mask = self.static_gate(current_iter=current_iter)
                else:  # sigmoid
                    static_mask = torch.sigmoid(self.static_gate)
                return static_mask
        
        # Dynamic mode: get static mask for combination
        if self.gate_type == "hardconcrete":
            static_mask = self.static_gate(current_iter=current_iter)
        else:  # sigmoid
            static_mask = torch.sigmoid(self.static_gate)
        
        # Compute dynamic gating
        dynamic_gate = self.input_predictor(x)  # (B, n_in)
        
        # Combine static and dynamic masks
        # Keep batch dimension for per-sample dynamic gating
        if dynamic_gate.dim() > 1:
            # Broadcast static mask to match batch dimension
            static_mask = static_mask.unsqueeze(0).expand_as(dynamic_gate)  # (B, n_in)
        
        # Weighted combination: final = (1-w) * static + w * (static * dynamic)
        final_mask = (1 - self.dynamic_weight) * static_mask + \
                    self.dynamic_weight * (static_mask * dynamic_gate)
        
        return final_mask
    
    def l0_norm(self) -> torch.Tensor:
        """Compute expected L0 norm of the static mask.
        
        Returns:
            Expected L0 norm as a scalar tensor.
        """
        if self.gate_type == "hardconcrete":
            return self.static_gate.l0_norm()
        else:  # sigmoid
            return torch.sigmoid(self.static_gate).sum()
    
    def get_dynamic_regularization_loss(self) -> torch.Tensor:
        """Compute L1 regularization loss for dynamic gating.
        
        Returns:
            L1 regularization loss tensor.
        """
        if not self.use_input_gating or self.input_predictor is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # L1 regularization on predictor parameters
        l1_loss = 0.0
        for param in self.input_predictor.parameters():
            l1_loss += torch.norm(param, p=1)
        
        return l1_loss
    
    def export_static_mask(self) -> torch.Tensor:
        """Export static mask for deployment.
        
        Returns:
            Static mask tensor of shape (n_in,).
        """
        if self.gate_type == "hardconcrete":
            # Use deterministic approximation
            with torch.no_grad():
                return self.static_gate(current_iter=None)
        else:  # sigmoid
            with torch.no_grad():
                return torch.sigmoid(self.static_gate)
    
    def extra_repr(self) -> str:
        return f"n_in={self.n_in}, input_dim={self.input_dim}, " \
               f"gate_type={self.gate_type}, dynamic_weight={self.dynamic_weight}, " \
               f"use_input_gating={self.use_input_gating}"
