# Dynamic Pruning Parameters Guide

## Overview

This document explains the key parameters used in dynamic pruning and their effects on model behavior.

## Core Parameters

### 1. `gate_type` - Static Gate Type

**Purpose**: Determines the type of static gate used in the `DynamicStructuredGate`.

**Options**:
- `'hardconcrete'` (default): Uses HardConcrete distribution for differentiable binary masks
- `'sigmoid'`: Uses sigmoid activation for continuous gates

**Effect**:
- **HardConcrete**: Provides stochastic training with deterministic inference, supports L0 regularization
- **Sigmoid**: Provides continuous gates throughout training and inference

**Example**:
```yaml
dynamic_pruning_config:
  gate_type: "hardconcrete"  # or "sigmoid"
```

### 2. `dynamic_weight` - Dynamic vs Static Balance

**Purpose**: Controls the balance between static and dynamic gating.

**Range**: `[0.0, 1.0]`

**Effect**:
- `0.0`: Pure static gating (only HardConcrete/sigmoid gates)
- `1.0`: Pure dynamic gating (only input-dependent gates)
- `0.5`: Equal balance between static and dynamic

**Formula**:
```
final_mask = (1 - dynamic_weight) * static_mask + dynamic_weight * (static_mask * dynamic_gate)
```

**Example**:
```yaml
dynamic_pruning_config:
  dynamic_weight: 0.5  # Balanced static-dynamic gating
```

### 3. `use_input_gating` - Enable Input-Dependent Gating

**Purpose**: Controls whether to use input-dependent dynamic gating.

**Options**:
- `true` (default): Enable dynamic gating based on input statistics
- `false`: Disable dynamic gating, use only static gates

**Effect**:
- When `true`: The `InputPredictor` analyzes input statistics and generates dynamic masks
- When `false`: Only static gates are used, similar to standard structured pruning

**Example**:
```yaml
dynamic_pruning_config:
  use_input_gating: true
```

## Predictor Configuration

### 4. `predictor_config` - Input Predictor Settings

**Purpose**: Configures the lightweight MLP that predicts dynamic gating values from input statistics.

#### 4.1 `reduction_ratio` - SE-style Bottleneck Ratio

**Purpose**: Controls the compression ratio in the SE-style bottleneck architecture.

**Default**: `16`

**Effect**:
- Higher values (e.g., 32): More compression, fewer parameters, less capacity
- Lower values (e.g., 8): Less compression, more parameters, more capacity

**Example**:
```yaml
dynamic_pruning_config:
  predictor_config:
    reduction_ratio: 16  # Compress features by 16x in bottleneck
```

#### 4.2 `use_global_pool` - Global Pooling

**Purpose**: Whether to use global average pooling for sequence-level statistics.

**Options**:
- `true` (default): Use global average pooling
- `false`: Use temporal statistics

**Effect**:
- `true`: Computes global statistics across the entire sequence
- `false`: Computes statistics for each time step

**Example**:
```yaml
dynamic_pruning_config:
  predictor_config:
    use_global_pool: true
```

#### 4.3 `use_std` - Standard Deviation Statistics

**Purpose**: Whether to include standard deviation statistics in addition to mean.

**Options**:
- `true` (default): Include both mean and std statistics
- `false`: Use only mean statistics

**Effect**:
- `true`: Input features doubled (mean + std), more informative
- `false`: Input features unchanged, simpler but less informative

**Example**:
```yaml
dynamic_pruning_config:
  predictor_config:
    use_std: true  # Include std statistics for richer input representation
```

## Training Configuration

### 5. `dynamic_l1_weight` - L1 Regularization Weight

**Purpose**: Controls the strength of L1 regularization on dynamic predictor parameters.

**Default**: `0.001`

**Effect**:
- Higher values: More regularization, sparser predictors
- Lower values: Less regularization, denser predictors

**Example**:
```yaml
# In main training config
dynamic_l1_weight: 0.001
```

### 6. `dynamic_mode` - Training Mode

**Purpose**: Controls whether to use dynamic mode during training.

**Options**:
- `true` (default): Enable dynamic mode
- `false`: Use static mode

**Effect**:
- `true`: Dynamic gating is active during training
- `false`: Only static gating is used

**Example**:
```yaml
# In main training config
dynamic_mode: true
```

## Complete Configuration Example

```yaml
# Enable dynamic pruning
use_dynamic_pruning: true

# Dynamic pruning configuration
dynamic_pruning_config:
  gate_type: "hardconcrete"           # Static gate type
  dynamic_weight: 0.5                 # Balance between static and dynamic
  use_input_gating: true              # Enable input-dependent gating
  predictor_config:
    reduction_ratio: 16               # SE-style bottleneck compression
    use_global_pool: true             # Global pooling for statistics
    use_std: true                     # Include standard deviation

# Training configuration
dynamic_l1_weight: 0.001              # L1 regularization weight
dynamic_mode: true                    # Enable dynamic mode during training

# HardConcrete configuration (for static gates)
init_mean: 0.01
temperature: 1.0
min_temperature: 0.1
temperature_decay: 0.95
temperature_decay_freq: 100
```

## Parameter Interactions

### Static vs Dynamic Balance
- When `dynamic_weight = 0.0`: Pure static pruning, similar to standard structured pruning
- When `dynamic_weight = 1.0`: Pure dynamic pruning, no static gates
- When `dynamic_weight = 0.5`: Balanced approach, combines benefits of both

### Gate Type Effects
- **HardConcrete**: Better for training with L0 regularization, deterministic inference
- **Sigmoid**: Simpler, continuous gates throughout training

### Predictor Capacity
- **Higher `reduction_ratio`**: More efficient, less expressive
- **Lower `reduction_ratio`**: More expressive, less efficient
- **`use_std = true`**: Richer input representation, more parameters

## Best Practices

1. **Start with balanced settings**: `dynamic_weight = 0.5`, `reduction_ratio = 16`
2. **Enable both statistics**: `use_std = true` for richer input representation
3. **Use global pooling**: `use_global_pool = true` for sequence-level decisions
4. **Adjust regularization**: Tune `dynamic_l1_weight` based on sparsity requirements
5. **Monitor training**: Watch for convergence and adjust `dynamic_weight` if needed
