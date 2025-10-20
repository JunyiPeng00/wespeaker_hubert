#!/usr/bin/env python3
"""Export dynamic pruning model to static deployment format.

This script converts a dynamically pruned model to a static format suitable
for deployment, where the dynamic gating is replaced with fixed masks.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

from wespeaker.utils.prune_utils import set_dynamic_pruning_mode
from wespeaker.frontend.wav2vec2.model import wav2vec2_model, wavlm_model


def export_static_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Export static masks from all dynamic gates in the model.
    
    Args:
        model: PyTorch model containing dynamic pruning gates.
    
    Returns:
        Dictionary mapping module names to their static masks.
    """
    masks = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'export_static_mask'):
            mask = module.export_static_mask()
            masks[name] = mask.detach().cpu()
            print(f"Exported mask for {name}: shape={mask.shape}, "
                  f"sparsity={1 - mask.mean().item():.3f}")
    
    return masks


def convert_to_static_model(model: nn.Module) -> nn.Module:
    """Convert dynamic pruning model to static model.
    
    This function sets all dynamic gates to static mode and returns the model
    ready for static deployment.
    
    Args:
        model: PyTorch model with dynamic pruning.
    
    Returns:
        Model with dynamic gates set to static mode.
    """
    # Set all dynamic gates to static mode
    set_dynamic_pruning_mode(model, dynamic=False)
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def save_static_model(
    model: nn.Module,
    output_path: str,
    masks: Optional[Dict[str, torch.Tensor]] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Save static model and associated data.
    
    Args:
        model: Static model to save.
        output_path: Path to save the model.
        masks: Optional dictionary of exported masks.
        config: Optional configuration dictionary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save data
    save_data = {
        'state_dict': model.state_dict(),
        'model_config': config or {},
        'export_info': {
            'static_model': True,
            'dynamic_pruning_exported': True,
        }
    }
    
    if masks is not None:
        save_data['static_masks'] = masks
        save_data['export_info']['num_masks'] = len(masks)
    
    # Save model
    torch.save(save_data, output_path)
    print(f"Static model saved to: {output_path}")
    
    # Save masks separately if provided
    if masks is not None:
        masks_path = output_path.with_suffix('.masks.pt')
        torch.save(masks, masks_path)
        print(f"Static masks saved to: {masks_path}")


def load_dynamic_model(checkpoint_path: str, device: str = "cpu") -> tuple[nn.Module, Dict[str, Any]]:
    """Instantiate a model from a WeSpeaker-format checkpoint.
    
    This supports the WeSpeaker upstream SSL checkpoints used by the
    HuggingfaceFrontend builder, which store 'config' and 'state_dict'.
    
    Args:
        checkpoint_path: Path to the WeSpeaker-format checkpoint (.pth/.bin).
        device: Device to place the model on.
    
    Returns:
        (model, config) where model is an nn.Module ready for inference and
        config is the model configuration dict stored in the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'config' not in ckpt or 'state_dict' not in ckpt:
        raise ValueError(
            "Checkpoint must contain 'config' and 'state_dict' keys for export."
        )
    config: Dict[str, Any] = ckpt['config']

    # Pick model constructor based on model name hints inside config
    model_name = str(config.get('model_name', config.get('upstream_name', ''))).lower()
    if "wavlm" in model_name:
        model = wavlm_model(**config, hard_concrete_config=None)
    else:
        # Default to wav2vec2 family
        model = wav2vec2_model(**config, hard_concrete_config=None)

    missing = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f"Loaded checkpoint: missing={len(missing.missing_keys)}, unexpected={len(missing.unexpected_keys)}")

    model.to(device)
    return model, config


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description="Export dynamic pruning model to static format")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to dynamic pruning checkpoint"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output path for static model"
    )
    parser.add_argument(
        "--export-masks", 
        action="store_true",
        help="Export static masks separately"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu",
        help="Device to use for export (cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dynamic Pruning Model Export Tool")
    print("=" * 60)
    
    try:
        device = args.device
        use_cuda = device.startswith('cuda') and torch.cuda.is_available()
        if device.startswith('cuda') and not use_cuda:
            print("CUDA requested but not available; falling back to CPU.")
            device = 'cpu'

        # 1) Load and instantiate dynamic model
        print(f"Loading dynamic model from: {args.checkpoint}")
        model, config = load_dynamic_model(args.checkpoint, device=device)

        # 2) Convert to static (disable dynamic gates) and export masks if requested
        print("Converting model to static (disabling dynamic gates)...")
        model_static = convert_to_static_model(model)

        masks = None
        if args.export_masks:
            print("Exporting static masks...")
            masks = export_static_masks(model_static)
        
        # 3) Save static model and optional masks
        print(f"Saving static model to: {args.output}")
        save_static_model(model_static, args.output, masks=masks, config=config)
        
        print("\n" + "=" * 60)
        print("✅ Model export completed successfully!")
        print("=" * 60)
        print(f"Static model saved to: {args.output}")
        if args.export_masks:
            print(f"Static masks saved to: {Path(args.output).with_suffix('.masks.pt')}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Export failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
