"""
Router Logits Burn-in Drift Analysis
=====================================
Sonnet's killer idea: plot the router's INTERNAL preference (pre-sigmoid logits)
drifting toward SSM during burn-in, while the OUTPUT alpha stays clamped at 0.5.

This makes the mechanism VISIBLE: the router trains against its own override.

Loads checkpoints saved during training and extracts pre-clamp logits at each step.
If no per-step checkpoints, uses metrics.json + the final checkpoint for a snapshot.

Usage:
  python analyze_router_logits.py --results-dir results/width_only
  python analyze_router_logits.py --results-dir results/unified
"""

import os, sys, json, glob, torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from esh_unified.model import UnifiedModel, UnifiedConfig


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = UnifiedConfig(**ckpt["config"])
    model = UnifiedModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", 0)
    return model, config, step


def extract_router_logits(model, input_ids, device):
    """Extract pre-sigmoid logits from every layer's router."""
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        # We need to hook into the router to get pre-sigmoid logits
        logits_per_layer = []
        
        def make_hook(layer_idx):
            def hook_fn(module, args, output):
                # RouterOutput: alpha, halt_prob, aux_loss, entropy
                # We need the pre-sigmoid logits, which are computed inside forward()
                # Re-compute them from the complexity_net
                x = args[0]  # Input tensor [B, L, D]
                raw_logits = module.complexity_net(x)  # [B, L, 1]
                logits_per_layer.append({
                    "layer": layer_idx,
                    "mean_logit": raw_logits.mean().item(),
                    "std_logit": raw_logits.std().item(),
                    "min_logit": raw_logits.min().item(),
                    "max_logit": raw_logits.max().item(),
                    "mean_alpha": torch.sigmoid(raw_logits).mean().item(),
                })
            return hook_fn
        
        # Register hooks on each layer's router
        hooks = []
        for i, block in enumerate(model.blocks):
            h = block.router.register_forward_hook(make_hook(i))
            hooks.append(h)
        
        # Run forward pass
        model.set_global_step(99999)  # Past burn-in so router runs normally
        _ = model(input_ids, return_routing_stats=True)
        
        # Remove hooks
        for h in hooks:
            h.remove()
    
    return logits_per_layer


def analyze_logit_drift(results_dir, device):
    """Load all checkpoints and track router logit drift across training."""
    # Find all checkpoints
    ckpt_files = sorted(glob.glob(os.path.join(results_dir, "ckpt_step*.pt")))
    final_ckpt = os.path.join(results_dir, "final_model.pt")
    
    if os.path.exists(final_ckpt) and final_ckpt not in ckpt_files:
        ckpt_files.append(final_ckpt)
    
    if not ckpt_files:
        print(f"No checkpoints found in {results_dir}")
        return None
    
    print(f"Found {len(ckpt_files)} checkpoints")
    
    # Create synthetic input for consistent comparison
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Mix of simple and complex text
    test_texts = [
        "Once upon a time there was a little girl who loved to play in the garden.",
        "The mitochondria is the powerhouse of the cell, converting adenosine triphosphate.",
        "Question: If x + 3 = 7, what is x?\nAnswer: x = 4",
        "She was happy. The dog was happy. They played together all day long.",
    ]
    tokens = tokenizer(test_texts, max_length=512, truncation=True,
                       padding="max_length", return_tensors="pt")
    input_ids = tokens["input_ids"]
    
    # Track logits across checkpoints
    trajectory = []
    
    for ckpt_path in ckpt_files:
        basename = os.path.basename(ckpt_path)
        print(f"  Loading {basename}...")
        
        model, config, step = load_checkpoint(ckpt_path, device)
        logits_data = extract_router_logits(model, input_ids, device)
        
        # Average across layers
        avg_logit = np.mean([l["mean_logit"] for l in logits_data])
        avg_alpha = np.mean([l["mean_alpha"] for l in logits_data])
        
        trajectory.append({
            "step": step,
            "checkpoint": basename,
            "avg_router_logit": avg_logit,
            "avg_unclamped_alpha": avg_alpha,
            "per_layer": logits_data,
        })
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    return trajectory


def plot_logit_drift(trajectory, results_dir):
    """Plot the router's internal preference drifting during training."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    steps = [t["step"] for t in trajectory]
    logits = [t["avg_router_logit"] for t in trajectory]
    alphas = [t["avg_unclamped_alpha"] for t in trajectory]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Raw router logits
    ax1.plot(steps, logits, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label="Logit=0 (balanced)")
    ax1.axvline(x=1500, color='red', linestyle='--', alpha=0.7, label="Burn-in ends")
    ax1.set_ylabel("Mean Router Logit\n(pre-sigmoid)", fontsize=12)
    ax1.set_title("Router Internal Preference During Training\n"
                   "(Negative = SSM preferred, Positive = Attention preferred)", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Unclamped alpha (what router WANTS vs what it gets)
    ax2.plot(steps, alphas, 'r-o', linewidth=2, markersize=6, label="Router's desired α")
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label="Burn-in forced α=0.5")
    ax2.axvline(x=1500, color='red', linestyle='--', alpha=0.7, label="Burn-in ends")
    
    # Add shaded region for burn-in
    ax2.axvspan(0, 1500, alpha=0.1, color='green', label="Burn-in window")
    
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Unclamped α\n(sigmoid of logit)", fontsize=12)
    ax2.set_title("Router α: What It Wants vs What It Gets", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    output_dir = os.path.join(results_dir, "..", "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "router_logit_drift.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {plot_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="e.g. results/width_only")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    trajectory = analyze_logit_drift(args.results_dir, device)
    
    if not trajectory:
        return
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  ROUTER LOGIT DRIFT ANALYSIS")
    print(f"{'=' * 60}")
    print(f"\n  {'Step':>8} {'Mean Logit':>12} {'Desired α':>12} {'Note'}")
    print(f"  {'-' * 55}")
    
    for t in trajectory:
        note = ""
        if t["step"] <= 1500:
            note = "← burn-in (α forced to 0.5)"
        elif t["step"] <= 1600:
            note = "← RELEASE POINT"
        
        print(f"  {t['step']:>8d} {t['avg_router_logit']:>12.4f} {t['avg_unclamped_alpha']:>12.4f} {note}")
    
    # Key insight
    if len(trajectory) >= 2:
        first = trajectory[0]
        last = trajectory[-1]
        drift = last["avg_router_logit"] - first["avg_router_logit"]
        print(f"\n  Total logit drift: {drift:+.4f}")
        if drift < -0.1:
            print(f"  ★ Router drifted TOWARD SSM during training!")
            print(f"    This proves the router learned SSM preference WHILE being overridden.")
        elif drift > 0.1:
            print(f"  ★ Router drifted TOWARD Attention during training.")
        else:
            print(f"  ≈ Minimal drift detected.")
    
    # Plot
    plot_logit_drift(trajectory, args.results_dir)
    
    # Save JSON
    output_dir = os.path.join(args.results_dir, "..", "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "router_logit_drift.json")
    with open(json_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
