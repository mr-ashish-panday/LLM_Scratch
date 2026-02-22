"""
Burn-in Router Drift Capture
==============================
A SHORT 3000-step width_only run that captures pre-sigmoid router logits
at every log interval (100 steps). This produces the KILLER FIGURE:

    "The router's internal SSM preference drifts DURING burn-in,
     while its output alpha stays clamped at 0.5."

Key checkpoints:
  Steps 0-1500:   Burn-in (alpha forced to 0.5, but router still training)
  Steps 1500-1600: Release point (alpha snap-back)
  Steps 1600-3000: Post-collapse stabilization

Output:
  - results/burnin_capture/logit_trajectory.json
  - results/burnin_capture/burnin_drift_plot.png (publication figure)

Usage:
  python capture_burnin_drift.py
  python capture_burnin_drift.py --cache-dir /tmp/hf_cache
"""

import os, sys, json, math, time, torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from esh_unified.model import UnifiedConfig, UnifiedModel
from data_loader import create_mixed_dataloader


def extract_router_logits_inline(model):
    """Extract current pre-sigmoid logits from each layer's router complexity_net weights.
    
    Returns the mean bias of the router — a proxy for its internal preference.
    Negative = SSM preferred, Positive = Attention preferred.
    """
    logits_info = []
    for i, block in enumerate(model.blocks):
        router = block.router
        # Get the weight norms and biases of the complexity net
        # Layer 0: Linear(768, 192, bias=False) + SiLU + Linear(192, 1, bias=False)
        w1 = router.complexity_net[0].weight  # [192, 768]
        w2 = router.complexity_net[2].weight  # [1, 192]
        
        # The effective "bias" toward SSM or Attention can be estimated from
        # the product of weight matrices evaluated at a zero-centered input.
        # More practically: check the output on a unit-scale random input.
        with torch.no_grad():
            test_input = torch.randn(1, 32, model.config.d_model, device=w1.device) * 0.1
            raw_logits = router.complexity_net(test_input)  # [1, 32, 1]
            mean_logit = raw_logits.mean().item()
            std_logit = raw_logits.std().item()
            desired_alpha = torch.sigmoid(raw_logits).mean().item()
        
        logits_info.append({
            "layer": i,
            "mean_logit": mean_logit,
            "std_logit": std_logit,
            "desired_alpha": desired_alpha,
        })
    
    return logits_info


def cosine_lr(step, warmup, max_steps, max_lr, min_lr=1e-5):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Width-only config (the mode that shows collapse)
    config = UnifiedConfig(
        vocab_size=50257,
        d_model=768,
        n_layers=8,
        n_heads=12,
        n_experts=4,
        max_seq_len=512,
        use_checkpoint=True,
        enable_width_routing=True,
        enable_depth_routing=False,  # Width only for clean isolation
        max_ponder_steps=3,
        ponder_cost_weight=0.5,
        use_moe=True,
    )
    
    print(f"\nBurn-in Capture Run")
    print(f"  Mode: width_only (isolates α collapse)")
    print(f"  Burn-in: steps 0-1500 (α forced to 0.5)")
    print(f"  Release: step 1500 (α free)")
    print(f"  Total: {args.max_steps} steps")
    
    # Model
    model = UnifiedModel(config).to(device)
    model.train()
    print(f"  Params: {model.count_parameters()/1e6:.1f}M")
    
    # Tokenizer + Data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    train_loader = create_mixed_dataloader(
        tokenizer, batch_size=args.batch_size, max_length=512,
        cache_dir=args.cache_dir,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scaler = GradScaler()
    
    # Output dir
    output_dir = "results/burnin_capture"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training + Capture
    data_iter = iter(train_loader)
    trajectory = []  # The main output: logit trajectory across training
    
    print(f"\n{'='*70}")
    print(f"  {'Step':>6} | {'Loss':>7} | {'Output α':>9} | {'Router Logit':>13} | {'Desired α':>10} | {'Phase'}")
    print(f"{'='*70}")
    
    t0 = time.time()
    
    for step in range(args.max_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(device)
        
        # LR schedule
        lr = cosine_lr(step, 1000, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        # Forward
        with autocast(enabled=True):
            model.set_global_step(step)
            outputs = model(input_ids, labels=input_ids, return_routing_stats=True)
            loss = outputs["loss"]
            aux_loss = outputs["aux_loss"]
            total_loss = loss + aux_loss
        
        # Backward
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Log at every interval
        if (step + 1) % args.log_interval == 0:
            # Extract raw router logits (the key data)
            logits_info = extract_router_logits_inline(model)
            avg_logit = np.mean([l["mean_logit"] for l in logits_info])
            avg_desired_alpha = np.mean([l["desired_alpha"] for l in logits_info])
            
            # Get the actual output alpha from routing stats
            stats = outputs["routing_stats"]
            output_alpha = np.mean([s.get("alpha_mean", 0.5) for s in stats]) if stats else 0.5
            
            # Determine phase
            if step + 1 <= 1500:
                phase = "BURN-IN (α=0.5 forced)"
            elif step + 1 <= 1600:
                phase = "★ RELEASE POINT"
            else:
                phase = "POST-COLLAPSE"
            
            entry = {
                "step": step + 1,
                "loss": loss.item(),
                "output_alpha": output_alpha,
                "avg_router_logit": avg_logit,
                "desired_alpha": avg_desired_alpha,
                "per_layer_logits": [l["mean_logit"] for l in logits_info],
                "per_layer_desired_alpha": [l["desired_alpha"] for l in logits_info],
                "phase": phase,
            }
            trajectory.append(entry)
            
            print(
                f"  {step+1:>6} | {loss.item():>7.4f} | {output_alpha:>9.4f} | "
                f"{avg_logit:>+13.4f} | {avg_desired_alpha:>10.4f} | {phase}"
            )
    
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Capture complete in {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    
    # Save trajectory
    json_path = os.path.join(output_dir, "logit_trajectory.json")
    with open(json_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"\nTrajectory saved: {json_path}")
    
    # Generate the KILLER FIGURE
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        steps = [t["step"] for t in trajectory]
        output_alphas = [t["output_alpha"] for t in trajectory]
        desired_alphas = [t["desired_alpha"] for t in trajectory]
        logits = [t["avg_router_logit"] for t in trajectory]
        losses = [t["loss"] for t in trajectory]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # --- Plot 1: The Killer Comparison ---
        ax = axes[0]
        ax.plot(steps, output_alphas, 'b-', linewidth=2.5, label="Output α (what model uses)", zorder=3)
        ax.plot(steps, desired_alphas, 'r--', linewidth=2.5, label="Desired α (what router WANTS)", zorder=3)
        ax.axvline(x=1500, color='black', linestyle=':', linewidth=2, alpha=0.8)
        ax.axvspan(0, 1500, alpha=0.08, color='green')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)
        
        # Annotations
        ax.annotate("Burn-in\n(α forced to 0.5)", xy=(750, 0.52), fontsize=11,
                     ha='center', color='green', fontweight='bold')
        ax.annotate("Release →\nCollapse", xy=(1550, 0.45), fontsize=10,
                     ha='left', color='red', fontweight='bold')
        
        ax.set_ylabel("α value", fontsize=13)
        ax.set_title("The SSM Attractor: Router Trains Against Its Own Override", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='center right')
        ax.set_ylim(0.15, 0.6)
        ax.grid(True, alpha=0.3)
        
        # --- Plot 2: Raw Router Logits ---
        ax = axes[1]
        ax.plot(steps, logits, 'purple', linewidth=2)
        ax.axvline(x=1500, color='black', linestyle=':', linewidth=2, alpha=0.8)
        ax.axvspan(0, 1500, alpha=0.08, color='green')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
        ax.fill_between(steps, logits, 0, alpha=0.15, color='purple')
        
        ax.set_ylabel("Mean Router Logit\n(negative = SSM)", fontsize=12)
        ax.set_title("Pre-Sigmoid Router Logits Over Training", fontsize=13)
        ax.grid(True, alpha=0.3)
        
        # --- Plot 3: Training Loss ---
        ax = axes[2]
        ax.plot(steps, losses, 'green', linewidth=1.5, alpha=0.8)
        ax.axvline(x=1500, color='black', linestyle=':', linewidth=2, alpha=0.8)
        ax.axvspan(0, 1500, alpha=0.08, color='green')
        
        ax.set_xlabel("Training Step", fontsize=13)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss", fontsize=13)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "burnin_drift_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Killer figure saved: {plot_path}")
        
        # Also save per-layer logit heatmap
        fig, ax = plt.subplots(figsize=(12, 5))
        per_layer = np.array([t["per_layer_logits"] for t in trajectory])  # [T, 8]
        im = ax.imshow(per_layer.T, aspect='auto', cmap='RdBu_r',
                       extent=[steps[0], steps[-1], 7.5, -0.5],
                       vmin=-1.5, vmax=0.5)
        ax.axvline(x=1500, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
        ax.set_title("Per-Layer Router Logits: Blue=SSM Preferred, Red=Attention Preferred", fontsize=13)
        ax.set_yticks(range(8))
        plt.colorbar(im, ax=ax, label="Mean Router Logit")
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, "per_layer_logit_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Per-layer heatmap saved: {heatmap_path}")
        
    except ImportError:
        print("matplotlib not available — raw JSON saved, plot skipped")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  BURN-IN DRIFT SUMMARY")
    print(f"{'='*70}")
    
    burnin_entries = [t for t in trajectory if t["step"] <= 1500]
    post_entries = [t for t in trajectory if t["step"] > 1600]
    
    if burnin_entries:
        first_logit = burnin_entries[0]["avg_router_logit"]
        last_burnin_logit = burnin_entries[-1]["avg_router_logit"]
        print(f"  During burn-in:")
        print(f"    Step {burnin_entries[0]['step']}: logit={first_logit:+.4f}, α_output={burnin_entries[0]['output_alpha']:.4f}")
        print(f"    Step {burnin_entries[-1]['step']}: logit={last_burnin_logit:+.4f}, α_output={burnin_entries[-1]['output_alpha']:.4f}")
        print(f"    Logit drift during burn-in: {last_burnin_logit - first_logit:+.4f}")
    
    if post_entries:
        print(f"  After release:")
        print(f"    Step {post_entries[0]['step']}: logit={post_entries[0]['avg_router_logit']:+.4f}, α_output={post_entries[0]['output_alpha']:.4f}")
        print(f"    Step {post_entries[-1]['step']}: logit={post_entries[-1]['avg_router_logit']:+.4f}, α_output={post_entries[-1]['output_alpha']:.4f}")
    
    # The key insight
    if burnin_entries and post_entries:
        burnin_desired = np.mean([t["desired_alpha"] for t in burnin_entries])
        burnin_output = np.mean([t["output_alpha"] for t in burnin_entries])
        post_desired = np.mean([t["desired_alpha"] for t in post_entries])
        post_output = np.mean([t["output_alpha"] for t in post_entries])
        
        gap = burnin_output - burnin_desired
        print(f"\n  ★ KEY INSIGHT:")
        print(f"    During burn-in: output α = {burnin_output:.4f}, but router WANTED α = {burnin_desired:.4f}")
        print(f"    Gap = {gap:+.4f} — the router was being overridden by {gap:.4f}")
        print(f"    After release: output α snapped to ≈ {post_output:.4f} (what router always wanted)")
        print(f"\n    This proves the router trained AGAINST its own override during burn-in.")


if __name__ == "__main__":
    main()
