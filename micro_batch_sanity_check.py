"""
Micro-Batch Sanity Check: Does Hard Routing Actually Work?
==========================================================

This is the STERILE test. We train on exactly 20 sequences:
  - 10 sequential patterns (ABABAB...) → SSM is mathematically perfect
  - 10 associative recall patterns      → Attention is strictly required

If the router learns to send:
  - Dataset A to SSM (attn_ratio → 0.0)
  - Dataset B to Attention (attn_ratio → 1.0)

...then the hard routing architecture WORKS, and we scale up.

If both stay at 50/50 after 500 steps, the architecture is still broken.

Usage:
    python micro_batch_sanity_check.py
    
    # On server:
    PYTHONUNBUFFERED=1 python -u micro_batch_sanity_check.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from esh_unified.model import UnifiedModel, UnifiedConfig


def create_micro_batch(vocab_size=1000, seq_len=32, batch_size=10):
    """
    Create two perfectly separable datasets:
    
    Dataset A (sequential): "10 11 10 11 10 11..."
      → Conv/SSM is mathematically perfect for this (local pattern)
      → Attention is overkill
    
    Dataset B (associative recall): random tokens, but token[-1] = token[5]
      → Requires looking back 27 positions to predict final token
      → Conv (local) CANNOT do this; Attention CAN
    """
    # Dataset A: Pure sequential repetition (SSM/Conv territory)
    seq_A = torch.zeros((batch_size, seq_len), dtype=torch.long)
    seq_A[:, 0::2] = 10
    seq_A[:, 1::2] = 11
    
    # Dataset B: Associative recall (Attention territory)
    # Random tokens, but the last token must equal token at position 5
    # This requires long-range attention — conv cannot solve it
    seq_B = torch.randint(20, vocab_size, (batch_size, seq_len))
    seq_B[:, -1] = seq_B[:, 5]  # token[-1] = token[5]
    
    # Combine: first half is A, second half is B
    x = torch.cat([seq_A, seq_B], dim=0)  # [20, seq_len]
    
    # Next-token prediction targets
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = -100  # Ignore last position
    
    return x, y


def get_per_dataset_attn_ratio(model, x, batch_size=10):
    """Compute attention ratio separately for dataset A and B."""
    model.eval()
    with torch.no_grad():
        _ = model(x, return_routing_stats=False)
    
    # Gather attn_masks from all layers
    attn_ratios_A = []
    attn_ratios_B = []
    for block in model.blocks:
        mask = block.current_attn_mask  # [20, L, 1]
        if mask is not None:
            # Dataset A = first 10 sequences, Dataset B = last 10
            attn_ratios_A.append(mask[:batch_size].mean().item())
            attn_ratios_B.append(mask[batch_size:].mean().item())
    
    mean_A = sum(attn_ratios_A) / len(attn_ratios_A) if attn_ratios_A else 0.5
    mean_B = sum(attn_ratios_B) / len(attn_ratios_B) if attn_ratios_B else 0.5
    model.train()
    return mean_A, mean_B


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-cost", type=float, default=0.01,
                        help="Compute penalty weight (tax on Attention)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Gumbel-Softmax temperature")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Small model for micro-batch (fast iteration)
    config = UnifiedConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        n_experts=4,
        max_seq_len=64,
        use_checkpoint=False,
        enable_width_routing=True,
        enable_depth_routing=False,
        max_ponder_steps=1,
        use_moe=False,          # No MoE — isolate routing behavior
        compute_penalty_weight=args.lambda_cost,
        router_temperature=args.temperature,
    )
    
    model = UnifiedModel(config).to(device)
    model.print_model_info()
    
    # Create the micro-batch (fixed data — we overfit deliberately)
    x, y = create_micro_batch(vocab_size=1000, seq_len=32, batch_size=10)
    x, y = x.to(device), y.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    print(f"\n{'='*70}")
    print(f"  MICRO-BATCH SANITY CHECK")
    print(f"  20 sequences: 10 sequential (SSM) + 10 associative (Attention)")
    print(f"  Training for {args.steps} steps with λ_cost = {args.lambda_cost}")
    print(f"{'='*70}")
    print(f"\n  {'Step':>6} | {'LM Loss':>8} | {'Comp.Pen':>8} | {'Total':>8} | "
          f"{'Attn%(A)':>8} | {'Attn%(B)':>8} | {'Verdict':>20}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*20}")
    
    for step in range(args.steps):
        model.train()
        
        outputs = model(x, labels=y)
        lm_loss = outputs["loss"]
        compute_penalty = outputs["compute_penalty"]
        
        total_loss = lm_loss + args.lambda_cost * compute_penalty
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Log every 25 steps
        if (step + 1) % 25 == 0:
            attn_A, attn_B = get_per_dataset_attn_ratio(model, x, batch_size=10)
            
            # Determine verdict
            if attn_A < 0.3 and attn_B > 0.7:
                verdict = "★ SPECIALIZING!"
            elif abs(attn_B - attn_A) > 0.15:
                verdict = "✓ Weak signal"
            else:
                verdict = "✗ Dead (50/50)"
            
            print(f"  {step+1:>6} | {lm_loss.item():>8.4f} | {compute_penalty.item():>8.4f} | "
                  f"{total_loss.item():>8.4f} | {attn_A:>7.1%} | {attn_B:>7.1%} | {verdict}",
                  flush=True)
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    
    attn_A, attn_B = get_per_dataset_attn_ratio(model, x, batch_size=10)
    delta = attn_B - attn_A
    
    print(f"  Attention usage (Dataset A — sequential):     {attn_A:.1%}")
    print(f"  Attention usage (Dataset B — associative):    {attn_B:.1%}")
    print(f"  Delta (B - A):                                {delta:+.1%}")
    
    if attn_A < 0.3 and attn_B > 0.7:
        print(f"\n  ★★★ ROUTING WORKS! The router learned to specialize!")
        print(f"      Sequential tokens → SSM (cheap)")
        print(f"      Associative recall → Attention (expensive but necessary)")
        print(f"\n  NEXT: Scale up to full model + real data.")
    elif delta > 0.15:
        print(f"\n  ✓ WEAK SPECIALIZATION: Signal present but not strong.")
        print(f"    Try: --steps 1000 or --lambda-cost 0.005")
    else:
        print(f"\n  ✗ ROUTING FAILED: No specialization detected.")
        print(f"    The architecture may need further changes.")
    
    # Per-layer breakdown
    print(f"\n  Per-layer attention ratio:")
    print(f"  {'Layer':>6} | {'Dataset A':>10} | {'Dataset B':>10} | {'Delta':>10}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10}")
    for i, block in enumerate(model.blocks):
        mask = block.current_attn_mask
        if mask is not None:
            a = mask[:10].mean().item()
            b = mask[10:].mean().item()
            print(f"  {i:>6} | {a:>9.1%} | {b:>9.1%} | {b-a:>+9.1%}")


if __name__ == "__main__":
    main()
