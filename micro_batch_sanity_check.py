"""
Micro-Batch Sanity Check v2: Fresh Data + No-Penalty Warmup
============================================================

v1 FAILURE ANALYSIS:
  The model memorized all 20 fixed sequences using SSM alone (loss→0.003).
  With only 20 sequences, any path can memorize. Compute penalty pushed
  everything to SSM since it's "free" and both paths could memorize equally.

v2 FIXES:
  1. Generate FRESH random sequences every step (no memorization possible)
  2. Phase 1 (steps 0-200): λ=0 (let model discover it needs Attention)
  3. Phase 2 (steps 200-500): λ=0.01 (add penalty, see if SSM takes over for easy tasks)

Usage:
    python micro_batch_sanity_check.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from esh_unified.model import UnifiedModel, UnifiedConfig


def create_fresh_batch(vocab_size=1000, seq_len=32, batch_size=10, device="cpu"):
    """
    Generate FRESH random data every call (no memorization possible).

    Dataset A (sequential): alternating token pairs [a, b, a, b, ...]
      → Next token is perfectly predictable from the previous one
      → Conv/SSM handles this trivially (local pattern)

    Dataset B (associative recall): random tokens, but at fixed positions
      token[15] = token[3], token[25] = token[7], token[31] = token[11]
      → Requires looking back 12+ positions to predict these tokens
      → Conv (kernel=4) CANNOT see back that far; Attention CAN
    """
    # Dataset A: Alternating pairs (different pair per sequence, per call)
    seq_A = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    for i in range(batch_size):
        a = torch.randint(10, 100, (1,)).item()
        b = torch.randint(100, 200, (1,)).item()
        seq_A[i, 0::2] = a
        seq_A[i, 1::2] = b

    # Dataset B: Random tokens with long-range copy requirements
    seq_B = torch.randint(200, vocab_size, (batch_size, seq_len), device=device)
    # Force long-range dependencies (must copy from distant positions)
    seq_B[:, 15] = seq_B[:, 3]    # copy from position 3 → 15  (distance 12)
    seq_B[:, 25] = seq_B[:, 7]    # copy from position 7 → 25  (distance 18)
    seq_B[:, 31] = seq_B[:, 11]   # copy from position 11 → 31 (distance 20)

    x = torch.cat([seq_A, seq_B], dim=0)  # [20, seq_len]

    # Next-token prediction targets
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = -100

    return x, y


def get_per_dataset_attn_ratio(model, x, batch_size=10):
    """Compute attention ratio separately for dataset A and B."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(x, return_routing_stats=False)

    attn_ratios_A = []
    attn_ratios_B = []
    for block in model.blocks:
        mask = block.current_attn_mask
        if mask is not None:
            attn_ratios_A.append(mask[:batch_size].mean().item())
            attn_ratios_B.append(mask[batch_size:].mean().item())

    mean_A = sum(attn_ratios_A) / len(attn_ratios_A) if attn_ratios_A else 0.5
    mean_B = sum(attn_ratios_B) / len(attn_ratios_B) if attn_ratios_B else 0.5
    if was_training:
        model.train()
    return mean_A, mean_B


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-cost", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200,
                        help="Steps with lambda=0 before enabling penalty")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
        use_moe=False,
        compute_penalty_weight=args.lambda_cost,
        router_temperature=args.temperature,
    )

    model = UnifiedModel(config).to(device)
    model.print_model_info()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"\n{'='*70}")
    print(f"  MICRO-BATCH SANITY CHECK v2 (Fresh Data + Warmup)")
    print(f"  20 fresh sequences each step (no memorization)")
    print(f"  Steps 0-{args.warmup_steps}: λ=0 (discover routing)")
    print(f"  Steps {args.warmup_steps}-{args.steps}: λ={args.lambda_cost} (add penalty)")
    print(f"{'='*70}")
    print(f"\n  {'Step':>6} | {'LM Loss':>8} | {'λ_eff':>6} | {'Pen':>7} | "
          f"{'Attn%(A)':>8} | {'Attn%(B)':>8} | {'Verdict':>20}")
    print(f"  {'-'*6} | {'-'*8} | {'-'*6} | {'-'*7} | {'-'*8} | {'-'*8} | {'-'*20}")

    for step in range(args.steps):
        model.train()

        # Fresh data each step — NO MEMORIZATION
        x, y = create_fresh_batch(vocab_size=1000, seq_len=32, batch_size=10, device=device)

        outputs = model(x, labels=y)
        lm_loss = outputs["loss"]
        compute_penalty = outputs["compute_penalty"]

        # Phase 1: no penalty (let model discover routing)
        # Phase 2: add penalty (push easy tokens to SSM)
        lambda_eff = args.lambda_cost if step >= args.warmup_steps else 0.0
        total_loss = lm_loss + lambda_eff * compute_penalty

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 25 == 0:
            # Evaluate on a fixed eval batch (same every time for fair comparison)
            torch.manual_seed(9999)
            eval_x, _ = create_fresh_batch(vocab_size=1000, seq_len=32, batch_size=10, device=device)
            attn_A, attn_B = get_per_dataset_attn_ratio(model, eval_x, batch_size=10)

            if attn_A < 0.3 and attn_B > 0.7:
                verdict = "★ SPECIALIZING!"
            elif abs(attn_B - attn_A) > 0.15:
                verdict = "✓ Weak signal"
            elif attn_B > attn_A + 0.05:
                verdict = "~ Slight signal"
            else:
                verdict = "✗ Dead"

            phase = "PEN" if step >= args.warmup_steps else "FREE"
            print(f"  {step+1:>6} | {lm_loss.item():>8.4f} | {lambda_eff:>5.3f} | "
                  f"{compute_penalty.item():>6.1%} | {attn_A:>7.1%} | {attn_B:>7.1%} | "
                  f"{verdict} [{phase}]", flush=True)

    # Final
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")

    torch.manual_seed(9999)
    eval_x, _ = create_fresh_batch(vocab_size=1000, seq_len=32, batch_size=10, device=device)
    attn_A, attn_B = get_per_dataset_attn_ratio(model, eval_x, batch_size=10)
    delta = attn_B - attn_A

    print(f"  Attention usage (Dataset A — sequential):     {attn_A:.1%}")
    print(f"  Attention usage (Dataset B — associative):    {attn_B:.1%}")
    print(f"  Delta (B - A):                                {delta:+.1%}")

    if attn_A < 0.3 and attn_B > 0.7:
        print(f"\n  ★★★ ROUTING WORKS!")
    elif delta > 0.15:
        print(f"\n  ✓ WEAK SPECIALIZATION — try more steps or different λ")
    else:
        print(f"\n  ✗ ROUTING FAILED")
        print(f"    Possible causes:")
        print(f"    - Conv placeholder CAN solve recall via 4-layer stacking")
        print(f"    - Need larger seq_len or harder recall task")
        print(f"    - Try: --steps 1000 --lambda-cost 0")

    # Per-layer
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
