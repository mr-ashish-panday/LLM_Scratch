"""
Inspect Run 4 checkpoints to measure TRUE ponder depth.
Uses the corrected sum-of-remainders formula.
"""
import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import create_mixed_eval_dataloader
from esh_unified.model import UnifiedModel, UnifiedConfig


def inspect(mode):
    ckpt_path = os.path.join("results", mode, "final_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"[{mode}] No checkpoint found at {ckpt_path}")
        return

    print(f"[{mode}] Loading checkpoint...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = UnifiedConfig(**checkpoint["config"])
    model = UnifiedModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    loader = create_mixed_eval_dataloader(
        tokenizer, batch_size=4, max_length=512, cache_dir="./data_cache"
    )

    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)

    print(f"[{mode}] Running inference on 1 batch...")
    with torch.no_grad():
        outputs = model(input_ids, return_routing_stats=True)

    # routing_stats is a LIST of dicts (one per layer)
    all_stats = outputs["routing_stats"]

    if not all_stats:
        print(f"[{mode}] No routing stats returned.")
        return

    # Average across layers
    avg_alpha = sum(s.get("alpha_mean", 0.5) for s in all_stats) / len(all_stats)
    avg_ponder = sum(s.get("avg_ponder_steps", 1.0) for s in all_stats) / len(all_stats)
    avg_halt = sum(s.get("halt_prob_mean", 1.0) for s in all_stats) / len(all_stats)
    avg_attn = sum(s.get("attention_ratio", 0.5) for s in all_stats) / len(all_stats)

    print(f"[{mode}] RESULTS (averaged across {len(all_stats)} layers):")
    print(f"  Alpha Mean:       {avg_alpha:.4f}")
    print(f"  Attention Ratio:  {avg_attn:.4f}")
    print(f"  Avg Ponder Steps: {avg_ponder:.4f}")
    print(f"  Halt Prob Mean:   {avg_halt:.4f}")

    # Per-layer breakdown
    print(f"\n  Per-layer ponder steps:")
    for i, s in enumerate(all_stats):
        ponder = s.get("avg_ponder_steps", 1.0)
        alpha = s.get("alpha_mean", 0.5)
        print(f"    Layer {i}: ponder={ponder:.3f}  alpha={alpha:.3f}")

    print("-" * 60)


if __name__ == "__main__":
    modes = ["baseline", "width_only", "depth_only", "unified"]
    print("=" * 60)
    print("INSPECTING RUN 4 CHECKPOINTS")
    print("=" * 60 + "\n")
    for m in modes:
        inspect(m)
