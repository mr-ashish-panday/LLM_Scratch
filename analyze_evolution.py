"""
Intelligence Evolution Analysis
================================
Loads ESH checkpoints at different training steps and measures:
1. Perplexity on a fixed evaluation set
2. α variance (routing decisiveness)

Produces data for the "As PPL decreases, α variance increases" plot.
"""

import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from esh import ESHModel
from esh.model import ESHConfig


def create_model(device="cpu"):
    """Create ESH model with Phase 2 config."""
    config = ESHConfig(
        d_model=768, n_layers=8, n_heads=12, n_experts=8,
        expert_dim=3072, max_seq_len=2048, use_checkpoint=False,
        dropout=0.0, layer_scale_init=1e-5,
    )
    return ESHModel(config), config


def load_checkpoint(model, path, device="cpu"):
    """Load a checkpoint into the model."""
    checkpoint = torch.load(path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint.get("step", 0)
    else:
        model.load_state_dict(checkpoint)
        step = 0
    return step


def get_eval_texts():
    """Fixed evaluation texts covering different complexity levels."""
    return [
        # Simple stories
        "Once upon a time there was a little bunny who loved to play in the garden.",
        "The cat sat on the mat and looked at the bird outside the window.",
        "A little girl named Lily went to the park with her mom to play.",
        # Medium complexity
        "The scientist observed that the chemical reaction produced unexpected results.",
        "In the ancient city, the merchants traded goods from distant lands across the sea.",
        # Math/reasoning
        "Question: If a train travels at 60 miles per hour for 3 hours, how far does it go? Answer:",
        "Question: What is 15 multiplied by 8? Answer: Let me calculate step by step.",
        "The total cost is calculated by multiplying the unit price by the quantity ordered.",
        # Complex language
        "Furthermore, the implications of this discovery extend beyond the immediate findings.",
        "The philosophical underpinnings of the argument rest on several key assumptions.",
    ]


def analyze_checkpoint(model, tokenizer, device="cpu"):
    """Analyze a single checkpoint for PPL and α statistics."""
    model.eval()
    model.to(device)

    eval_texts = get_eval_texts()
    all_alphas = []
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in eval_texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=512)["input_ids"].to(device)

            if tokens.shape[1] < 2:
                continue

            # Forward pass
            outputs = model(tokens, labels=tokens, return_routing_stats=True)
            loss = outputs["loss"]

            # Collect α values from routing stats
            for stats in outputs["routing_stats"]:
                alpha = stats["attention_ratio"]
                all_alphas.append(alpha)

            total_loss += loss.item() * (tokens.shape[1] - 1)
            total_tokens += tokens.shape[1] - 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = np.exp(min(avg_loss, 20))
    alpha_mean = np.mean(all_alphas)
    alpha_var = np.var(all_alphas)
    alpha_std = np.std(all_alphas)

    return {
        "ppl": float(ppl),
        "loss": float(avg_loss),
        "alpha_mean": float(alpha_mean),
        "alpha_var": float(alpha_var),
        "alpha_std": float(alpha_std),
        "n_samples": len(eval_texts),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Intelligence Evolution Analysis")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./esh_phase2_checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="evolution_data.json")
    args = parser.parse_args()

    device = args.device
    checkpoint_dir = Path(args.checkpoint_dir)

    # Find all checkpoints
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))

    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Analyze each checkpoint
    results = []
    for ckpt_path in checkpoints:
        print(f"\nAnalyzing: {ckpt_path.name}")

        model, config = create_model(device)
        step = load_checkpoint(model, str(ckpt_path), device)

        metrics = analyze_checkpoint(model, tokenizer, device)
        metrics["step"] = step
        metrics["checkpoint"] = ckpt_path.name
        results.append(metrics)

        print(f"  Step {step:>6d} | PPL {metrics['ppl']:6.2f} | "
              f"α_mean {metrics['alpha_mean']:.4f} | "
              f"α_var {metrics['alpha_var']:.6f} | "
              f"α_std {metrics['alpha_std']:.4f}")

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {args.output}")

    # Print summary table
    print(f"\n{'Step':>8} | {'PPL':>8} | {'α_mean':>8} | {'α_var':>10} | {'α_std':>8}")
    print("-" * 55)
    for r in results:
        print(f"{r['step']:>8d} | {r['ppl']:>8.2f} | "
              f"{r['alpha_mean']:>8.4f} | {r['alpha_var']:>10.6f} | "
              f"{r['alpha_std']:>8.4f}")


if __name__ == "__main__":
    main()
