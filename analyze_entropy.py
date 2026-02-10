"""
Self-Entropy Correlation Analysis
==================================
For each generated token, computes:
1. Shannon entropy of the model's output distribution (prediction uncertainty)
2. The router's α value (attention allocation)

Then computes Pearson correlation between them.
Hypothesis: High-entropy (uncertain) tokens → higher α (more attention needed)
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
    config = ESHConfig(
        d_model=768, n_layers=8, n_heads=12, n_experts=8,
        expert_dim=3072, max_seq_len=2048, use_checkpoint=False,
        dropout=0.0, layer_scale_init=1e-5,
    )
    return ESHModel(config), config


def compute_token_entropy(logits):
    """Compute Shannon entropy of the output distribution."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def analyze_entropy_correlation(model, tokenizer, texts, device="cpu"):
    """Analyze correlation between prediction entropy and routing α."""
    model.eval()
    model.to(device)

    all_entropies = []
    all_alphas = []
    per_text_results = []

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=512)["input_ids"].to(device)

            if tokens.shape[1] < 2:
                continue

            # Forward pass with routing stats
            outputs = model(tokens, return_routing_stats=True)
            logits = outputs["logits"]

            # Token-level entropy (for each position)
            entropies = compute_token_entropy(logits[0]).cpu().numpy()

            # Layer-averaged α for each position
            # routing_stats has one entry per layer
            alphas = []
            for stats in outputs["routing_stats"]:
                alphas.append(stats["attention_ratio"])
            avg_alpha = np.mean(alphas)

            # Store per-token data
            decoded_tokens = [tokenizer.decode([t]) for t in tokens[0]]

            text_result = {
                "text": text,
                "avg_entropy": float(np.mean(entropies)),
                "avg_alpha": float(avg_alpha),
                "tokens": [],
            }

            for i, (tok, ent) in enumerate(zip(decoded_tokens, entropies)):
                text_result["tokens"].append({
                    "token": tok,
                    "entropy": float(ent),
                    "alpha": float(avg_alpha),  # Layer-average
                })
                all_entropies.append(float(ent))
                all_alphas.append(float(avg_alpha))

            per_text_results.append(text_result)

    # Compute Pearson correlation
    if len(all_entropies) > 1:
        correlation = np.corrcoef(all_entropies, all_alphas)[0, 1]
    else:
        correlation = 0.0

    return {
        "correlation": float(correlation),
        "n_tokens": len(all_entropies),
        "avg_entropy": float(np.mean(all_entropies)),
        "avg_alpha": float(np.mean(all_alphas)),
        "entropies": all_entropies,
        "alphas": all_alphas,
        "per_text": per_text_results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Self-Entropy Correlation")
    parser.add_argument("--checkpoint", type=str,
                        default="./esh_phase2_checkpoints/final.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="entropy_data.json")
    args = parser.parse_args()

    device = args.device

    # Create and load model
    print("Loading model...")
    model, config = create_model(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint.get("step", "unknown")
        print(f"Loaded checkpoint from step {step}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Test texts - mix of simple and complex
    texts = [
        # Simple (low entropy expected)
        "Once upon a time there was a little bunny who loved to play.",
        "The cat sat on the mat and looked around.",
        "A little girl went to the park to play with her friends.",
        "The sun was shining and the birds were singing.",
        "He wanted a big red ball to play with.",
        # Medium
        "The scientist carefully examined the results of the experiment.",
        "In the ancient city, merchants traded goods from distant lands.",
        "The teacher explained the concept using a simple analogy.",
        # Complex (high entropy expected)
        "Question: If 3x + 7 = 22, what is the value of x? Answer:",
        "The mathematical proof requires establishing the base case first.",
        "Furthermore, the implications extend beyond the immediate findings.",
        "The algorithm's time complexity is O(n log n) in the average case.",
        "Calculate the derivative of f(x) = 3x^2 + 2x - 5 with respect to x.",
    ]

    print(f"\nAnalyzing {len(texts)} texts...")
    results = analyze_entropy_correlation(model, tokenizer, texts, device)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"SELF-ENTROPY CORRELATION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"\nPearson Correlation (Entropy ↔ α): {results['correlation']:.4f}")
    print(f"Total tokens analyzed: {results['n_tokens']}")
    print(f"Average entropy: {results['avg_entropy']:.4f}")
    print(f"Average α: {results['avg_alpha']:.4f}")

    print(f"\nPer-text breakdown:")
    print(f"{'Text (first 50 chars)':<52} | {'Entropy':>8} | {'α':>8}")
    print("-" * 75)
    for r in results["per_text"]:
        text_preview = r["text"][:50]
        print(f"{text_preview:<52} | {r['avg_entropy']:>8.4f} | {r['avg_alpha']:>8.4f}")

    # Save full results
    # Only save summary (not full token lists) to keep file small
    save_data = {
        "correlation": results["correlation"],
        "n_tokens": results["n_tokens"],
        "avg_entropy": results["avg_entropy"],
        "avg_alpha": results["avg_alpha"],
        "per_text": [{
            "text": r["text"],
            "avg_entropy": r["avg_entropy"],
            "avg_alpha": r["avg_alpha"],
        } for r in results["per_text"]],
    }
    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
