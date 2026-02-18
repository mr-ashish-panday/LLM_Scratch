"""
Phase 2: Entropy ↔ Ponder Depth Correlation Analysis
=====================================================
Proves the model uses MORE ponder steps for HARDER tokens.

For each token, records:
  - Prediction entropy (Shannon entropy of output distribution)
  - Ponder steps used (sum of remainders, per-token)
  - Alpha value (SSM↔Attention mix)
  - Domain source (TinyStories / WikiText / GSM8K)

Then computes:
  - Pearson correlation (entropy ↔ ponder)
  - Per-domain breakdown
  - Entropy bins analysis
  - Publication-quality scatter plots

Usage:
  python analyze_ponder_entropy.py --checkpoint results/depth_only/final_model.pt
  python analyze_ponder_entropy.py --checkpoint results/unified/final_model.pt
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from data_loader import create_mixed_eval_dataloader
from esh_unified.model import UnifiedModel, UnifiedConfig


def compute_per_token_entropy(logits):
    """Shannon entropy of the output distribution, per token.
    
    Args:
        logits: [B, L, V] raw logits
    Returns:
        entropy: [B, L] nats per token
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, L]
    return entropy


def load_model(checkpoint_path, device):
    """Load a trained checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = UnifiedConfig(**ckpt["config"])
    model = UnifiedModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Mode: {config.mode}")
    print(f"  Width routing: {config.enable_width_routing}")
    print(f"  Depth routing: {config.enable_depth_routing}")
    return model, config


def collect_token_data(model, tokenizer, device, n_tokens_target=100_000,
                       max_length=512, cache_dir="./data_cache"):
    """Run inference and collect per-token data across domains.
    
    Returns a list of dicts, one per token:
      {"entropy": float, "ponder": float, "alpha": float, "domain": str, "token": str}
    """
    loader = create_mixed_eval_dataloader(
        tokenizer, batch_size=8, max_length=max_length,
        samples_per_source=500, cache_dir=cache_dir,
    )
    
    all_tokens = []
    n_collected = 0
    
    print(f"\nCollecting {n_tokens_target:,} tokens...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)        # [B, L]
            attention_mask = batch["attention_mask"].to(device)  # [B, L]
            sources = batch["sources"]                        # list of str
            
            # Forward pass with routing stats
            outputs = model(input_ids, return_routing_stats=True)
            logits = outputs["logits"]  # [B, L, V]
            
            # Per-token entropy
            entropy = compute_per_token_entropy(logits).cpu()  # [B, L]
            
            # Collect per-layer per-token ponder and alpha, then average across layers
            layer_stats = outputs["routing_stats"]
            
            if layer_stats and "per_token_ponder" in layer_stats[0]:
                # Stack per-layer ponder: [n_layers, B, L] → mean → [B, L]
                all_ponder = torch.stack([s["per_token_ponder"] for s in layer_stats])
                avg_ponder = all_ponder.mean(dim=0)  # [B, L]
                
                all_alpha = torch.stack([s["per_token_alpha"] for s in layer_stats])
                avg_alpha = all_alpha.mean(dim=0)  # [B, L]
            else:
                # Fallback: use scalar stats
                avg_ponder_val = np.mean([s.get("avg_ponder_steps", 1.0) for s in layer_stats])
                avg_ponder = torch.full_like(entropy, avg_ponder_val)
                avg_alpha_val = np.mean([s.get("alpha_mean", 0.5) for s in layer_stats])
                avg_alpha = torch.full_like(entropy, avg_alpha_val)
            
            mask = attention_mask.cpu()  # [B, L]
            B, L = input_ids.shape
            
            for b in range(B):
                domain = sources[b]
                for t in range(L):
                    if mask[b, t] == 0:
                        continue  # skip padding
                    
                    token_str = tokenizer.decode([input_ids[b, t].item()])
                    
                    all_tokens.append({
                        "entropy": entropy[b, t].item(),
                        "ponder": avg_ponder[b, t].item(),
                        "alpha": avg_alpha[b, t].item(),
                        "domain": domain,
                        "token": token_str,
                    })
                    n_collected += 1
            
            if n_collected >= n_tokens_target:
                break
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}: {n_collected:,} tokens collected")
    
    print(f"Total tokens collected: {n_collected:,}")
    return all_tokens


def analyze_correlation(token_data):
    """Compute full statistical analysis."""
    entropies = np.array([t["entropy"] for t in token_data])
    ponders = np.array([t["ponder"] for t in token_data])
    alphas = np.array([t["alpha"] for t in token_data])
    domains = [t["domain"] for t in token_data]
    
    results = {}
    
    # === Global correlation ===
    r_ponder, p_ponder = scipy_stats.pearsonr(entropies, ponders)
    r_alpha, p_alpha = scipy_stats.pearsonr(entropies, alphas)
    
    results["global"] = {
        "n_tokens": len(token_data),
        "entropy_ponder_r": float(r_ponder),
        "entropy_ponder_p": float(p_ponder),
        "entropy_alpha_r": float(r_alpha),
        "entropy_alpha_p": float(p_alpha),
        "mean_entropy": float(np.mean(entropies)),
        "mean_ponder": float(np.mean(ponders)),
        "mean_alpha": float(np.mean(alphas)),
    }
    
    # === Per-domain breakdown ===
    results["per_domain"] = {}
    for domain in ["tiny", "wiki", "gsm"]:
        mask = np.array([d == domain for d in domains])
        if mask.sum() < 10:
            continue
        
        dom_ent = entropies[mask]
        dom_pond = ponders[mask]
        dom_alpha = alphas[mask]
        
        r_p, p_p = scipy_stats.pearsonr(dom_ent, dom_pond)
        r_a, p_a = scipy_stats.pearsonr(dom_ent, dom_alpha)
        
        results["per_domain"][domain] = {
            "n_tokens": int(mask.sum()),
            "mean_entropy": float(np.mean(dom_ent)),
            "mean_ponder": float(np.mean(dom_pond)),
            "mean_alpha": float(np.mean(dom_alpha)),
            "entropy_ponder_r": float(r_p),
            "entropy_ponder_p": float(p_p),
            "entropy_alpha_r": float(r_a),
            "entropy_alpha_p": float(p_a),
        }
    
    # === Entropy bin analysis ===
    # Split tokens into low/medium/high entropy bins
    percentiles = np.percentile(entropies, [33, 67])
    bins = {
        "low_entropy": entropies <= percentiles[0],
        "mid_entropy": (entropies > percentiles[0]) & (entropies <= percentiles[1]),
        "high_entropy": entropies > percentiles[1],
    }
    
    results["entropy_bins"] = {}
    for bin_name, bin_mask in bins.items():
        results["entropy_bins"][bin_name] = {
            "n_tokens": int(bin_mask.sum()),
            "entropy_range": [float(entropies[bin_mask].min()), float(entropies[bin_mask].max())],
            "mean_ponder": float(np.mean(ponders[bin_mask])),
            "std_ponder": float(np.std(ponders[bin_mask])),
            "mean_alpha": float(np.mean(alphas[bin_mask])),
        }
    
    # === Token-type analysis (common vs rare) ===
    # Check if simple tokens get fewer ponder steps
    simple_tokens = {"the", "a", "an", "is", "was", "to", "and", "of", "in", "it",
                     "that", "for", "on", "he", "she", "they", "I", "you", "we"}
    
    simple_mask = np.array([t["token"].strip().lower() in simple_tokens for t in token_data])
    complex_mask = ~simple_mask
    
    if simple_mask.sum() > 0 and complex_mask.sum() > 0:
        results["token_type"] = {
            "simple": {
                "n_tokens": int(simple_mask.sum()),
                "mean_entropy": float(np.mean(entropies[simple_mask])),
                "mean_ponder": float(np.mean(ponders[simple_mask])),
                "mean_alpha": float(np.mean(alphas[simple_mask])),
            },
            "complex": {
                "n_tokens": int(complex_mask.sum()),
                "mean_entropy": float(np.mean(entropies[complex_mask])),
                "mean_ponder": float(np.mean(ponders[complex_mask])),
                "mean_alpha": float(np.mean(alphas[complex_mask])),
            }
        }
    
    return results


def generate_plots(token_data, results, output_dir):
    """Generate publication-quality scatter plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    entropies = np.array([t["entropy"] for t in token_data])
    ponders = np.array([t["ponder"] for t in token_data])
    alphas = np.array([t["alpha"] for t in token_data])
    domains = [t["domain"] for t in token_data]
    
    domain_colors = {"tiny": "#4CAF50", "wiki": "#2196F3", "gsm": "#FF5722"}
    domain_labels = {"tiny": "TinyStories", "wiki": "WikiText-103", "gsm": "GSM8K"}
    
    # ── Plot 1: Entropy vs Ponder Steps (color-coded by domain) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Scatter
    ax = axes[0]
    for domain in ["tiny", "wiki", "gsm"]:
        mask = np.array([d == domain for d in domains])
        if mask.sum() == 0:
            continue
        ax.scatter(entropies[mask], ponders[mask],
                   alpha=0.05, s=2, c=domain_colors[domain],
                   label=domain_labels[domain])
    
    # Add binned means
    n_bins = 20
    bin_edges = np.linspace(entropies.min(), entropies.max(), n_bins + 1)
    bin_centers = []
    bin_means = []
    for i in range(n_bins):
        mask = (entropies >= bin_edges[i]) & (entropies < bin_edges[i + 1])
        if mask.sum() > 5:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(np.mean(ponders[mask]))
    
    ax.plot(bin_centers, bin_means, 'k-', linewidth=2.5, label="Binned Mean", zorder=5)
    ax.plot(bin_centers, bin_means, 'wo', markersize=4, zorder=6)
    
    r = results["global"]["entropy_ponder_r"]
    p = results["global"]["entropy_ponder_p"]
    ax.set_xlabel("Prediction Entropy (nats)", fontsize=12)
    ax.set_ylabel("Ponder Steps", fontsize=12)
    ax.set_title(f"Entropy → Ponder Depth\nr={r:.4f}, p={p:.2e}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ── Plot 2: Per-domain ponder distribution ──
    ax = axes[1]
    domain_ponders = {}
    for domain in ["tiny", "wiki", "gsm"]:
        mask = np.array([d == domain for d in domains])
        if mask.sum() > 0:
            domain_ponders[domain_labels[domain]] = ponders[mask]
    
    parts = ax.violinplot(domain_ponders.values(), showmeans=True, showextrema=True)
    ax.set_xticks(range(1, len(domain_ponders) + 1))
    ax.set_xticklabels(domain_ponders.keys())
    ax.set_ylabel("Ponder Steps", fontsize=12)
    ax.set_title("Ponder Depth by Domain", fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Color violins
    for i, (domain, pc) in enumerate(zip(["tiny", "wiki", "gsm"], parts["bodies"])):
        pc.set_facecolor(domain_colors[domain])
        pc.set_alpha(0.6)
    
    # ── Plot 3: Entropy bins bar chart ──
    ax = axes[2]
    bins_data = results["entropy_bins"]
    bin_names = ["low_entropy", "mid_entropy", "high_entropy"]
    bin_labels = ["Low\nEntropy", "Medium\nEntropy", "High\nEntropy"]
    bar_ponders = [bins_data[b]["mean_ponder"] for b in bin_names]
    bar_stds = [bins_data[b]["std_ponder"] for b in bin_names]
    bar_colors = ["#4CAF50", "#FFC107", "#FF5722"]
    
    bars = ax.bar(bin_labels, bar_ponders, yerr=bar_stds, capsize=5,
                  color=bar_colors, edgecolor="black", linewidth=0.5)
    
    for bar, val in zip(bars, bar_ponders):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    
    ax.set_ylabel("Mean Ponder Steps", fontsize=12)
    ax.set_title("Ponder Steps by Entropy Level", fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "entropy_ponder_correlation.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {plot_path}")
    
    # ── Plot 4: Entropy vs Alpha (if width routing active) ──
    if abs(np.std(alphas)) > 0.001:
        fig, ax = plt.subplots(figsize=(8, 5))
        for domain in ["tiny", "wiki", "gsm"]:
            mask = np.array([d == domain for d in domains])
            if mask.sum() == 0:
                continue
            ax.scatter(entropies[mask], alphas[mask],
                       alpha=0.05, s=2, c=domain_colors[domain],
                       label=domain_labels[domain])
        
        r_a = results["global"]["entropy_alpha_r"]
        p_a = results["global"]["entropy_alpha_p"]
        ax.set_xlabel("Prediction Entropy (nats)", fontsize=12)
        ax.set_ylabel("Alpha (→1 = Attention, →0 = SSM)", fontsize=12)
        ax.set_title(f"Entropy → Width Routing\nr={r_a:.4f}, p={p_a:.2e}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        alpha_path = os.path.join(output_dir, "entropy_alpha_correlation.png")
        plt.savefig(alpha_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Alpha plot saved: {alpha_path}")


def print_results(results):
    """Pretty-print the analysis."""
    g = results["global"]
    
    print("\n" + "=" * 70)
    print("  PHASE 2: ENTROPY ↔ PONDER DEPTH CORRELATION")
    print("=" * 70)
    
    print(f"\n  Total tokens analyzed: {g['n_tokens']:,}")
    print(f"  Mean entropy: {g['mean_entropy']:.4f} nats")
    print(f"  Mean ponder:  {g['mean_ponder']:.4f} steps")
    print(f"  Mean alpha:   {g['mean_alpha']:.4f}")
    
    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │ Entropy ↔ Ponder:  r = {g['entropy_ponder_r']:+.4f}  (p = {g['entropy_ponder_p']:.2e}) │")
    print(f"  │ Entropy ↔ Alpha:   r = {g['entropy_alpha_r']:+.4f}  (p = {g['entropy_alpha_p']:.2e}) │")
    print(f"  └──────────────────────────────────────────────────┘")
    
    if g['entropy_ponder_p'] < 0.001 and g['entropy_ponder_r'] > 0.1:
        print("  ★ STATISTICALLY SIGNIFICANT: Model uses more steps for harder tokens!")
    elif g['entropy_ponder_p'] < 0.05:
        print("  ✓ Weak but significant correlation detected.")
    else:
        print("  ✗ No significant correlation found.")
    
    # Per-domain
    print(f"\n  {'Domain':<15} {'Tokens':>8} {'Entropy':>9} {'Ponder':>8} {'Alpha':>7} {'r(E→P)':>8}")
    print("  " + "-" * 60)
    for domain, d in results["per_domain"].items():
        label = {"tiny": "TinyStories", "wiki": "WikiText", "gsm": "GSM8K"}.get(domain, domain)
        sig = "***" if d["entropy_ponder_p"] < 0.001 else "**" if d["entropy_ponder_p"] < 0.01 else "*" if d["entropy_ponder_p"] < 0.05 else ""
        print(f"  {label:<15} {d['n_tokens']:>8,} {d['mean_entropy']:>9.4f} {d['mean_ponder']:>8.4f} {d['mean_alpha']:>7.4f} {d['entropy_ponder_r']:>+7.4f}{sig}")
    
    # Entropy bins
    print(f"\n  Entropy Bins:")
    print(f"  {'Bin':<15} {'Tokens':>8} {'Entropy Range':>18} {'Mean Ponder':>12} {'Mean Alpha':>11}")
    print("  " + "-" * 68)
    for bin_name, b in results["entropy_bins"].items():
        label = bin_name.replace("_", " ").title()
        ent_range = f"[{b['entropy_range'][0]:.2f}, {b['entropy_range'][1]:.2f}]"
        print(f"  {label:<15} {b['n_tokens']:>8,} {ent_range:>18} {b['mean_ponder']:>12.4f} {b['mean_alpha']:>11.4f}")
    
    # Token type analysis
    if "token_type" in results:
        s = results["token_type"]["simple"]
        c = results["token_type"]["complex"]
        print(f"\n  Token Type Analysis:")
        print(f"    Simple tokens (the, a, is...): ponder={s['mean_ponder']:.4f}  entropy={s['mean_entropy']:.4f}  (n={s['n_tokens']:,})")
        print(f"    Complex tokens (all other):    ponder={c['mean_ponder']:.4f}  entropy={c['mean_entropy']:.4f}  (n={c['n_tokens']:,})")
        delta = c['mean_ponder'] - s['mean_ponder']
        print(f"    Delta: {delta:+.4f} ponder steps ({'MORE thinking for complex ✓' if delta > 0 else 'unexpected ✗'})")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Entropy ↔ Ponder Correlation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (e.g. results/depth_only/final_model.pt)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-tokens", type=int, default=100_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dir", type=str, default="./data_cache")
    parser.add_argument("--output-dir", type=str, default="./analysis_results")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Collect per-token data
    token_data = collect_token_data(
        model, tokenizer, device,
        n_tokens_target=args.n_tokens,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
    )
    
    # Analyze
    results = analyze_correlation(token_data)
    
    # Print
    print_results(results)
    
    # Plots
    generate_plots(token_data, results, args.output_dir)
    
    # Save JSON
    json_path = os.path.join(args.output_dir, f"entropy_correlation_{config.mode}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")


if __name__ == "__main__":
    main()
