"""
Quick Domain α Probe
====================
Fast 2-minute diagnostic: feeds pure TinyStories and pure GSM8K batches
into a width_only checkpoint and compares the α means.

If α(GSM8K) > α(TinyStories), the router IS doing token-level specialization.
If they're identical, the collapse is truly degenerate.

Usage:
  python probe_alpha_per_domain.py --checkpoint results/width_only/final_model.pt
"""

import os, sys, torch, json
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from datasets import load_dataset
from esh_unified.model import UnifiedModel, UnifiedConfig


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = UnifiedConfig(**ckpt["config"])
    model = UnifiedModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def tokenize_texts(texts, tokenizer, max_length=512):
    tokens = tokenizer(
        texts, max_length=max_length, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    return tokens["input_ids"]


def probe_domain(model, input_ids, device, domain_name):
    """Run one batch and extract per-layer alpha stats."""
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        # Set step high (past burn-in) so alpha is not forced
        model.set_global_step(99999)
        outputs = model(input_ids, return_routing_stats=True)
    
    stats = outputs["routing_stats"]
    alphas = [s.get("alpha_mean", 0.5) for s in stats]
    ponders = [s.get("avg_ponder_steps", 1.0) for s in stats]
    
    # Per-token alpha if available
    per_token_alphas = []
    if stats and "per_token_alpha" in stats[0]:
        for s in stats:
            per_token_alphas.append(s["per_token_alpha"].cpu())
        stacked = torch.stack(per_token_alphas).mean(dim=0)  # [B, L]
        alpha_std = stacked.std().item()
    else:
        alpha_std = 0.0
    
    result = {
        "domain": domain_name,
        "alpha_mean": sum(alphas) / len(alphas),
        "alpha_std": alpha_std,
        "avg_ponder": sum(ponders) / len(ponders),
        "per_layer_alpha": alphas,
    }
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-dir", type=str, default="/tmp/hf_cache")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model, config = load_model(args.checkpoint, device)
    print(f"Mode: {config.mode} | Width: {config.enable_width_routing} | Depth: {config.enable_depth_routing}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load pure domain batches
    print("\nLoading TinyStories...")
    tiny = load_dataset("roneneldan/TinyStories", split="validation",
                        streaming=True)
    tiny_texts = []
    for i, ex in enumerate(tiny):
        text = ex.get("text", "")
        if text and len(text.strip()) >= 50:
            tiny_texts.append(text)
        if len(tiny_texts) >= args.batch_size:
            break
    
    print("Loading GSM8K...")
    gsm = load_dataset("openai/gsm8k", "main", split="test",
                        streaming=True)
    gsm_texts = []
    for i, ex in enumerate(gsm):
        text = f"Question: {ex.get('question', '')}\nAnswer: {ex.get('answer', '')}"
        gsm_texts.append(text)
        if len(gsm_texts) >= args.batch_size:
            break
    
    print("Loading WikiText-103...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation",
                        streaming=True)
    wiki_texts = []
    for i, ex in enumerate(wiki):
        text = ex.get("text", "")
        if text and len(text.strip()) >= 50:
            wiki_texts.append(text)
        if len(wiki_texts) >= args.batch_size:
            break
    
    # Tokenize
    tiny_ids = tokenize_texts(tiny_texts, tokenizer)
    gsm_ids = tokenize_texts(gsm_texts, tokenizer)
    wiki_ids = tokenize_texts(wiki_texts, tokenizer)
    
    # Probe each domain
    print("\n" + "=" * 60)
    print("  DOMAIN-SPECIFIC ALPHA PROBE")
    print("=" * 60)
    
    results = {}
    for name, ids in [("TinyStories", tiny_ids), ("WikiText-103", wiki_ids), ("GSM8K", gsm_ids)]:
        r = probe_domain(model, ids, device, name)
        results[name] = r
        print(f"\n  {name}:")
        print(f"    α mean:     {r['alpha_mean']:.4f}")
        print(f"    α std:      {r['alpha_std']:.4f}")
        print(f"    Ponder:     {r['avg_ponder']:.4f}")
        print(f"    Per-layer:  {[f'{a:.3f}' for a in r['per_layer_alpha']]}")
    
    # Verdict
    tiny_a = results["TinyStories"]["alpha_mean"]
    gsm_a = results["GSM8K"]["alpha_mean"]
    wiki_a = results["WikiText-103"]["alpha_mean"]
    delta = gsm_a - tiny_a
    
    print(f"\n{'=' * 60}")
    print(f"  VERDICT")
    print(f"{'=' * 60}")
    print(f"  α(GSM8K)      = {gsm_a:.4f}  (math reasoning)")
    print(f"  α(WikiText)   = {wiki_a:.4f}  (complex text)")
    print(f"  α(TinyStories)= {tiny_a:.4f}  (simple stories)")
    print(f"  Δ(GSM-Tiny)   = {delta:+.4f}")
    
    if delta > 0.05:
        print(f"\n  ★ SPECIALIZATION CONFIRMED: Router sends MORE attention to complex tokens!")
        print(f"    Even within the attractor state (α≈0.26), the router differentiates.")
    elif delta > 0.01:
        print(f"\n  ✓ WEAK SPECIALIZATION: Small but measurable difference.")
    else:
        print(f"\n  ✗ NO SPECIALIZATION: α is flat across domains — true degenerate collapse.")
    
    # Save
    os.makedirs("analysis_results", exist_ok=True)
    with open("analysis_results/domain_alpha_probe.json", "w") as f:
        # Remove non-serializable items
        serializable = {}
        for k, v in results.items():
            serializable[k] = {kk: vv for kk, vv in v.items()}
        json.dump(serializable, f, indent=2)
    print(f"\nSaved: analysis_results/domain_alpha_probe.json")


if __name__ == "__main__":
    main()
