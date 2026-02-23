"""
Quick Domain Probe: Train + Probe in One Shot
=============================================
Trains a fast model for 500 steps (no burn-in override — router learns freely),
then immediately probes per-domain alpha to check if the router differentiates
between simple (TinyStories) and complex (GSM8K) tokens.

This answers the critical question: Is the router a dead coin flip, or does it
actually learn per-token specialization?

Usage:
    python quick_domain_probe.py --cache-dir /tmp/hf_cache
"""

import os, sys, time, json, argparse
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer
from datasets import load_dataset
from esh_unified.model import UnifiedModel, UnifiedConfig
from data_loader import create_mixed_dataloader


def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))


def train_phase(model, train_loader, device, steps=500, lr=3e-4):
    """Train for N steps with router losses ACTIVE (no burn-in override)."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.1
    )
    scaler = GradScaler()
    data_iter = iter(train_loader)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING: {steps} steps (router fully active)")
    print(f"{'='*60}")
    
    t0 = time.time()
    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(device)
        
        lr_now = cosine_lr(step, 100, steps, lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        
        # Key: set global_step past burn-in so router trains freely
        with autocast(enabled=True):
            model.set_global_step(99999)  # Past burn-in, router is FREE
            outputs = model(input_ids, labels=input_ids, return_routing_stats=(step % 100 == 99))
            loss = outputs["loss"]
            aux = outputs["aux_loss"]
            total_loss = loss + aux
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if (step + 1) % 10 == 0:
            print(f"  step {step+1:>4} | loss={loss.item():.4f} | aux={aux.item():.6f}", flush=True)
        
        if (step + 1) % 100 == 0:
            stats = outputs.get("routing_stats", [{}])
            if stats:
                alpha = stats[0].get("alpha_mean", 0.5)
                print(f"  >>> α_mean = {alpha:.4f}", flush=True)
    
    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.0f}s ({elapsed/steps:.1f}s/step)")
    return model


def probe_domain(model, input_ids, device, domain_name):
    """Probe alpha on a single-domain batch."""
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        model.set_global_step(99999)
        outputs = model(input_ids, return_routing_stats=True)
    
    stats = outputs["routing_stats"]
    alphas = [s.get("alpha_mean", 0.5) for s in stats]
    alpha_stds = [s.get("alpha_std", 0.0) for s in stats]
    
    return {
        "domain": domain_name,
        "alpha_mean": sum(alphas) / len(alphas),
        "alpha_std": sum(alpha_stds) / len(alpha_stds),
        "per_layer_alpha": [f"{a:.4f}" for a in alphas],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--cache-dir", type=str, default="/tmp/hf_cache")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Fast model config (same as --fast capture)
    config = UnifiedConfig(
        vocab_size=50257,
        d_model=512,
        n_layers=4,
        n_heads=8,
        n_experts=4,
        max_seq_len=256,
        use_checkpoint=False,
        enable_width_routing=True,
        enable_depth_routing=False,
        max_ponder_steps=3,
        ponder_cost_weight=0.5,
        use_moe=True,
    )
    
    model = UnifiedModel(config).to(device)
    model.train()
    print(f"Params: {model.count_parameters()/1e6:.1f}M")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Train
    loader = create_mixed_dataloader(
        tokenizer, batch_size=args.batch_size, max_length=256,
        cache_dir=args.cache_dir,
    )
    model = train_phase(model, loader, device, steps=args.steps)
    
    # Save checkpoint
    os.makedirs("results/domain_probe", exist_ok=True)
    ckpt_path = "results/domain_probe/trained_model.pt"
    torch.save({
        "model": model.state_dict(),
        "config": vars(config),
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")
    
    # ===================== DOMAIN PROBE =====================
    model.eval()
    
    print(f"\nLoading domain-specific data...", flush=True)
    
    # TinyStories (simple)
    tiny = load_dataset("roneneldan/TinyStories", split="validation",
                        streaming=True, trust_remote_code=True)
    tiny_texts = []
    for ex in tiny:
        text = ex.get("text", "")
        if text and len(text.strip()) >= 50:
            tiny_texts.append(text)
        if len(tiny_texts) >= 8:
            break
    
    # GSM8K (complex math)
    gsm = load_dataset("openai/gsm8k", "main", split="test",
                        streaming=True, trust_remote_code=True)
    gsm_texts = []
    for ex in gsm:
        text = f"Question: {ex.get('question', '')}\nAnswer: {ex.get('answer', '')}"
        gsm_texts.append(text)
        if len(gsm_texts) >= 8:
            break
    
    # WikiText (academic prose)
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation",
                        streaming=True, trust_remote_code=True)
    wiki_texts = []
    for ex in wiki:
        text = ex.get("text", "")
        if text and len(text.strip()) >= 50:
            wiki_texts.append(text)
        if len(wiki_texts) >= 8:
            break
    
    # Tokenize
    def tok(texts):
        return tokenizer(texts, max_length=256, truncation=True,
                        padding="max_length", return_tensors="pt")["input_ids"]
    
    tiny_ids = tok(tiny_texts)
    gsm_ids = tok(gsm_texts)
    wiki_ids = tok(wiki_texts)
    
    # Probe
    print(f"\n{'='*60}")
    print(f"  DOMAIN ALPHA PROBE (after {args.steps} training steps)")
    print(f"{'='*60}")
    
    results = {}
    for name, ids in [("TinyStories", tiny_ids), ("WikiText-103", wiki_ids), ("GSM8K", gsm_ids)]:
        r = probe_domain(model, ids, device, name)
        results[name] = r
        print(f"\n  {name}:")
        print(f"    α mean:     {r['alpha_mean']:.4f}")
        print(f"    α std:      {r['alpha_std']:.4f}")
        print(f"    Per-layer:  {r['per_layer_alpha']}")
    
    # Verdict
    tiny_a = results["TinyStories"]["alpha_mean"]
    gsm_a = results["GSM8K"]["alpha_mean"]
    wiki_a = results["WikiText-103"]["alpha_mean"]
    delta = gsm_a - tiny_a
    
    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")
    print(f"  α(GSM8K)       = {gsm_a:.4f}  (math reasoning)")
    print(f"  α(WikiText)    = {wiki_a:.4f}  (complex text)")
    print(f"  α(TinyStories) = {tiny_a:.4f}  (simple stories)")
    print(f"  Δ(GSM-Tiny)    = {delta:+.4f}")
    
    if abs(delta) > 0.05:
        print(f"\n  ★ SPECIALIZATION CONFIRMED: Router DIFFERENTIATES by domain!")
        if delta > 0:
            print(f"    Complex tokens → more Attention (α↑)")
            print(f"    Simple tokens → more SSM (α↓)")
        else:
            print(f"    Complex tokens → more SSM (α↓)")
            print(f"    Simple tokens → more Attention (α↑)")
    elif abs(delta) > 0.01:
        print(f"\n  ✓ WEAK SPECIALIZATION: Small but measurable difference.")
    else:
        print(f"\n  ✗ NO SPECIALIZATION: Router is a dead coin flip. α is flat across domains.")
    
    # Save
    with open("results/domain_probe/probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/domain_probe/probe_results.json")


if __name__ == "__main__":
    main()
