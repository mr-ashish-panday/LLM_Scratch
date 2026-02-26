"""
ESH Width × Depth Ablation Training Script
============================================
Runs 4 ablation modes to populate the core paper table:

  python run_ablation.py --mode baseline
  python run_ablation.py --mode width_only
  python run_ablation.py --mode depth_only
  python run_ablation.py --mode unified

All modes use identical architecture, data, and hyperparameters.
The ONLY difference is whether width routing (α) and depth routing
(pondering) are enabled or disabled.

Results are saved to results/<mode>/ for comparison.
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from esh_unified.model import UnifiedConfig, UnifiedModel
from data_loader import create_mixed_dataloader, create_mixed_eval_dataloader


def set_deterministic(seed: int = 42):
    """Lock all RNG seeds for reproducibility across ablation runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Deterministic mode: seed={seed}")


def estimate_flops_per_token(config: UnifiedConfig, avg_ponder_steps: float) -> float:
    """
    Estimate FLOPs per token for the current config and ponder depth.

    Per-layer FLOPs (forward pass, approximate):
      Attention:  8 * d^2 (QKV proj + output proj + attention compute)
      SSM:        4 * d * d_expand (mamba linear ops)
      FFN:       12 * d^2 (SwiGLU: 3 projections of 4d)
      Router:     d * d/4 + d/4 (complexity_net)

    Total per layer ≈ 24 * d^2 (dominated by attn + FFN)
    Total model ≈ n_layers * 24 * d^2 * avg_ponder_steps
    """
    d = config.d_model
    n = config.n_layers
    # Per-layer estimate (forward only)
    attn_flops = 8 * d * d
    ssm_flops = 4 * d * (d * config.ssm_expand if hasattr(config, 'ssm_expand') else d * 2)
    ffn_flops = 12 * d * d  # SwiGLU 3-way
    router_flops = d * (d // 4) * 2
    layer_flops = attn_flops + ssm_flops + ffn_flops + router_flops
    # Scale by depth (pondering) and layers
    total_flops = n * layer_flops * avg_ponder_steps
    return total_flops


def parse_args():
    parser = argparse.ArgumentParser(description="ESH v2 Hard Routing Ablation")
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["baseline", "width_only", "compare"],
        help="Ablation mode (depth routing disabled — ACT+SSM collision)"
    )
    parser.add_argument("--max_steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (same for all modes)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")

    # Model size
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_experts", type=int, default=4)
    parser.add_argument("--no-moe", action="store_true",
                        help="Replace MoE with plain SwiGLU")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to cache datasets locally")

    # Hard routing params
    parser.add_argument("--lambda-cost", type=float, default=0.005,
                        help="Compute penalty weight (tax on Attention)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Gumbel-Softmax temperature")
    parser.add_argument("--penalty-warmup", type=int, default=2000,
                        help="Steps before enabling compute penalty")

    return parser.parse_args()


def get_config(args) -> UnifiedConfig:
    """Create config based on ablation mode."""
    width = args.mode == "width_only"

    return UnifiedConfig(
        vocab_size=50257,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_experts=args.n_experts,
        max_seq_len=args.seq_len,
        use_checkpoint=True,
        enable_width_routing=width,
        enable_depth_routing=False,  # Disabled (ACT+SSM collision)
        max_ponder_steps=1,
        compute_penalty_weight=args.lambda_cost,
        router_temperature=args.temperature,
        use_moe=not getattr(args, 'no_moe', False),
    )


def cosine_lr(step, warmup, max_steps, max_lr, min_lr=1e-5):
    """Cosine learning rate with warmup."""
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(args):
    # Determinism — identical seeds across all 4 ablation runs
    set_deterministic(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config
    config = get_config(args)
    print(f"\n{'='*60}")
    print(f"  ABLATION MODE: {config.mode.upper()}")
    print(f"  Width routing: {config.enable_width_routing}")
    print(f"  Depth routing: {config.enable_depth_routing}")
    print(f"{'='*60}\n")

    # Model
    model = UnifiedModel(config).to(device)
    model.print_model_info()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Data loaders (mixed complexity: TinyStories + WikiText + GSM8K)
    cache_dir = getattr(args, 'cache_dir', None)
    train_loader = create_mixed_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.seq_len,
        cache_dir=cache_dir,
    )
    eval_loader = create_mixed_eval_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.seq_len,
        cache_dir=cache_dir,
    )

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(
            model.parameters(), lr=args.lr, weight_decay=0.1
        )
        print("Using 8-bit PagedAdamW")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.1
        )
        print("Using standard AdamW")

    # AMP
    scaler = GradScaler()
    use_amp = device.type == "cuda"

    # Results directory
    results_dir = os.path.join("results", config.mode)
    os.makedirs(results_dir, exist_ok=True)

    # Resume from checkpoint
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    data_iter = iter(train_loader)
    metrics_log = []
    accum_metrics = {}
    global_step = start_step
    start_time = time.time()
    tokens_processed = 0
    best_ppl = float("inf")

    print(f"\nStarting training: {args.max_steps} steps")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Total parameters: {model.count_parameters() / 1e6:.2f}M\n")

    while global_step < args.max_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        tokens_processed += input_ids.numel()

        # LR schedule
        lr = cosine_lr(global_step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward
        is_log_step = (global_step + 1) % args.log_interval == 0
        with autocast(enabled=use_amp):
            outputs = model(input_ids, labels=input_ids, return_routing_stats=is_log_step)
            lm_loss = outputs["loss"]
            aux_loss = outputs["aux_loss"]
            compute_penalty = outputs["compute_penalty"]

            # Compute penalty warmup:
            # Steps 0-N: λ=0 (let model discover routing)
            # Steps N+: λ=configured (add economic incentive)
            penalty_warmup = getattr(args, 'penalty_warmup', 2000)
            if global_step < penalty_warmup:
                lambda_eff = 0.0
            else:
                lambda_eff = config.compute_penalty_weight

            total_loss = (lm_loss + aux_loss + lambda_eff * compute_penalty) / args.grad_accum

        # Backward
        scaler.scale(total_loss).backward()

        # Optimizer step
        if (global_step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Collect metrics
        step_metrics = {
            "loss": lm_loss.item(),
            "ppl": math.exp(min(lm_loss.item(), 20)),
            "aux_loss": aux_loss.item() if torch.is_tensor(aux_loss) else float(aux_loss),
            "compute_penalty": compute_penalty.item() if torch.is_tensor(compute_penalty) else float(compute_penalty),
            "lambda_eff": lambda_eff,
        }

        # Add routing stats
        if outputs["routing_stats"]:
            attn_ratios = [s.get("attn_ratio", 0.5) for s in outputs["routing_stats"]]
            ssm_ratios = [s.get("ssm_ratio", 0.5) for s in outputs["routing_stats"]]
            step_metrics["attn_ratio"] = sum(attn_ratios) / len(attn_ratios)
            step_metrics["ssm_ratio"] = sum(ssm_ratios) / len(ssm_ratios)

        for k, v in step_metrics.items():
            accum_metrics[k] = accum_metrics.get(k, 0) + v

        global_step += 1

        # Logging
        if global_step % args.log_interval == 0:
            elapsed = time.time() - start_time
            tok_sec = tokens_processed / elapsed

            avg = {k: v / args.log_interval for k, v in accum_metrics.items()}
            accum_metrics = {}

            ppl = avg.get("ppl", 0)
            if ppl < best_ppl:
                best_ppl = ppl

            log_entry = {
                "step": global_step,
                "lr": lr,
                "tok_sec": tok_sec,
                **avg,
            }
            metrics_log.append(log_entry)

            attn_pct = avg.get('attn_ratio', 0.5) * 100
            ssm_pct = avg.get('ssm_ratio', 0.5) * 100
            phase = "PEN" if avg.get('lambda_eff', 0) > 0 else "FREE"

            print(
                f"[{config.mode.upper():>11}] "
                f"Step {global_step:>6d} | "
                f"Loss {avg['loss']:.4f} | "
                f"PPL {ppl:.2f} | "
                f"Attn {attn_pct:.1f}% | "
                f"SSM {ssm_pct:.1f}% | "
                f"λ={avg.get('lambda_eff', 0):.3f} | "
                f"LR {lr:.2e} | "
                f"Tok/s {tok_sec:.0f} [{phase}]"
            )

        # Save checkpoint
        if global_step % args.save_every == 0:
            ckpt_path = os.path.join(results_dir, f"ckpt_step{global_step}.pt")
            torch.save({
                "step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": vars(config),
                "metrics": metrics_log[-10:],
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")

            # Save metrics log
            metrics_path = os.path.join(results_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics_log, f, indent=2)

    # =========================================================================
    # Final save
    # =========================================================================
    elapsed = time.time() - start_time

    final_metrics = {
        "mode": config.mode,
        "enable_width_routing": config.enable_width_routing,
        "enable_depth_routing": config.enable_depth_routing,
        "total_steps": global_step,
        "best_ppl": best_ppl,
        "final_loss": metrics_log[-1]["loss"] if metrics_log else None,
        "final_ppl": metrics_log[-1]["ppl"] if metrics_log else None,
        "final_attn_ratio": metrics_log[-1].get("attn_ratio", 0.5) if metrics_log else None,
        "total_time_hours": elapsed / 3600,
        "params_M": model.count_parameters() / 1e6,
    }

    # Save final checkpoint
    torch.save({
        "step": global_step,
        "model": model.state_dict(),
        "config": vars(config),
    }, os.path.join(results_dir, "final_model.pt"))

    # Save all metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    with open(os.path.join(results_dir, "final_results.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE: {config.mode.upper()}")
    print(f"{'='*60}")
    print(f"  Best PPL:       {best_ppl:.4f}")
    print(f"  Final Loss:     {final_metrics['final_loss']:.4f}")
    print(f"  Final Attn%:    {final_metrics['final_attn_ratio']*100:.1f}%")
    print(f"  Training Time:  {elapsed / 3600:.2f} hours")
    print(f"  Results saved:  {results_dir}/")
    print(f"{'='*60}")

    return final_metrics


def compare_results():
    """Compare results across all completed ablation modes."""
    print(f"\n{'='*80}")
    print(f"  ESH WIDTH × DEPTH ABLATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Mode':<15} {'Width':>6} {'Depth':>6} {'PPL':>8} {'α Mean':>8} {'Ponder':>8} {'Time(h)':>8}")
    print(f"{'-'*80}")

    modes = ["baseline", "width_only", "depth_only", "unified"]
    all_results = {}

    for mode in modes:
        fpath = os.path.join("results", mode, "final_results.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                r = json.load(f)
            all_results[mode] = r
            print(
                f"{mode:<15} "
                f"{'✓' if r['enable_width_routing'] else '✗':>6} "
                f"{'✓' if r['enable_depth_routing'] else '✗':>6} "
                f"{r['best_ppl']:>8.4f} "
                f"{r.get('final_alpha_mean', 0.5):>8.3f} "
                f"{r.get('final_avg_ponder', 1.0):>8.2f} "
                f"{r.get('total_time_hours', 0):>8.2f}"
            )
        else:
            print(f"{mode:<15} {'— not yet run —':>65}")

    print(f"{'='*80}")

    if all_results:
        with open("results/ablation_comparison.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Comparison saved to results/ablation_comparison.json")


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "compare":
        compare_results()
    else:
        train(args)
        compare_results()
