"""
Debug script: isolates unified mode hang.
Runs a single forward+backward pass with SYNTHETIC data (no streaming).
If this hangs, the problem is in the model. If it works, the problem is data loading.
"""
import os, sys, time, torch
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from esh_unified.model import UnifiedConfig, UnifiedModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create unified config (both width + depth routing ON)
    config = UnifiedConfig(
        vocab_size=50257,
        d_model=768,
        n_layers=8,
        n_heads=12,
        n_experts=4,
        max_seq_len=512,
        use_checkpoint=True,
        enable_width_routing=True,
        enable_depth_routing=True,
        max_ponder_steps=3,
        ponder_cost_weight=0.5,
        use_moe=True,
    )

    print("Creating model...")
    model = UnifiedModel(config).to(device)
    model.train()
    model.set_global_step(0)
    print(f"Model created: {model.count_parameters()/1e6:.1f}M params")

    # Synthetic data — no network dependencies
    input_ids = torch.randint(0, 50257, (4, 512), device=device)
    scaler = GradScaler()

    print("\n--- Forward pass ---")
    t0 = time.time()

    with autocast(enabled=True):
        print("  [1/4] Entering model.forward()...")
        outputs = model(input_ids, labels=input_ids, return_routing_stats=True)
        print(f"  [2/4] Forward done in {time.time()-t0:.2f}s")

        loss = outputs["loss"]
        aux_loss = outputs["aux_loss"]
        ponder_cost = outputs["ponder_cost"]
        total_loss = loss + aux_loss + 0.5 * ponder_cost
        print(f"  Loss={loss.item():.4f}, Aux={aux_loss.item():.4f}, Ponder={ponder_cost:.4f}")

    print("  [3/4] Starting backward...")
    t1 = time.time()
    scaler.scale(total_loss).backward()
    print(f"  [4/4] Backward done in {time.time()-t1:.2f}s")

    print(f"\nTotal time: {time.time()-t0:.2f}s")
    print(f"Alpha mean: {outputs['routing_stats'][0].get('alpha_mean', 'N/A')}")
    print(f"Avg ponder: {outputs['avg_ponder_steps']:.2f}")
    print("\n✅ Unified forward+backward PASSED — model is fine, issue is elsewhere.")

if __name__ == "__main__":
    main()
