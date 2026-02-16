"""Quick verification: test all 4 ablation modes."""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from esh_unified.model import UnifiedConfig, UnifiedModel

modes = [
    (False, False, "baseline"),
    (True,  False, "width_only"),
    (False, True,  "depth_only"),
    (True,  True,  "unified"),
]

for w, d, name in modes:
    c = UnifiedConfig(
        d_model=256, n_layers=2, n_heads=4, n_experts=2,
        max_seq_len=64,
        enable_width_routing=w,
        enable_depth_routing=d,
    )
    m = UnifiedModel(c)
    x = torch.randint(0, 1000, (1, 32))
    out = m(x, labels=x, return_routing_stats=True)

    loss_val = out["loss"].item()
    aux_val = out["aux_loss"].item() if torch.is_tensor(out["aux_loss"]) else float(out["aux_loss"])
    ponder_val = out["avg_ponder_steps"]

    print(f"  {name:>12}: loss={loss_val:.3f}  aux={aux_val:.4f}  ponder={ponder_val:.2f}  -- OK")

print("\nALL 4 MODES PASSED!")
