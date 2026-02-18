
import os
import sys
import torch
from transformers import AutoTokenizer
from data_loader import create_mixed_eval_dataloader
from esh_unified.model import UnifiedModel, UnifiedConfig

# Use existing files
sys.path.insert(0, os.getcwd())

def inspect(mode):
    cwd = os.getcwd()
    ckpt_path = os.path.join(cwd, "results", mode, "final_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"[{mode}] No checkpoint found at {ckpt_path}")
        return

    print(f"[{mode}] Loading checkpoint...")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    # Reconstruct config from checkpoint
    config_dict = checkpoint["config"]
    config = UnifiedConfig(**config_dict)
    
    # Ensure correct weights in the loaded model class (1.0 vs 0.1) 
    # doesn't matter for evaluation since we just forward pass
    model = UnifiedModel(config).to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Load data for a quick check
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Use cache_dir if available, assuming it's in ./data_cache
    loader = create_mixed_eval_dataloader(tokenizer, batch_size=4, max_length=512, cache_dir="./data_cache")
    
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(next(model.parameters()).device)

    print(f"[{mode}] Running inference on 1 batch...")
    with torch.no_grad():
        outputs = model(input_ids, return_routing_stats=True)
    
    stats = outputs["routing_stats"]
    print(f"[{mode}] STATS RECOVERED:")
    print(f"  Alpha Mean:       {stats['alpha_mean']:.4f}")
    if config.enable_depth_routing:
        print(f"  Avg Ponder Steps: {stats['avg_ponder_steps']:.4f} (True Value)")
        print(f"  Halt Prob Mean:   {stats['halt_prob_mean']:.4f}")
    else:
        print(f"  Avg Ponder Steps: 1.0 (Fixed)")
    
    print("-" * 60)

if __name__ == "__main__":
    modes = ["unified", "depth_only"]
    print("INSPECTING RUN 4 CHECKPOINTS FOR TRUE PONDER DEPTH\n")
    for m in modes:
        inspect(m)
