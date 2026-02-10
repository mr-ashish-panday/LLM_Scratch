"""
Generate with Random Router Model (Ablation Test)
==================================================
Loads the Random Router baseline checkpoint and tests
whether it shows any α differentiation (it shouldn't).
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_random_baseline import create_random_router_model
from generate import generate_with_routing, print_routing_summary


def main():
    parser = argparse.ArgumentParser(description="Generate with Random Router")
    parser.add_argument("--checkpoint", type=str,
                        default="./esh_random_baseline_checkpoints/final.pt")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device

    # Create model with Random Router (same as training)
    print("Creating model with Random Router...")
    model, config = create_random_router_model()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint.get("step", "unknown")
        print(f"Loaded checkpoint from step {step}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.prompt:
        _, routing_info = generate_with_routing(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            device=device,
        )
        print_routing_summary(routing_info)
    else:
        # Default: run both Math and Story prompts
        prompts = [
            ("MATH", "Question: What is 5 + 7? Answer:"),
            ("STORY", "Once upon a time there was a little bunny"),
        ]
        for label, prompt in prompts:
            print(f"\n{'='*50}")
            print(f"  [{label}] Prompt")
            print(f"{'='*50}")
            _, routing_info = generate_with_routing(
                model, tokenizer, prompt,
                max_new_tokens=30,
                device=device,
            )
            print_routing_summary(routing_info)


if __name__ == "__main__":
    main()
