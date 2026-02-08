"""
ESH Interpretable Generation Script
====================================
Visualize the "Elastic Intelligence" of ESH by showing which tokens
are routed to Attention vs SSM paths.

Usage:
    python generate.py --prompt "Once upon a time" --max_tokens 100
    python generate.py --interactive
"""

import argparse
import torch
from transformers import AutoTokenizer
from pathlib import Path

from esh import ESHModel
from esh.model import esh_scaled


# ANSI Color codes for terminal output
class Colors:
    BLUE = '\033[94m'      # Attention-heavy tokens
    GREEN = '\033[92m'     # SSM-heavy tokens
    YELLOW = '\033[93m'    # Balanced tokens
    BOLD = '\033[1m'
    RESET = '\033[0m'


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load ESH model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model with same config as training
    config = esh_scaled()
    model = ESHModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint.get("step", "unknown")
        print(f"Loaded checkpoint from step {step}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return model


def get_routing_color(alpha: float) -> str:
    """Get color based on attention ratio (alpha)."""
    if alpha > 0.6:
        return Colors.BLUE   # Attention-heavy
    elif alpha < 0.4:
        return Colors.GREEN  # SSM-heavy
    else:
        return Colors.YELLOW # Balanced


def generate_with_routing(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    show_routing: bool = True,
    device: str = "cuda",
):
    """
    Generate text with routing visualization.
    
    Returns:
        generated_text: The full generated text
        routing_info: List of (token, alpha) tuples
    """
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    
    # Track routing decisions
    routing_info = []
    generated_tokens = []
    
    # Print prompt
    if show_routing:
        print(f"\n{Colors.BOLD}Prompt:{Colors.RESET} {prompt}")
        print(f"{Colors.BOLD}Generated:{Colors.RESET} ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            input_cond = input_ids[:, -model.config.max_seq_len:]
            
            # Forward pass with routing stats
            outputs = model(input_cond, return_routing_stats=True)
            logits = outputs["logits"][:, -1, :] / temperature
            routing_stats = outputs["routing_stats"]
            
            # Get average attention ratio across layers
            avg_alpha = sum(
                stats["attention_ratio"] for stats in routing_stats
            ) / len(routing_stats)
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode token
            token_text = tokenizer.decode(next_token[0])
            
            # Store routing info
            routing_info.append({
                "token": token_text,
                "token_id": next_token.item(),
                "alpha": avg_alpha,
                "path": "Attention" if avg_alpha > 0.5 else "SSM"
            })
            generated_tokens.append(token_text)
            
            # Print colored token
            if show_routing:
                color = get_routing_color(avg_alpha)
                print(f"{color}{token_text}{Colors.RESET}", end="", flush=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    if show_routing:
        print("\n")
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text, routing_info


def print_routing_summary(routing_info: list):
    """Print summary of routing decisions."""
    if not routing_info:
        return
    
    attn_count = sum(1 for r in routing_info if r["alpha"] > 0.5)
    ssm_count = len(routing_info) - attn_count
    avg_alpha = sum(r["alpha"] for r in routing_info) / len(routing_info)
    
    print(f"\n{Colors.BOLD}═══ Routing Summary ═══{Colors.RESET}")
    print(f"Total tokens: {len(routing_info)}")
    print(f"{Colors.BLUE}Attention-routed:{Colors.RESET} {attn_count} ({100*attn_count/len(routing_info):.1f}%)")
    print(f"{Colors.GREEN}SSM-routed:{Colors.RESET} {ssm_count} ({100*ssm_count/len(routing_info):.1f}%)")
    print(f"Average α: {avg_alpha:.3f}")
    
    print(f"\n{Colors.BOLD}Legend:{Colors.RESET}")
    print(f"  {Colors.BLUE}■ Blue{Colors.RESET} = Attention-heavy (α > 0.6)")
    print(f"  {Colors.GREEN}■ Green{Colors.RESET} = SSM-heavy (α < 0.4)")
    print(f"  {Colors.YELLOW}■ Yellow{Colors.RESET} = Balanced (0.4 ≤ α ≤ 0.6)")


def interactive_mode(model, tokenizer, device):
    """Interactive generation loop."""
    print(f"\n{Colors.BOLD}═══ ESH Interactive Generation ═══{Colors.RESET}")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")
    
    while True:
        try:
            prompt = input(f"{Colors.BOLD}>>> {Colors.RESET}")
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt.strip():
                continue
            
            _, routing_info = generate_with_routing(
                model, tokenizer, prompt,
                max_new_tokens=100,
                device=device,
            )
            print_routing_summary(routing_info)
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="ESH Interpretable Generation")
    parser.add_argument("--checkpoint", type=str, 
                        default="./esh_neurips_checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cuda/cpu) - default CPU to allow testing during training")
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Available checkpoints:")
        checkpoint_dir = Path("./esh_neurips_checkpoints")
        if checkpoint_dir.exists():
            for f in checkpoint_dir.glob("*.pt"):
                print(f"  {f}")
        return
    
    # Load model
    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.prompt:
        _, routing_info = generate_with_routing(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print_routing_summary(routing_info)
    else:
        # Demo with default prompt
        demo_prompts = [
            "Once upon a time",
            "The quick brown fox",
            "In the year 2024, artificial intelligence",
        ]
        for prompt in demo_prompts:
            _, routing_info = generate_with_routing(
                model, tokenizer, prompt,
                max_new_tokens=50,
                device=device,
            )
            print_routing_summary(routing_info)
            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
