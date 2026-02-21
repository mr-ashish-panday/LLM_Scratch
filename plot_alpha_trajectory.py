import os
import json
import matplotlib.pyplot as plt

def plot_alpha_trajectories():
    """
    Reads alpha metrics from width_only and unified modes and plots them 
    to visualize the alpha trajectory (burn-in recovery vs collapse).
    """
    modes = ["width_only", "unified"]
    plt.figure(figsize=(10, 6))

    data_found = False
    for mode in modes:
        fpath = os.path.join("results", mode, "metrics.json")
        if not os.path.exists(fpath):
            print(f"Skipping {mode} - {fpath} not found.")
            continue
            
        with open(fpath, "r") as f:
            metrics = json.load(f)
            
        steps = [m["step"] for m in metrics if "alpha_mean" in m]
        alphas = [m["alpha_mean"] for m in metrics if "alpha_mean" in m]
        
        if steps:
            plt.plot(steps, alphas, label=f"{mode}", linewidth=2.5, alpha=0.8)
            data_found = True

    # Reference lines
    plt.axhline(y=0.5, color='g', linestyle='--', alpha=0.7, label='Ideal Balance (0.5)')
    plt.axhline(y=0.25, color='r', linestyle=':', alpha=0.7, label='Typical Collapse point (~0.25)')

    plt.title("Alpha Trajectory: Pathway Collapse vs. Burn-in Recovery", fontsize=14, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Alpha Mean (0 = Pure SSM, 1 = Pure Attention)", fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    out_path = "alpha_trajectory_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    
    if data_found:
        print(f"\n✅ Plot successfully saved to {out_path}!")
        print("Check the plot to see if the alpha trajectory stayed near 0.5 (burn-in success) or dropped to 0.25 (collapse).")
    else:
        print("\n❌ No metrics.json files found yet. Wait for the training runs to log some steps.")

if __name__ == "__main__":
    plot_alpha_trajectories()
