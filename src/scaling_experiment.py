import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.config import load_config

def generate_report(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "report.md")
    
    with open(report_path, "w") as f:
        f.write("# GPT-2 Reproduction Results\n\n")
        f.write("## Overview\n")
        f.write("This report was auto-generated. It summarizes the scaling properties and final metrics of the models trained.\n\n")
        
        f.write("## Plots\n")
        f.write("- ![Loss vs Step](plots/nano/loss_vs_step.png)\n")
        f.write("- ![Validation PPL vs Step](plots/nano/ppl_vs_step.png)\n")
        f.write("- ![Scaling Laws](plots/scaling.png)\n\n")
        
        f.write("## Known Deviations From the Paper\n")
        f.write("- **Dataset**: OpenWebText subset instead of full WebText.\n")
        f.write("- **Scale**: Up to 124M parameters, trained on limited Colab budgets.\n")
        f.write("- **Compute budget**: Single T4/A100 GPU for a few hours rather than TPU pods.\n")
        f.write("- **Evaluation suite**: Focused on zero-shot Perplexity and LAMBADA.\n")
        f.write("- **Hyperparameters**: Used standard community-accepted AdamW parameters.\n")
        
    print(f"Report generated at {report_path}")

def scaling_plot(logs_dir, out_path):
    configs = ["nano", "gpt2_small_scaled", "gpt2_small_original"]
    params = [11_000_000, 25_000_000, 124_000_000]
    final_ppls = []
    
    for c in configs:
        log_file = os.path.join(logs_dir, c, "metrics.csv")
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            final_ppls.append(df['val_perplexity'].iloc[-1])
        else:
            final_ppls.append(None)
            
    # Filter valid
    valid_params = [p for p, ppl in zip(params, final_ppls) if ppl is not None]
    valid_ppls = [ppl for ppl in final_ppls if ppl is not None]
    
    if len(valid_params) >= 2:
        plt.figure(figsize=(8, 5))
        plt.plot(valid_params, valid_ppls, marker='o', linestyle='-')
        plt.xscale('log')
        plt.xlabel('Parameter Count')
        plt.ylabel('Final Validation Perplexity')
        plt.title('Scaling Laws (Log-Linear)')
        plt.savefig(out_path)
        plt.close()
        print(f"Scaling plot saved to {out_path}")
    else:
        print("Not enough configs trained yet to generate a scaling plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, default="logs")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.outdir, "plots"), exist_ok=True)
    scaling_plot(args.logs, os.path.join(args.outdir, "plots", "scaling.png"))
    generate_report(args.outdir)
