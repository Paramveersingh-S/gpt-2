import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_logs(logdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    log_file = os.path.join(logdir, "metrics.csv")
    if not os.path.exists(log_file):
        print(f"No metrics found at {log_file}")
        return
        
    df = pd.read_csv(log_file)
    
    # 1. Train and Val loss
    plt.figure(figsize=(10, 5))
    plt.plot(df['step'], df['train_loss'], label='Train Loss')
    plt.plot(df['step'], df['val_loss'], label='Val Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss vs Step')
    plt.legend()
    plt.savefig(os.path.join(outdir, "loss_vs_step.png"))
    plt.close()
    
    # 2. Learning Rate
    plt.figure(figsize=(10, 5))
    plt.plot(df['step'], df['learning_rate'])
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Step')
    plt.savefig(os.path.join(outdir, "lr_vs_step.png"))
    plt.close()
    
    # 3. Validation PPL
    plt.figure(figsize=(10, 5))
    plt.plot(df['step'], df['val_perplexity'])
    plt.xlabel('Step')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity vs Step')
    plt.savefig(os.path.join(outdir, "ppl_vs_step.png"))
    plt.close()
    
    print(f"Plots saved to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    plot_logs(args.logdir, args.out)
