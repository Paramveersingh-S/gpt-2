import os

def main():
    print("GPT-2 Reproduction - Colab/Server Training Script")
    print("Checking GPU...")
    os.system("nvidia-smi")
    
    print("Cloning repository if not present...")
    # Check if we are already in the repo root (i.e. requirements.txt and src/ exist here)
    if os.path.exists('requirements.txt') and os.path.exists('src'):
        print("Already inside the gpt-2 repository.")
    else:
        if not os.path.exists('gpt-2'):
            os.system("git clone https://github.com/Paramveersingh-S/gpt-2.git")
        os.chdir('gpt-2')
        os.system("git pull origin main")
    
    print("Installing dependencies...")
    os.system("pip install -r requirements.txt")
    
    print("Ensuring checkpoints directory exists...")
    os.makedirs("checkpoints", exist_ok=True)
    
    config = "configs/nano.yaml"
    
    # Use python -m to ensure the current directory is in sys.path
    print(f"Preparing data using config {config}...")
    os.system(f"python -m src.data --config {config}")
    
    print(f"Training using config {config}...")
    os.system(f"python -m src.train --config {config}")
    
    print(f"Evaluating model...")
    os.system(f"python -m src.evaluate --config {config} --checkpoint checkpoints/nano/best.pt")
    
    print("Generating plots...")
    os.system("python -m src.utils.plotting --logdir logs/nano --out results/plots/nano/")
    
    print("Done. Plots saved in results/plots/nano/")

if __name__ == "__main__":
    main()
