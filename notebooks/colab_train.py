import os

def main():
    print("GPT-2 Reproduction - Colab/Server Training Script")
    print("Checking GPU...")
    os.system("nvidia-smi")
    
    print("Cloning repository if not present...")
    if not os.path.exists('gpt-2'):
        os.system("git clone https://github.com/Paramveersingh-S/gpt-2.git")
    
    # Normally on Colab we cd, but in a script we'll just run commands from the repo root
    # if we are already in the repo, otherwise we change dir
    if os.path.exists('gpt-2/requirements.txt'):
        os.chdir('gpt-2')
        os.system("git pull origin main")
    
    print("Installing dependencies...")
    os.system("pip install -r requirements.txt")
    
    # Note: Google Drive mounting only works in IPython/Colab notebooks natively.
    # In a simple py script on a standard server, we just use local paths.
    # If running on Colab via !python, the user should mount drive manually beforehand.
    print("Ensuring checkpoints directory exists...")
    os.makedirs("checkpoints", exist_ok=True)
    
    config = "configs/nano.yaml"
    
    print(f"Preparing data using config {config}...")
    os.system(f"python src/data.py --config {config}")
    
    print(f"Training using config {config}...")
    os.system(f"python src/train.py --config {config}")
    
    print(f"Evaluating model...")
    os.system(f"python src/evaluate.py --config {config} --checkpoint checkpoints/nano/best.pt")
    
    print("Generating plots...")
    os.system("python -m src.utils.plotting --logdir logs/nano --out results/plots/nano/")
    
    print("Done. Plots saved in results/plots/nano/")

if __name__ == "__main__":
    main()
