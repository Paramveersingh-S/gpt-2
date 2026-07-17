import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from src.tokenizer import Tokenizer
from src.utils.config import load_config
import argparse

def prepare_data(config_path: str):
    cfg = load_config(config_path)
    tokenizer = Tokenizer(mode=cfg.tokenizer.mode)
    
    # We will use Elriggs/openwebtext-100k for faster pipeline iteration (parquet format)
    # In a real run, this would be "openwebtext"
    dataset = load_dataset("Elriggs/openwebtext-100k", split="train")
    
    if cfg.data.max_documents > 0:
        dataset = dataset.select(range(min(len(dataset), cfg.data.max_documents)))
        
    os.makedirs("data", exist_ok=True)
    bin_path = os.path.join("data", "train.bin")
    
    # Tokenize and shard
    all_tokens = []
    eot = tokenizer.encode("<|endoftext|>")[0]
    
    # We write a memory-mapped array for efficiency, but for this reproduction
    # we just build it in memory and save.
    for item in dataset:
        text = item["text"]
        if len(text.strip()) < 10:
            continue
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(eot)
        
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f"Saved {len(all_tokens)} tokens to {bin_path}")

class MemmapDataset(Dataset):
    def __init__(self, bin_path, block_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        
    def __len__(self):
        return max(0, len(self.data) - self.block_size)
        
    def __getitem__(self, idx):
        x = torch.from_numpy((self.data[idx:idx+self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[idx+1:idx+1+self.block_size]).astype(np.int64))
        return x, y

def get_dataloader(config, split='train'):
    bin_path = os.path.join("data", f"{split}.bin")
    if not os.path.exists(bin_path):
        # Fallback to train.bin if val doesn't exist for simplicity in debugging
        bin_path = os.path.join("data", "train.bin")
        
    dataset = MemmapDataset(bin_path, config.model.block_size)
    loader = DataLoader(
        dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        pin_memory=torch.cuda.is_available()
    )
    return loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    prepare_data(args.config)
