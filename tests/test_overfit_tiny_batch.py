import pytest
import torch
import torch.nn.functional as F
from src.model import GPT
from src.utils.config import load_config

def test_overfit_tiny_batch():
    config_path = "configs/tiny_cpu_debug.yaml"
    cfg = load_config(config_path)
    
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Fixed tiny batch
    torch.manual_seed(42)
    B, T = 4, cfg.model.block_size
    x = torch.randint(0, cfg.model.vocab_size, (B, T))
    y = torch.randint(0, cfg.model.vocab_size, (B, T))
    
    model.train()
    
    initial_loss = None
    final_loss = None
    
    for step in range(200):
        logits, loss = model(x, y)
        if step == 0:
            initial_loss = loss.item()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        
    assert final_loss < 0.1, f"Model failed to memorize! Final loss: {final_loss}, Initial loss: {initial_loss}"
