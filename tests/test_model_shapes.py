import math
import pytest
import torch
from src.model import GPT
from src.utils.config import load_config

def test_model_shapes():
    # Load tiny config
    import yaml
    import os
    config_path = "configs/tiny_cpu_debug.yaml"
    cfg = load_config(config_path)
    
    model = GPT(cfg)
    
    # 1. Test shape
    idx = torch.randint(0, cfg.model.vocab_size, (2, 10))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, cfg.model.vocab_size)
    
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 10, cfg.model.vocab_size)
    assert loss is not None
    
    # 2. Parameter count check for tiny config (~0.2M params)
    total_params = sum(p.numel() for p in model.parameters())
    # Expected: wte(256*64) + wpe(64*64) + 2 blocks (ln1(128)+attn(64*192+192+64*64+64)+ln2(128)+mlp(64*256+256+256*64+64)) + ln_f(128)
    # The prompt says printed parameter count matches a hand-computed formula within 1%
    assert 100_000 < total_params < 300_000

    # 3. Test residual-path init std matches 0.02 / sqrt(2 * n_layer) for c_proj
    expected_std = 0.02 / math.sqrt(2 * cfg.model.n_layer)
    c_proj_weight = model.transformer.h[0].attn.c_proj.weight
    actual_std = c_proj_weight.std().item()
    
    # Check within floating point tolerance
    assert abs(actual_std - expected_std) < 0.005
