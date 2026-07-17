import os
import time
import math
import argparse
import torch
from torch.nn.utils import clip_grad_norm_

from src.utils.config import load_config
from src.model import GPT
from src.data import get_dataloader
from src.utils.logger import Logger

def get_lr(it, max_steps, max_lr, warmup_steps=100):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return max_lr * 0.1
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return max_lr * 0.1 + coeff * (max_lr - max_lr * 0.1)

def train(config_path, resume_from=None):
    cfg = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if device=="cuda" else torch.float32
    
    model = GPT(cfg).to(device)
    
    # Optimizer (nanoGPT style AdamW setup)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.training.learning_rate, betas=(0.9, 0.95))
    
    start_step = 0
    tokens_seen = 0
    
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        tokens_seen = checkpoint['tokens_seen']
        print(f"Resumed from {resume_from} at step {start_step}")
        
    train_loader = get_dataloader(cfg, 'train')
    val_loader = get_dataloader(cfg, 'val')
    
    # Just a placeholder for actual infinite iteration
    train_iter = iter(train_loader)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    
    log_dir = os.path.join("logs", os.path.basename(config_path).split('.')[0])
    ckpt_dir = os.path.join("checkpoints", os.path.basename(config_path).split('.')[0])
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_dir)
    
    t0 = time.time()
    best_val_loss = float('inf')
    
    for step in range(start_step, cfg.training.max_steps):
        lr = get_lr(step, cfg.training.max_steps, cfg.training.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        
        with torch.autocast(device_type=device, dtype=dtype, enabled=(device=="cuda")):
            logits, loss = model(x, y)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        tokens_seen += x.numel()
        
        if step % cfg.training.eval_interval == 0 or step == cfg.training.max_steps - 1:
            # Eval
            model.eval()
            with torch.no_grad():
                # Simplified eval for the sake of the demo, in real run we average over many batches
                try:
                    vx, vy = next(iter(val_loader))
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.autocast(device_type=device, dtype=dtype, enabled=(device=="cuda")):
                        _, vloss = model(vx, vy)
                    val_loss = vloss.item()
                except StopIteration:
                    val_loss = loss.item() # fallback if no val data
                    
            val_perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')
            
            t1 = time.time()
            wall_clock_seconds = t1 - t0
            
            logger.log(step, loss.item(), val_loss, val_perplexity, lr, tokens_seen, wall_clock_seconds)
            print(f"Step {step}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, ppl={val_perplexity:.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(ckpt_dir, "best.pt")
                torch.save({
                    'step': step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'tokens_seen': tokens_seen,
                    'val_loss': val_loss,
                }, ckpt_path)
            
            model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()
    train(args.config, args.resume_from)
