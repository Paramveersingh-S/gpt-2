import torch
import argparse
from src.utils.config import load_config
from src.model import GPT
from src.tokenizer import Tokenizer

def sample(config_path, checkpoint_path, prompt, num_samples=1, max_new_tokens=50, temperature=1.0, top_k=None):
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = Tokenizer(mode=cfg.tokenizer.mode)
    model = GPT(cfg).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for k in range(num_samples):
            x = idx
            for _ in range(max_new_tokens):
                # crop context if needed
                x_cond = x if x.size(1) <= cfg.model.block_size else x[:, -cfg.model.block_size:]
                logits, _ = model(x_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                x = torch.cat((x, idx_next), dim=1)
            
            out = tokenizer.decode(x[0].tolist())
            print(f"--- Sample {k} ---\n{out}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="The future of AI is")
    args = parser.parse_args()
    sample(args.config, args.checkpoint, args.prompt)
