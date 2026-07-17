import yaml
import argparse
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257

@dataclass
class TokenizerConfig:
    mode: str = "gpt2_exact"

@dataclass
class DataConfig:
    max_documents: int = 10000

@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 0.00025
    max_steps: int = 10000
    eval_interval: int = 500

@dataclass
class GPTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def load_config(config_path: str) -> GPTConfig:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_cfg = ModelConfig(**config_dict.get('model', {}))
    tokenizer_cfg = TokenizerConfig(**config_dict.get('tokenizer', {}))
    data_cfg = DataConfig(**config_dict.get('data', {}))
    training_cfg = TrainingConfig(**config_dict.get('training', {}))
    
    return GPTConfig(model=model_cfg, tokenizer=tokenizer_cfg, data=data_cfg, training=training_cfg)

def get_config_from_cli() -> GPTConfig:
    parser = argparse.ArgumentParser(description="GPT-2 Reproduction")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    # In a full implementation, we'd add CLI overrides here.
    args, _ = parser.parse_known_args()
    return load_config(args.config)
