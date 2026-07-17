import os
import pytest
import numpy as np
from src.data import prepare_data, MemmapDataset
from src.utils.config import GPTConfig, DataConfig, ModelConfig

def test_data_pipeline(tmp_path, monkeypatch):
    import os as orig_os
    orig_join = orig_os.path.join
    monkeypatch.setattr("src.data.os.path.join", lambda *args: orig_join(tmp_path, *args[1:]) if args[0] == "data" else orig_join(*args))
    monkeypatch.setattr("src.data.os.makedirs", lambda *args, **kwargs: None)
    
    import yaml
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"data": {"max_documents": 2}, "tokenizer": {"mode": "gpt2_exact"}, "model": {"block_size": 10}}, f)
        
    prepare_data(str(config_path))
    
    bin_file = tmp_path / "train.bin"
    assert bin_file.exists()
    
    arr = np.memmap(bin_file, dtype=np.uint16, mode='r')
    assert len(arr) > 0
    
    dataset = MemmapDataset(bin_file, block_size=10)
    x, y = dataset[0]
    
    assert x.shape == (10,)
    assert y.shape == (10,)
