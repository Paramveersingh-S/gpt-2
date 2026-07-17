import tiktoken
from typing import List, Union

class Tokenizer:
    def __init__(self, mode: str = "gpt2_exact"):
        self.mode = mode
        if self.mode == "gpt2_exact":
            self.encoder = tiktoken.get_encoding("gpt2")
        elif self.mode == "custom_bpe":
            raise NotImplementedError("custom_bpe is not yet loaded in Tokenizer, use bpe_trainer")
        else:
            raise ValueError(f"Unknown tokenizer mode: {mode}")

    def encode(self, text: str) -> List[int]:
        if self.mode == "gpt2_exact":
            return self.encoder.encode(text, allowed_special={"<|endoftext|>"})
        return []

    def decode(self, tokens: List[int]) -> str:
        if self.mode == "gpt2_exact":
            return self.encoder.decode(tokens)
        return ""
    
    @property
    def vocab_size(self):
        if self.mode == "gpt2_exact":
            return self.encoder.n_vocab
        return 0
