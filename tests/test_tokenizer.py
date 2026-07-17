import pytest
from src.tokenizer import Tokenizer

def test_tokenizer_roundtrip():
    tokenizer = Tokenizer(mode="gpt2_exact")
    test_cases = [
        "",
        "Hello, world!",
        "   Whitespace test   ",
        "Unicode: 😊 🚀",
        "Non-Latin: 你好, 안녕, नमस्ते",
        "Special chars: !@#$%^&*()_+{}|:<>?-=[]\\;',./",
        "<|endoftext|>"
    ]

    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert text == decoded, f"Failed roundtrip for: {text!r}"

def test_tokenizer_vocab_size():
    tokenizer = Tokenizer(mode="gpt2_exact")
    assert tokenizer.vocab_size == 50257
