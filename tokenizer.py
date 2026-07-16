from transformers import GPT2Tokenizer

def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

def encode(text, tokenizer):
    return tokenizer.encode(text)

def decode(token_ids, tokenizer):
    return tokenizer.decode(token_ids)

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    
    text = "The cat sat"
    ids = encode(text, tokenizer)
    back = decode(ids, tokenizer)
    
    print(f"Text:             {text}")
    print(f"Token IDs:        {ids}")
    print(f"Decoded:          {back}")