import numpy as np
import time
from model import gpt2_forward, gpt2_forward_unfused, draft_head, linear, gelu, softmax
from weights import load_gpt2_weights
from tokenizer import get_tokenizer, encode, decode

def sample(logits, temperature=0.9):
    logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))

def cross_entropy_loss(logits, target_id):
    probs = softmax(logits)
    return -np.log(probs[target_id] + 1e-8)

TRAINING_TEXTS = [
    "The cat sat on the mat and looked around.",
    "Scientists discovered that the universe is expanding.",
    "Machine learning models learn from data.",
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning there was darkness and then light appeared.",
    "Neural networks are inspired by the human brain.",
    "The stock market crashed after the announcement.",
    "She walked into the room and saw something strange.",
    "Python is a popular programming language for data science.",
    "The weather today is sunny with a chance of rain.",
]

def init_draft_weights(d_model=768, vocab_size=50257):
    return {
        'fc':   {'w': np.random.randn(d_model, d_model) * 0.02,
                 'b': np.zeros(d_model)},
        'proj': {'w': np.random.randn(d_model, vocab_size) * 0.02,
                 'b': np.zeros(vocab_size)}
    }

def train_draft_heads(draft_weights, weights, epochs=3, lr=0.01):
    tokenizer = get_tokenizer()
    for epoch in range(epochs):
        total_loss = 0
        n_samples = 0
        for text in TRAINING_TEXTS:
            token_ids = encode(text, tokenizer)
            if len(token_ids) < 3:
                continue
            _, hidden = gpt2_forward(np.array(token_ids), weights, return_hidden=True)
            for pos in range(len(token_ids) - 1):
                x = hidden[pos]
                target = token_ids[pos + 1]
                h = linear(x, draft_weights[0]['fc']['w'], draft_weights[0]['fc']['b'])
                h_gelu = gelu(h)
                logits = linear(h_gelu, draft_weights[0]['proj']['w'], draft_weights[0]['proj']['b'])
                loss = cross_entropy_loss(logits, target)
                total_loss += loss
                n_samples += 1
                probs = softmax(logits)
                probs[target] -= 1
                draft_weights[0]['proj']['w'] -= lr * np.outer(h_gelu, probs)
                draft_weights[0]['proj']['b'] -= lr * probs
        print(f"Epoch {epoch+1} loss: {total_loss/n_samples:.4f}")
    return draft_weights

def generate(prompt, n_tokens=20, temperature=0.7):
    print("Loading weights...")
    weights = load_gpt2_weights()
    tokenizer = get_tokenizer()
    token_ids = encode(prompt, tokenizer)
    print(f"Prompt: {prompt}")
    print(f"\nGenerating {n_tokens} tokens...\n")
    kv_cache = [{'k': None, 'v': None} for _ in range(12)]
    start = time.time()
    logits = gpt2_forward(np.array(token_ids), weights, kv_cache=kv_cache)
    prefill_time = time.time() - start
    print(f"Prefill done in {prefill_time:.2f}s")
    token_times = []
    for i in range(n_tokens):
        next_token = sample(logits[-1], temperature)
        token_ids.append(next_token)
        t0 = time.time()
        logits = gpt2_forward(np.array([next_token]), weights, kv_cache=kv_cache)
        token_times.append(time.time() - t0)
        print(f"  step {i+1}: {decode(token_ids, tokenizer)}")
    print(f"\nFinal output:\n{decode(token_ids, tokenizer)}")
    print(f"\n--- Timing ---")
    print(f"Prefill:        {prefill_time:.2f}s")
    print(f"Avg time/token: {np.mean(token_times):.2f}s")
    print(f"Tokens/sec:     {1/np.mean(token_times):.2f}")

def generate_mtp(prompt, n_tokens=20, temperature=0.7, draft_weights=None, weights=None):
    if weights is None:
        weights = load_gpt2_weights()
    tokenizer = get_tokenizer()
    token_ids = encode(prompt, tokenizer)
    kv_cache = [{'k': None, 'v': None} for _ in range(12)]
    if draft_weights is None:
        draft_weights = [init_draft_weights(), init_draft_weights()]
    logits, hidden = gpt2_forward(np.array(token_ids), weights, kv_cache=kv_cache, return_hidden=True)
    token_times = []
    accepted_counts = []
    i = 0
    while i < n_tokens:
        t0 = time.time()
        main_token = sample(logits[-1], temperature)
        draft_token_1 = int(np.argmax(draft_head(hidden[-1], draft_weights[0])))
        logits, hidden = gpt2_forward(np.array([main_token]), weights, kv_cache=kv_cache, return_hidden=True)
        verified_token_1 = int(np.argmax(logits[-1]))
        token_ids.append(main_token)
        i += 1
        accepted = 1
        if draft_token_1 == verified_token_1 and i < n_tokens:
            token_ids.append(draft_token_1)
            i += 1
            accepted += 1
        token_times.append(time.time() - t0)
        accepted_counts.append(accepted)
        print(f"  step {i}: {decode(token_ids, tokenizer)}")
    print(f"\nFinal output:\n{decode(token_ids, tokenizer)}")
    print(f"\n--- MTP Stats ---")
    print(f"Avg tokens/step:  {np.mean(accepted_counts):.2f}")
    print(f"Avg time/step:    {np.mean(token_times):.2f}s")
    print(f"Effective tok/s:  {np.mean(accepted_counts)/np.mean(token_times):.2f}")

def benchmark_kvcache(prompt, n_tokens=10):
    print("Loading weights...")
    weights = load_gpt2_weights()
    tokenizer = get_tokenizer()
    token_ids = encode(prompt, tokenizer)
    print("\n--- Without KV Cache ---")
    times_no_cache = []
    ids_copy = token_ids.copy()
    for i in range(n_tokens):
        t0 = time.time()
        logits = gpt2_forward(np.array(ids_copy), weights)
        times_no_cache.append(time.time() - t0)
        next_token = int(np.argmax(logits[-1]))
        ids_copy.append(next_token)
    print(f"Avg time/token: {np.mean(times_no_cache):.2f}s")
    print(f"Tokens/sec:     {1/np.mean(times_no_cache):.2f}")
    print("\n--- With KV Cache ---")
    kv_cache = [{'k': None, 'v': None} for _ in range(12)]
    ids_copy = token_ids.copy()
    logits = gpt2_forward(np.array(ids_copy), weights, kv_cache=kv_cache)
    times_cache = []
    for i in range(n_tokens):
        next_token = int(np.argmax(logits[-1]))
        ids_copy.append(next_token)
        t0 = time.time()
        logits = gpt2_forward(np.array([next_token]), weights, kv_cache=kv_cache)
        times_cache.append(time.time() - t0)
    print(f"Avg time/token: {np.mean(times_cache):.2f}s")
    print(f"Tokens/sec:     {1/np.mean(times_cache):.2f}")
    print(f"\nKV Cache Speedup: {np.mean(times_no_cache)/np.mean(times_cache):.1f}x")

def benchmark_fusion(prompt, runs=3):
    print("Loading weights...")
    weights = load_gpt2_weights()
    tokenizer = get_tokenizer()
    token_ids = np.array(encode(prompt, tokenizer))
    print("\n--- Without Fusion ---")
    t0 = time.time()
    for _ in range(runs):
        gpt2_forward_unfused(token_ids, weights)
    unfused_time = (time.time() - t0) / runs
    print(f"Avg time: {unfused_time:.2f}s")
    print("\n--- With Fusion ---")
    t0 = time.time()
    for _ in range(runs):
        gpt2_forward(token_ids, weights)
    fused_time = (time.time() - t0) / runs
    print(f"Avg time: {fused_time:.2f}s")
    print(f"\nFusion Speedup: {unfused_time/fused_time:.2f}x")

if __name__ == "__main__":
    weights = load_gpt2_weights()
    draft_weights = [init_draft_weights(), init_draft_weights()]
    print("Training draft heads...")
    draft_weights = train_draft_heads(draft_weights, weights, epochs=3, lr=0.01)
    print("\nGenerating with MTP...")
    generate_mtp("In the beginning, scientists discovered that", n_tokens=20, draft_weights=draft_weights, weights=weights)
