import numpy as np
import time
from model import gpt2_forward, gpt2_forward_unfused
from weights import load_gpt2_weights
from tokenizer import get_tokenizer, encode, decode

def sample(logits, temperature=0.9):
    logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))

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
        print(f"   step {i+1}: {decode(token_ids, tokenizer)}")
    print(f"\nFinal output:\n{decode(token_ids, tokenizer)}")
    print(f"\n--- Timing ---")
    print(f"Prefill:    {prefill_time:.2f}s")
    print(f"Avg time/token: {np.mean(token_times):.2f}s")
    print(f"Tokens/sec:    {1/np.mean(token_times):.2f}")
    
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
    print(f"Avg time/token:  {np.mean(times_no_cache):.2f}s")
    print(f"Tokens/sec:   {1/np.mean(times_no_cache):.2f}")
    
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
    print(f"Tokens/sec:  {1/np.mean(times_cache):.2f}")
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
    benchmark_fusion("In the beginning, scientists discovered that the universe was expanding", runs=3) 
    