# TinyServe

A GPT-2 inference engine built from scratch in pure NumPy. No PyTorch. No CUDA. No abstractions.

Every matrix multiply, every attention head, every residual connection — written by hand.

---

## Why?

Most people call `model.generate()` and move on. I wanted to know what actually happens inside that black box — so I built it myself, from the matrix multiplies up.

TinyServe implements the same techniques production inference engines like vLLM use — KV caching, kernel fusion — but in ~300 lines of NumPy you can actually read.

---

## What's inside

model.py      — full GPT-2 forward pass (attention, layernorm, MLP, residuals)
weights.py    — downloads and organizes real GPT-2 weights from HuggingFace
tokenizer.py  — text ↔ token ID conversion
generate.py   — text generation with temperature sampling + benchmarks

---

## How it works

**Forward pass** — token embeddings + positional embeddings → 12 transformer blocks → logits over 50,257 tokens. Every operation implemented manually:

- Multi-head causal self-attention with scaled dot-product
- KV cache for efficient autoregressive generation  
- Fused LayerNorm + residual kernels
- Temperature sampling for non-deterministic generation

**No PyTorch in the forward pass.** Weights are loaded once using HuggingFace, then it's pure NumPy all the way down.

---

## Benchmarks (CPU, GPT-2 124M)

| Optimization | Speedup |
|---|---|
| KV Cache | 1.4x |
| Fused LayerNorm + Residual | 1.66x |
| Combined | ~2.3x over naive baseline |

---

## Run it

```bash
git clone https://github.com/yourusername/tinyserve
cd tinyserve
python -m venv venv
venv\Scripts\activate        # Windows
pip install numpy requests tqdm torch transformers
python generate.py
```

First run downloads GPT-2 weights (~500MB, cached after).

---

## Example output

Prompt: In the beginning, scientists discovered that
Prefill done in 2.1s
step 1: In the beginning, scientists discovered that,
step 2: In the beginning, scientists discovered that, this
...
Tokens/sec: 1.62

---

## What I learned

- Why transformers use pre-norm instead of post-norm (gradient flow in deep stacks)
- Why KV cache works — K and V don't change for past tokens, only Q does
- Why kernel fusion matters — memory bandwidth is the bottleneck, not compute
- How batched matrix multiplication maps to multi-head attention

---

## What's next

- [ ] Multi-token prediction (MTP)
- [ ] Prefill-decode disaggregation
- [ ] INT8 quantization
