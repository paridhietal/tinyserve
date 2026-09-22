import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):
    return g * (x - np.mean(x)) / (np.std(x) + eps) + b

def linear(x, W, b):
    return x @ W + b

def attention(x, W, n_head, kv_cache=None):
    x = linear(x, W['c_attn']['w'], W['c_attn']['b'])
    q, k, v = np.split(x, 3, axis=-1)

    def split_heads(x):
        seq_len, d_model = x.shape
        head_dim = d_model // n_head
        x = x.reshape(seq_len, n_head, head_dim)
        return x.transpose(1, 0, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    if kv_cache is not None:
        if kv_cache['k'] is None:
            kv_cache['k'] = k
            kv_cache['v'] = v
        else:
            k = np.concatenate([kv_cache['k'], k], axis=1)
            v = np.concatenate([kv_cache['v'], v], axis=1)
            kv_cache['k'] = k
            kv_cache['v'] = v

    head_dim = q.shape[-1]
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)

    seq_len = q.shape[1]
    mask = np.tril(np.ones((seq_len, seq_len)))
    scores = np.where(mask == 0, -1e10, scores)

    weights = softmax(scores, axis=-1)
    out = weights @ v
    out = out.transpose(1, 0, 2).reshape(seq_len, -1)

    return linear(out, W['c_proj']['w'], W['c_proj']['b'])

def fused_layer_norm_residual(x, residual, g, b, eps=1e-5):
    normed = layer_norm(x, g, b, eps)
    x = x + residual
    return normed, x

def transformer_block(x, block, n_head, kv_cache=None):
    normed1 = layer_norm(x, block['ln_1']['g'], block['ln_1']['b'])
    attn_out = attention(normed1, block['attn'], n_head, kv_cache)
    normed2, x = fused_layer_norm_residual(x, attn_out, block['ln_2']['g'], block['ln_2']['b'])
    h = linear(normed2, block['mlp']['c_fc']['w'], block['mlp']['c_fc']['b'])
    h = gelu(h)
    mlp_out = linear(h, block['mlp']['c_proj']['w'], block['mlp']['c_proj']['b'])
    x = x + mlp_out
    return x

def transformer_block_unfused(x, block, n_head, kv_cache=None):
    normed1 = layer_norm(x, block['ln_1']['g'], block['ln_1']['b'])
    attn_out = attention(normed1, block['attn'], n_head, kv_cache)
    x = x + attn_out
    normed2 = layer_norm(x, block['ln_2']['g'], block['ln_2']['b'])
    h = linear(normed2, block['mlp']['c_fc']['w'], block['mlp']['c_fc']['b'])
    h = gelu(h)
    mlp_out = linear(h, block['mlp']['c_proj']['w'], block['mlp']['c_proj']['b'])
    x = x + mlp_out
    return x

def gpt2_forward(token_ids, weights, n_head=12, kv_cache=None, return_hidden=False):
    token_emb = weights['wte'][token_ids]
    positions = np.arange(len(token_ids))
    pos_emb = weights['wpe'][positions]
    x = token_emb + pos_emb
    for i, block in enumerate(weights['blocks']):
        cache = kv_cache[i] if kv_cache is not None else None
        x = transformer_block(x, block, n_head, cache)
    x = layer_norm(x, weights['ln_f']['g'], weights['ln_f']['b'])
    logits = x @ weights['lm_head']
    if return_hidden:
        return logits, x
    return logits

def gpt2_forward_unfused(token_ids, weights, n_head=12):
    token_emb = weights['wte'][token_ids]
    positions = np.arange(len(token_ids))
    pos_emb = weights['wpe'][positions]
    x = token_emb + pos_emb
    for i, block in enumerate(weights['blocks']):
        x = transformer_block_unfused(x, block, n_head)
    x = layer_norm(x, weights['ln_f']['g'], weights['ln_f']['b'])
    return x @ weights['lm_head']

def draft_head(x, W):
    h = linear(x, W['fc']['w'], W['fc']['b'])
    h = gelu(h)
    return linear(h, W['proj']['w'], W['proj']['b'])
