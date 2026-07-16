import numpy as np
import torch
from pathlib import Path

def load_gpt2_weights(model_size="gpt2"):
    print("Downloading GPT-2 weights...")
    from transformers import GPT2Model
    model = GPT2Model.from_pretrained(model_size)
    model.eval()
    
    sd = model.state_dict()
    
    #Convert all tensors to numpy
    sd = {k: v.numpy() for k, v in sd.items()}
    
    #build the weights dictionary
    weights = {}
    
    #token + positional embeddings
    weights['wte'] = sd['wte.weight']
    weights['wpe'] = sd['wpe.weight']
    
    #12 transformer blocks
    weights['blocks'] = []
    for i in range(12):
        block = {
            'ln_1': {
                'g': sd[f'h.{i}.ln_1.weight'],
                'b': sd[f'h.{i}.ln_1.bias']
            },
            'attn': {
                'c_attn': {
                    'w': sd[f'h.{i}.attn.c_attn.weight'],
                    'b': sd[f'h.{i}.attn.c_attn.bias']
                },
                'c_proj': {
                    'w': sd[f'h.{i}.attn.c_proj.weight'],
                    'b': sd[f'h.{i}.attn.c_proj.bias']
                }
            },
            'ln_2': {
                'g': sd[f'h.{i}.ln_2.weight'],
                'b': sd[f'h.{i}.ln_2.bias']
            },
            'mlp': {
                'c_fc': {
                    'w': sd[f'h.{i}.mlp.c_fc.weight'],
                    'b': sd[f'h.{i}.mlp.c_fc.bias']                
                    },
                'c_proj': {
                    'w': sd[f'h.{i}.mlp.c_proj.weight'],
                    'b': sd[f'h.{i}.mlp.c_proj.bias']
                }
            }
        }
        weights['blocks'].append(block)
        
    #final layernorm
    weights['ln_f'] = {
        'g': sd['ln_f.weight'],
        'b': sd['ln_f.bias']
    }
    #lm_head - GPT2Model doesn't include it, so we tie it to wte
    weights['lm_head'] = weights['wte'].T

    print("Weights loaded successfully.")
    print(f"  wte shape:  {weights['wte'].shape}")
    print(f"  wpe shape:  {weights['wpe'].shape}")
    print(f"  blocks:     {len(weights['blocks'])}")
    print(f"  lm_head shape: {weights['lm_head'].shape}")    

    return weights 

if __name__ == "__main__":
    weights = load_gpt2_weights()
        
    