"""
nanoGPT-notorch sampling.

usage:
    python sample.py
    python sample.py --weights weights/nanogpt.bin --prompt "The Count"
"""

import os
import sys
import argparse
from model import GPT, GPTConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/nanogpt.bin')
    parser.add_argument('--dataset', type=str, default='dracula.txt')
    parser.add_argument('--prompt', type=str, default='CHAPTER I\n\n')
    parser.add_argument('--max_new', type=int, default=300)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()

    with open(args.dataset, 'rb') as f:
        raw = f.read()
    chars = sorted(set(raw))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for c, i in char_to_id.items()}

    meta_path = args.weights + '.meta'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            lines = f.read().strip().split('\n')
        dim = int(lines[3]) if len(lines) > 3 else 320
        heads = int(lines[4]) if len(lines) > 4 else 8
        layers = int(lines[5]) if len(lines) > 5 else 8
        ctx = int(lines[6]) if len(lines) > 6 else 256
    else:
        dim, heads, layers, ctx = 320, 8, 8, 256

    config = GPTConfig(block_size=ctx, vocab_size=len(chars),
                       n_layer=layers, n_head=heads, n_embd=dim)
    model = GPT(config)
    model.load_weights(args.weights)
    print(f"loaded {model.count_params():,} params from {args.weights}\n")

    for s in range(args.num_samples):
        ids = [char_to_id.get(b, 0) for b in args.prompt.encode()]
        gen = model.generate(ids, max_new=args.max_new,
                           temperature=args.temperature, top_k=args.top_k)
        text = ''.join(chr(id_to_char[g]) if g in id_to_char else '?' for g in gen)
        print(f"--- sample {s+1} ---")
        print(f"{args.prompt}{text}\n")

if __name__ == '__main__':
    main()
