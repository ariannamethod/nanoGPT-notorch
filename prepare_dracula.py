"""
Prepare the Dracula dataset for character-level language modeling.
Reads dracula.txt, maps characters to integers, creates train/val split.
Saves train.bin, val.bin, and meta.pkl.

Usage:
    python prepare_dracula.py
"""
import os
import pickle
import numpy as np

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_dracula')
os.makedirs(data_dir, exist_ok=True)

input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dracula.txt')

with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"vocab size: {vocab_size}")
print("characters:", ''.join(chars[:50]), '...')

# create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# train/val split (90/10)
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# save meta information
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"\nData saved to {data_dir}/")
print(f"  train.bin: {os.path.getsize(os.path.join(data_dir, 'train.bin')):,} bytes")
print(f"  val.bin:   {os.path.getsize(os.path.join(data_dir, 'val.bin')):,} bytes")
print(f"  meta.pkl:  {os.path.getsize(os.path.join(data_dir, 'meta.pkl')):,} bytes")
print("\nDracula awaits. The training can begin.")
