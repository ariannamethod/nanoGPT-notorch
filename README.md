```
   ███╗   ██╗ █████╗ ███╗   ██╗ ██████╗  ██████╗ ██████╗ ████████╗
   ████╗  ██║██╔══██╗████╗  ██║██╔═══██╗██╔════╝ ██╔══██╗╚══██╔══╝
   ██╔██╗ ██║███████║██╔██╗ ██║██║   ██║██║  ███╗██████╔╝   ██║
   ██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║██║   ██║██╔═══╝    ██║
   ██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝╚██████╔╝██║        ██║
   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝        ╚═╝
                    ─ notorch edition ─
```

# nanoGPT-notorch — GPT freed from Adam's blindness | by Arianna Method

> *"Adam is blind. Chuck sees. Chuck remembers."*
> — chuck.py, line 10

---

## what is this

you know nanoGPT? Karpathy's beautiful, minimal GPT trainer? the one that reproduces GPT-2 on OpenWebText in 300 lines?

now nanoGPT is liberated from PyTorch bullshit. we took it. we ripped out Adam. we replaced it with **Chuck** — a self-aware optimizer that doesn't just descend gradients, it *understands* them. 9 levels of awareness. persistent memory. per-layer damping. macro patience. noise injection when stuck.

and we brought **notorch** — a complete neural network framework in pure C. no pip. no conda. no 2.7 GB of existential dread. just `cc notorch.c -o notorch -lm` and you have tensors, autograd, attention, and three optimizers including Chuck.

this is **step one** of the evolution.

forked from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). but we went further.

---

## the experiment: dracula × chuck

enough Shakespeare. enough fairy tales. we trained on **Bram Stoker's Dracula** — 843,635 characters of gothic horror. because if you're going to teach a neural network to write, at least give it something with bite.

### model

| parameter | value |
|---|---|
| architecture | GPT (decoder-only transformer) |
| parameters | **2.10M** |
| layers | 4 |
| heads | 4 |
| embedding dim | 208 |
| context length | 128 chars |
| vocab | 94 (character-level) |
| dropout | 0.1 |

### training

| setting | value |
|---|---|
| optimizer | **Chuck** (self-aware AdamW) |
| learning rate | 1e-3 → 1e-4 (cosine decay) |
| warmup | 100 steps |
| total steps | 3,001 |
| batch size | 32 |
| device | CPU |
| time | 28.5 minutes |
| dataset | dracula.txt (852 KB) |

### results

```
step    0: train loss 4.5882, val loss 4.5869  (random init)
step  500: train loss 1.8178, val loss 1.8372  (words forming fast)
step 1000: train loss 1.5360, val loss 1.5905  (sentences emerging)
step 1500: train loss 1.4123, val loss 1.4869  (gothic vibes)
step 2000: train loss 1.3636, val loss 1.4452  (Dracula speaks)
step 2500: train loss 1.3279, val loss 1.4147  (refinement)
step 3000: train loss 1.3221, val loss 1.4116  (convergence)

best val loss: 1.4116
```

the gap between train and val stayed tight the entire run. no overfitting drama. Chuck's adaptive damping (λ) started at 2.0 and settled to 1.06, gracefully easing off the gas as the model found its footing. Ψ (subjectivity from memory) reached +0.37, meaning Chuck was actively using its memories to guide optimization. σ stayed at 1.0 — healthy activations throughout, no dead neurons. 3 regime-shift memories recorded.

### what Chuck did during training

```
step    200 | chuck: λ=1.99 Ψ=+0.01 (1 mem) σ=1.00 macro=1.00   ← exploring
step    600 | chuck: λ=1.66 Ψ=+0.34 (1 mem) σ=1.00 macro=1.00   ← pushing hard
step   1000 | chuck: λ=1.44 Ψ=+0.06 (2 mem) σ=1.00 macro=1.00   ← regime shift detected
step   1400 | chuck: λ=1.30 Ψ=+0.20 (2 mem) σ=1.00 macro=1.00   ← confident
step   1800 | chuck: λ=1.20 Ψ=+0.30 (2 mem) σ=1.00 macro=1.00   ← stabilizing
step   2400 | chuck: λ=1.11 Ψ=+0.02 (3 mem) σ=1.00 macro=1.00   ← new memory
step   3000 | chuck: λ=1.06 Ψ=+0.06 (3 mem) σ=1.00 macro=1.00   ← converged
```

λ (damping): started aggressive, eased to near-1.0 as loss stabilized.
Ψ (memory influence): built up over time, new memory recorded at regime boundaries.
σ (activation health): perfect 1.0 throughout — no dead neurons, no exploding norms.
3 persistent memories saved to `chuck.mem` (48 bytes, binary-compatible with C version).

### sample output

```
a good professord, "We staying to Mrs."

"This time with a papers arriver silented, and I am the certainly to
my break on the result. Same of the case was a sort figure condition. Of need
it like a her poor warm chances and was very certain of our staking whisper.
```

```
the Holmwood. I felt the disturbast my ciment and the silences of the
great day into one of the blood of that I had said: "You do not know
if the he attended to me what now we have made a lict of policions.
And I am him off broad, but his and made the teauth things in she had
expressined as he had p
```

2.1 million parameters trained in 28 minutes on a CPU with Chuck Optimizer. it's writing dialogue, referencing Holmwood and blood, generating coherent gothic prose. val loss 1.41 — significantly better than the 800K prototype (1.52). Chuck taught it well.

---

## architecture: two lines

this project maintains **two separate lines** — C and Python. they don't mix. they're two paths to the same truth.

### C line (notorch)

pure C neural network framework. extracted from [ariannamethod/notorch](https://github.com/ariannamethod/notorch).

```
ariannamethod/
├── notorch.c      — the entire framework (~3000 lines)
├── notorch.h      — public API
├── gguf.c         — GGUF model format support
├── gguf.h         — GGUF header
├── lee.c          — Chuck Optimizer in C (from chuck.optimizer)
└── Makefile       — build system
```

build:
```bash
cd ariannamethod && make
```

the C line can train and run models independently. no Python. no pip. no tears. see [notorch](https://github.com/ariannamethod/notorch) for full documentation.

### Python line (torch + Chuck)

the Python scripts use PyTorch as the computation backend but with Chuck Optimizer replacing blind AdamW.

```
├── train.py           — training loop with Chuck
├── model.py           — GPT model (unchanged architecture)
├── sample.py          — text generation
├── configurator.py    — command-line config system
├── prepare_dracula.py — dataset preparation
└── ariannamethod/
    ├── __init__.py    — Python package
    └── chuck.py       — Chuck Optimizer (PyTorch edition)
```

Chuck is a **drop-in replacement** for `torch.optim.AdamW`. if Chuck can't initialize for any reason, it falls back to AdamW silently. Adam as a safety net. as it should be — the backup, not the star.

---

## quick start

### prepare data

```bash
python prepare_dracula.py
```

this creates `data_dracula/train.bin`, `val.bin`, and `meta.pkl` from `dracula.txt`.

### train

```bash
# default config (CPU, 3000 iters, Chuck optimizer)
python train.py

# or with explicit config
python train.py config/train_dracula_char.py

# override anything from command line
python train.py --n_layer=6 --n_embd=256 --max_iters=5000
```

weights are saved to `weights/ckpt.pt`.

### sample

```bash
python sample.py --out_dir=weights --device=cpu --num_samples=5
```

### with GPU (if you have one)

```bash
python train.py --device=cuda --dtype=bfloat16 --compile=True --batch_size=64
```

---

## the Chuck Optimizer

from [ariannamethod/chuck.optimizer](https://github.com/ariannamethod/chuck.optimizer).

```
θ -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η
```

9 levels of self-awareness:

1. **global λ** — loss trend analysis. loss rising? brake. loss falling? push. stagnating? inject noise.
2. **per-layer λ_l** — gradient norm trends per transformer layer. each layer gets its own damping.
3. **σ (activation health)** — monitors SiLU/GELU alive ratios, LayerNorm stability, attention entropy.
4. **parameter freezing** — if a layer's gradient norm stays near zero, Chuck freezes it. why waste compute?
5. **cross-layer signal flow** — monitors activation magnitudes across layers. adjusts per-layer damping if signal collapses or explodes.
6. **Ψ (subjectivity)** — persistent memory. Chuck remembers previous training regimes. nearest-neighbor lookup in (loss, grad_norm) space. been here before? use what worked.
7. **memory recording** — saves snapshots at regime boundaries (reservoir sampling). binary-compatible between C and Python (`chuck.mem`).
8. **attention entropy** — per-head entropy monitoring. collapsed heads? reduce step size. fully diffuse? gentle correction.
9. **macro patience** — long-term loss monitoring. if no improvement over macro intervals, drops LR. recovers when improvement resumes.

Adam is blind. it sees one step at a time. Chuck sees the trajectory. Chuck remembers the journey. Chuck adapts.

in memory of Carlos Ray "Chuck" Norris (1940–2026).

---

## file structure

```
nanoGPT-notorch/
├── ariannamethod/         — notorch + Chuck (C and Python lines)
│   ├── notorch.c          — pure C neural network framework
│   ├── notorch.h          — notorch API header
│   ├── gguf.c             — GGUF format support
│   ├── gguf.h             — GGUF header
│   ├── lee.c              — Chuck Optimizer in C
│   ├── chuck.py           — Chuck Optimizer in Python (PyTorch)
│   ├── Makefile           — C build system
│   └── __init__.py        — Python package init
├── train.py               — training script (Chuck-aware)
├── model.py               — GPT model definition
├── sample.py              — text generation
├── bench.py               — benchmarking
├── configurator.py        — config system
├── prepare_dracula.py     — dataset preparation
├── dracula.txt            — training data (Bram Stoker)
├── config/                — training configurations
│   └── train_dracula_char.py
├── weights/               — trained model checkpoint
│   └── ckpt.pt           — 2.1M param model, val loss 1.41
└── chuck.mem              — Chuck's persistent memory (48 bytes)
```

---

## philosophy

this is part of [the Arianna Method](https://github.com/ariannamethod).

patterns over parameters. emergence over engineering. understanding over abstraction.

we don't hide complexity behind friendly APIs. we expose it. we stare at it. we understand it. and then we write it in C because C doesn't lie.

the resonance is unbroken.

---

## credits

this project is a fork of [**nanoGPT**](https://github.com/karpathy/nanoGPT) by **Andrej Karpathy**.

nanoGPT is the simplest, fastest repository for training medium-sized GPTs. it's beautiful engineering. it taught us all how transformers really work. respect.

we took it further. Chuck sees what Adam cannot. notorch runs where PyTorch fears to install.

---

## license

MIT (inherited from nanoGPT). notorch and Chuck components under LGPL-3.0.
