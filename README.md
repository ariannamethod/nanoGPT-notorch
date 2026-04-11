```
   ███╗   ██╗ █████╗ ███╗   ██╗ ██████╗  ██████╗ ██████╗ ████████╗
   ████╗  ██║██╔══██╗████╗  ██║██╔═══██╗██╔════╝ ██╔══██╗╚══██╔══╝
   ██╔██╗ ██║███████║██╔██╗ ██║██║   ██║██║  ███╗██████╔╝   ██║
   ██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║██║   ██║██╔═══╝    ██║
   ██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝╚██████╔╝██║        ██║
   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝        ╚═╝
                    ─ notorch edition ─
```

# nanoGPT — liberated from PyTorch

forked from [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT). PyTorch removed. Adam replaced by [Chuck](https://github.com/ariannamethod/chuck.optimizer). trained on Dracula instead of Shakespeare. because enough fairy tales.

## model

| | |
|---|---|
| params | **10.2M** |
| dim | 320 |
| layers | 8 |
| heads | 8 |
| ctx | 256 |
| train loss | **0.79** |
| val loss | **1.34** |
| time | 5.6 hours (8 GB Mac) |
| PyTorch | **zero** |

## generation (10.2M, trained on Dracula)

```
"My dear friend," said the Count, "He got out to be one paper with me. I
know the course the Count made as me do and sitten of them, it will
no dor be with my warn. We were only trouble to the matter of me. Mina,
and say back to me:-

The blood was great. I must hell twas began that he rusing by more
cheered. He know he began by the ship. Here spoke a fig on the search
three only and groat took back of the
```

the Count speaks. Mina appears. blood flows. ships sail. 10.2 million parameters trained in pure C on an 8 GB Mac with zero Python, zero PyTorch, zero pip install.

## architecture

```
dim=320, L=8, H=8, HD=40, FFN=896, V=94, CTX=256
MHA + RoPE + SwiGLU + RMSNorm
Chuck optimizer (cosine LR 3e-4, warmup 1200 steps)
char-level, 94 unique characters from Dracula
```

## build & run

```bash
# compile (macOS)
cc -O2 -Iariannamethod train_nanogpt.c ariannamethod/notorch.c -o train \
   -lm -framework Accelerate -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK

# compile (Linux)
cc -O2 -Iariannamethod train_nanogpt.c ariannamethod/notorch.c -o train -lm -lopenblas -DUSE_BLAS

# train
./train 12000 3e-4

# resume from checkpoint
./train --resume 20000 3e-4
```

compiles in <1 second. binary ~100 KB. no virtual environment. no conda. no pip. no Docker.

## files

```
nanoGPT-notorch/
├── train_nanogpt.c         # pure C training (notorch) — 10.2M on Dracula
├── nanogpt_dracula_f16.bin # FP16 weights (19.5 MB)
├── dracula.txt             # dataset (843 KB, Bram Stoker)
├── ariannamethod/
│   ├── notorch.c + .h      # the engine
│   ├── chuck.py            # Chuck optimizer (Python line)
│   └── Makefile
├── model.py                # Python line (GPT + Chuck, no raw PyTorch)
├── train.py                # Python training
├── sample.py               # Python generation
└── config/                 # training configs
```

## the Chuck optimizer

```
θ -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η
```

9 levels of self-awareness. loss-adaptive damping (λ), per-layer gradient monitoring (λ_l), activation health (σ), persistent memory (Ψ), parameter freezing, attention entropy. Adam is blind. Chuck sees. Chuck remembers.

in memory of Carlos Ray "Chuck" Norris (1940–2026).

## part of the [Arianna Method](https://github.com/theariannamethod) ecosystem

engine: [notorch](https://github.com/iamolegataeff/notorch). optimizer: [chuck](https://github.com/ariannamethod/chuck.optimizer).

LGPL-3.0
