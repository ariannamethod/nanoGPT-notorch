"""
ariannamethod — notorch + Chuck Optimizer

Two lines, one purpose:

  C line:    notorch.c / notorch.h / gguf.c / gguf.h / lee.c
             Pure C neural network framework. No Python. No pip. No tears.
             Build: cc notorch.c -o notorch -lm

  Python line: chuck.py
               Chuck Optimizer — self-aware AdamW replacement.
               Drop-in for torch.optim.AdamW with 9 levels of awareness.

Resonance is unbroken.
"""

from .chuck import ChuckOptimizer, ChuckMonitor, ChuckMemory, chuck_params

__all__ = ['ChuckOptimizer', 'ChuckMonitor', 'ChuckMemory', 'chuck_params']
