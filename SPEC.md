# TinyRecurseLab — SPEC

**One-liner:** An interactive, dependency-free lab where you *train a real Tiny Recursive Model in your browser* and watch a small network learn to solve Sudoku by thinking in loops — the "Less is More" recursive-reasoning idea (TRM, arXiv:2510.04871) that beat billion-parameter LLMs on hard puzzles.

## Scientific grounding (real, cited)
- **TRM** — "Less is More: Recursive Reasoning with Tiny Networks", Alexia Jolicoeur-Martineau (Samsung SAIL Montréal), arXiv:2510.04871 (6 Oct 2025). Single 2-layer net, ~7M params, recurses on a latent scratchpad `z` and current solution `y`; deep supervision; adaptive halting. 87.4% Sudoku-Extreme, 85.3% Maze-Hard, 45% ARC-AGI-1, 8% ARC-AGI-2.
- **HRM** — "Hierarchical Reasoning Model", arXiv:2506.21734. Two coupled recurrent modules (slow H / fast L), 1-step gradient approximation, ACT halting via Q-learning, 27M params, ~1000 examples, 55% Sudoku-Extreme.

## Real in-browser computation (no fake animations)
- From-scratch reverse-mode tensor autograd engine (matmul, add, tanh, relu, softmax cross-entropy), verified live vs finite differences.
- Trains a genuine TRM-style recursive net on 4×4 Sudoku generated on the fly. Deep supervision over T improvement steps, each doing n latent recursions.
- Exact-match accuracy, cross-entropy loss, live curves — computed, not scripted.

## Modules (6)
1. The Loop — refine y-grid + z-scratchpad across recursion steps on a live puzzle.
2. Train a Tiny Reasoner — live training on 4×4 Sudoku; controls n, T, hidden, LR.
3. Test-Time Compute — sweep inference recursion steps → accuracy-vs-compute curve.
4. TRM vs HRM — full-backprop vs 1-step-gradient ablation + param calculator.
5. Adaptive Halting (ACT) — learned halt head; steps-used vs difficulty.
6. Under the Hood — live autograd gradient check + architecture + paper numbers.

## Tech
Single-file index.html, vanilla JS + Canvas, no deps. vercel.json {"cleanUrls": true}. MIT.
