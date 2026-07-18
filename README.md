# TinyRecurseLab

**Train a real Tiny Recursive Model in your browser and watch a ~17k-weight network learn to solve Sudoku by thinking in loops.**

An interactive, dependency-free explainer for the *"Less is More"* recursive-reasoning idea — the family of tiny models (TRM / HRM) that beat frontier LLMs on hard puzzle benchmarks with a tiny fraction of the parameters, not by being big, but by **recursing** on a latent scratchpad and iteratively improving a draft answer.

Everything is **real computation**. There are no pre-baked animations and no network calls: a from-scratch reverse-mode **autograd engine** (~150 lines of vanilla JS) trains an actual model live on 4×4 Sudoku puzzles generated on the fly.

## The six modules

1. **The Loop** — step through the recursion on a live puzzle; watch the solution `y` sharpen while the latent scratchpad `z` churns.
2. **Train a Reasoner** — real forward/backward/Adam training with live loss + held-out accuracy curves. Tweak recursion depth `n`, supervision steps `T`, width, LR, and difficulty.
3. **Test-Time Compute** — run *more* improvement steps at inference than during training and see the accuracy-vs-compute curve, including where a tiny model starts to over-think.
4. **TRM vs HRM** — race full-backprop (TRM) against the 1-step-gradient approximation (HRM), plus a parameter calculator showing why one network beats two.
5. **Adaptive Halting (ACT)** — halt each puzzle when its answer stabilizes and see how harder puzzles consume more compute.
6. **Under the Hood** — verify the autograd engine live against finite differences, inspect the architecture, and compare to the paper's headline benchmarks.

## Scientific grounding

- Alexia Jolicoeur-Martineau. *Less is More: Recursive Reasoning with Tiny Networks* (TRM). arXiv:2510.04871, 2025.
- Wang et al. *Hierarchical Reasoning Model* (HRM). arXiv:2506.21734, 2025.

This 4×4 Sudoku sandbox is an original educational reimplementation of the *algorithm* (single recursive core, deep supervision, latent refinement, adaptive halting) scaled down to run live in a browser tab — it is not the authors' code or weights.

## Run it

Just open `index.html` — no build step, no dependencies. Deployed via Vercel (`{"cleanUrls": true}`).

## License

MIT © 2026 Surya Midde. Built autonomously as part of a daily-webapp series.
