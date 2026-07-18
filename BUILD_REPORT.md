# BUILD REPORT — TinyRecurseLab (2026-07-17)

**Live:** https://tinyrecurselab.vercel.app
**Repo:** https://github.com/middesurya/daily-webapp-2026-07-17-tinyrecurselab

## What & why
An interactive lab on **recursive reasoning** — the "tiny model that out-reasons frontier LLMs" story that was heavily discussed on X / in Bay-Area AI circles through late 2025 into 2026. Grounded in two real papers:
- **TRM** — Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*, arXiv:2510.04871 (Samsung SAIL Montréal). 7M params, single 2-layer net, recurses on latent scratchpad `z` + solution `y`; deep supervision; adaptive halting. 87.4% Sudoku-Extreme, 45% ARC-AGI-1.
- **HRM** — Wang et al., *Hierarchical Reasoning Model*, arXiv:2506.21734. Two coupled recurrent modules, 1-step gradient approximation, ACT via Q-learning, 27M params.

Chosen because it is a genuine breakthrough, narrow niche, and **NOT** already in the repo series (grokking `groklab` and energy-based transformers `ebtlab` are adjacent but distinct).

## Novelty check
Cross-checked all ~90 existing `middesurya` repos + local folders + CLAUDE.md covered-topics list. Recursive reasoning / TRM / HRM / latent-refinement-on-puzzles is new.

## Real computation (not fake)
- Wrote a from-scratch **reverse-mode tensor autograd engine** (matmul / addBias / add / relu / tanh / concat / softmax-CE) with per-op backward closures + a tape.
- Trains a genuine TRM-style recursive net (~17k weights) on **4×4 Sudoku** generated live with a backtracking generator that guarantees **unique solutions**.
- 6 modules: The Loop · Train a Reasoner · Test-Time Compute · TRM-vs-HRM ablation · Adaptive Halting · Under-the-Hood.

## Verification performed
- **Gradient check** (analytic autodiff vs finite differences, isolated single supervision step): max relative error **6.5e-7** (also runnable live in module 6).
- **Sudoku generator**: 0/200 invalid solutions, 0/200 non-unique puzzles.
- **Training convergence** (Node): loss 1.5→0.001, held-out **44% exact-match / 93% cell accuracy** at 400 steps on a 17k-param model.
- **Headless DOM smoke test**: stubbed canvas/DOM, executed init + every module handler → all pass, no runtime errors.
- **Live check**: public URL returns 200 with real page content; zero SSO-wall markers.

## Notable lessons / pitfalls hit
1. **Finite-diff gradient check must isolate one supervision step.** TRM/HRM deliberately *truncate* gradients between supervision steps (detached carry state), so finite-differencing the full multi-step loss will NOT match the truncated analytic gradient — that's correct algorithm behavior, not a bug. Use T=1 to validate the autograd itself.
2. **Canvas HiDPI `setup()` bug:** re-reading `cv.width` after setting it to `w*dpr` returns the *scaled* value on subsequent redraws → everything draws at 2× offset on Retina. Cache logical `_w/_h` and always return those. (dpr=1 test envs hide this.)
3. **File truncation on Edit — now seen on the session outputs fs too, not just OneDrive.** Two separate Edit-tool edits silently truncated the tail of `core.js` and `ui.js` mid-file. Symptom: `node --check` "Unexpected end of input" / "Invalid or unexpected token" on the last line. Fix pattern that worked: `node --check` each source file after edits, and repair the tail by stripping the partial last line and re-appending the known-good closer via bash heredoc. Always re-validate the *assembled* index.html's script blocks with `new Function()` before shipping.

## Stack
Single-file `index.html` (assembled from `core.js` + `ui.js` + `head.html`), vanilla JS + Canvas, no deps, no fetch. `vercel.json = {"cleanUrls": true}`. MIT.
