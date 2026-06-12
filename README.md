# StateDependentMobility.jl

Julia code for the benchmark pipelines used in:

Ludovico T. Giorgini, "Conditional Score-Based Modeling of Effective Langevin
Dynamics", arXiv:2604.23952, 2026.

The public repository contains the lightweight source code and configuration
files needed to reproduce the paper benchmarks. Large generated artifacts
(`*.h5`, `*.bson`, figures, logs, and trained checkpoints) are intentionally not
tracked.

## Contents

- `2D/`: two-dimensional reference benchmark.
- `SoftSpinLLGChain/`: soft-spin chain benchmark and manuscript-result configs.
- `ScoreUNet1D.jl/`: pinned submodule providing the periodic 1D U-Net backbone.

## Setup

Clone with the submodule:

```bash
git clone --recurse-submodules https://github.com/ludogiorgi/StateDependentMobility.jl.git
cd StateDependentMobility.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

If the repository was already cloned:

```bash
git submodule update --init --recursive
```

## Workflows

Run commands from the repository root.

```bash
julia --project=. --threads auto 2D/sim.jl 2D/sim.toml
julia --project=. --threads auto 2D/score.jl 2D/score.toml
julia --project=. --threads auto 2D/joint_score.jl 2D/joint_score.toml
julia --project=. --threads auto 2D/fit_dm.jl 2D/fit_dm.toml
```

For the soft-spin benchmark, the core entry points are:

```bash
julia --project=. --threads auto SoftSpinLLGChain/code/sim.jl SoftSpinLLGChain/configs/sim.toml
julia --project=. --threads auto SoftSpinLLGChain/code/score.jl SoftSpinLLGChain/configs/score.toml
julia --project=. --threads auto SoftSpinLLGChain/code/fit_Phi.jl SoftSpinLLGChain/configs/fit_Phi.toml
julia --project=. --threads auto SoftSpinLLGChain/code/cond_score.jl SoftSpinLLGChain/configs/cond_score.toml
julia --project=. --threads auto SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/mobility11_analytic_fullcache_nosignal_gpu0.toml
```

The soft-spin manuscript result manifests are in:

```text
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/
```

The primary paper branch is `data_epoch240_best_corr`. The comparison branches
are `data_stein270_best_corr` and `phys_score_11obs_cond`.
