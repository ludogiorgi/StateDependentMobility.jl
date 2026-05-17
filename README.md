# StateDependentMobility.jl

Julia code for learning and validating state-dependent mobility tensors in reduced Langevin models from simulated trajectory data.

## Main Scripts

- `2D/sim.jl`: simulate the 2D reference SDE and save trajectories.
- `2D/score.jl`: train the stationary score model for the 2D system.
- `2D/joint_score.jl`: train the joint score model for the 2D system.
- `2D/fit_dm.jl`: fit the mobility model `M(x)` from correlation operators and optionally run forward validation for the 2D system.
- `L96/sim.jl`: simulate the stochastic Lorenz--96 system and save trajectories plus diagnostics.
- `L96/score.jl`: train the stationary score model for the Lorenz--96 system and save diagnostics.
- `L96/joint_score.jl`: train the lag-conditioned joint score model for the Lorenz--96 system and save diagnostics.

## Main Configs

- `2D/sim.toml`
- `2D/score.toml`
- `2D/joint_score.toml`
- `2D/fit_dm.toml`
- `L96/sim.toml`
- `L96/score.toml`
- `L96/joint_score.toml`

## Typical Workflow

Run these commands from the repository root.

### 2D Pipeline

```bash
julia --project=. --threads 36 2D/sim.jl 2D/sim.toml
julia --project=. --threads 36 2D/score.jl 2D/score.toml
julia --project=. --threads 36 2D/joint_score.jl 2D/joint_score.toml
julia --project=. --threads 36 2D/fit_dm.jl 2D/fit_dm.toml
```

### L96 Pipeline

```bash
julia --project=. --threads 36 L96/sim.jl L96/sim.toml
julia --project=. --threads 36 L96/score.jl L96/score.toml
julia --project=. --threads 36 L96/joint_score.jl L96/joint_score.toml
```

2D run outputs are organized under `2D/runs/run_###/`.
L96 score outputs are written to `L96/outputs/score.bson` and `L96/outputs/score_diagnostics.png` by default.
L96 joint-score outputs are written to `L96/outputs/joint_score.bson` and `L96/outputs/joint_score_diagnostics.png` by default.

To inspect `L96/outputs/joint_score.bson` in Julia, first load the joint-score definitions so the saved model and config types are available:

```julia
include("L96/joint_score.jl")
state = BSON.load("L96/outputs/joint_score.bson")
```
