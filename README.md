# StateDependentMobility.jl

Julia code for learning and validating state-dependent mobility tensors in reduced Langevin models from simulated trajectory data.

## Main Scripts

- `sim.jl`: simulate the reference SDE and save trajectories.
- `score.jl`: train the stationary score model.
- `joint_score.jl`: train the joint score model.
- `fit_dm.jl`: fit the mobility model `M(x)` from correlation operators and optionally run forward validation.

## Main Configs

- `sim.toml`
- `score.toml`
- `joint_score.toml`
- `fit_dm.toml`

## Typical Workflow

```bash
julia --project=. --threads 36 sim.jl sim.toml
julia --project=. --threads 36 score.jl score.toml
julia --project=. --threads 36 joint_score.jl joint_score.toml
julia --project=. --threads 36 fit_dm.jl fit_dm.toml
```

Run outputs are organized under `runs/run_###/`.
