# FHDChainCapillaryN16 Report

## Current status

Accepted simulation and analytic constant-mobility validation are complete.
This system is a periodic one-dimensional fluctuating hydrodynamic chain with
`N=16`, conserved total mass, conserved total momentum, and a capillary
gradient free-energy term. No neural network was trained for the analytic Phi
validation.

## Scripts and commands

Simulation:

```bash
julia --project=. --threads 36 FHDChainCapillaryN16/code/sim.jl FHDChainCapillaryN16/configs/sim.toml 2>&1 | tee FHDChainCapillaryN16/logs/sim.log
```

Analytic score / true mobility / Phi-forward validation:

```bash
julia --project=. FHDChainCapillaryN16/code/fit_Phi_analytic.jl FHDChainCapillaryN16/configs/sim.toml 2>&1 | tee FHDChainCapillaryN16/logs/fit_Phi_analytic.log
```

The analytic validation script is CPU-only. It imports no Flux/CUDA code, trains
no score model, and uses only the analytic score and analytic true mobility.

## Final parameters

The accepted stable regime is:

- `N=16`, `L=2pi`, `rho0=1.0`
- `c_s=2.0`, `Theta=0.025`
- `eta0=0.08`, `zeta=0.5`
- `kappa=0.02`
- production `dt=2.5e-4`
- `72` trajectories, `36` Julia threads
- save resolution `save_dt=0.242`, with `t_D/save_dt=100`

The older strong-compressibility FHD parameters were not retained because they
gave rare near-vacuum and nonfinite events after correcting the LLNS noise
normalization. The accepted parameters are weakly compressible and maintain
strictly positive density over the production run.

## Simulation artifacts

- Dataset: `FHDChainCapillaryN16/data/fhd_chain_capillary_n16.h5`
- Summary figure: `FHDChainCapillaryN16/figures/sim_summary.png`
- Dynamics figure: `FHDChainCapillaryN16/figures/sim_dynamics.png`
- Trajectory figure: `FHDChainCapillaryN16/figures/sim_trajectories.png`
- Simulation log: `FHDChainCapillaryN16/logs/sim.log`

Dataset layout:

- `/trajectories/time`
- `/trajectories/states`, shape `(277779, 16, 2, 72)`
- `/trajectories/states_flat`, flattened order `[rho_1..rho_N, m_1..m_N]`
- metadata includes `dt`, `save_dt`, `t_D`, `T`, conservation diagnostics,
  state ordering, channel names, parameters, score formula, and capillary checks.

## Accepted simulation checks

From the final run:

- pilot `t_D = 24.2`
- production `T = 67222.276`
- `T/t_D = 2777.780`, matching the previous `FHDChainN16` sample budget
- `t_D/save_dt = 100.0`
- final-window decorrelation check from saved data: `t_D = 26.136`
- max mass drift: `1.498028e-6` from Float32 stored snapshots
- max total momentum magnitude: `2.311411e-7`
- minimum density over saved production data: `0.298366`
- post-burnin density range: `[0.298366, 1.668193]`
- mean edge viscosity: `0.079931`, std: `0.003314`
- capillary force sum: `-2.081668e-17`
- score kappa-term max absolute error: `1.401657e-15`
- constrained score finite-difference relative error: `5.355801e-8`

## Analytic score

The stationary density is proportional to
`exp[-F_kappa(rho,m)/Theta]` on the subspace of fixed total mass and fixed total
momentum, where

```text
F_kappa = dx * sum_i [
    c_s^2 * (rho_i log(rho_i/rho0) - rho_i + rho0)
    + m_i^2/(2 rho_i)
] + kappa/(2 dx) * sum_i (rho_{i+1} - rho_i)^2 .
```

With periodic indexing and
`Delta_h rho_i = (rho_{i+1} - 2 rho_i + rho_{i-1})/dx^2`, the unconstrained
analytic score is

```text
s_rho_i = -dx/Theta * [
    c_s^2 log(rho_i/rho0) - 0.5 (m_i/rho_i)^2 - kappa Delta_h rho_i
]
s_m_i = -dx/Theta * (m_i/rho_i)
```

The implemented score then projects out the conserved zero modes:

```text
s_rho <- s_rho - mean(s_rho)
s_m   <- s_m   - mean(s_m)
```

## Analytic true mobility

The paper convention used here is

```text
dx = [M(x) s(x) + div_x M(x)] dt + sqrt(2) Sigma(x) dW,
Sigma Sigma' = sym(M).
```

For this model, `div_x M = 0` in the implemented finite-volume coordinates. The
true mobility is the sum of a state-dependent viscous symmetric block and the
finite-volume reversible hydrodynamic operator. For a covector
`v=(a,b)=(v_rho,v_m)`, define edge viscosities
`eta_i = eta0 * (((rho_i + rho_{i+1})/2)/rho0)^zeta`. The symmetric block acts
only on momentum:

```text
(D_mm b)_i = Theta/dx^3 * [
    (eta_i + eta_{i-1}) b_i - eta_i b_{i+1} - eta_{i-1} b_{i-1}
].
```

The reversible operator is represented through its transpose action:

```text
(L_h v)_rho_i =
    -[(rho_i b_i + rho_{i+1} b_{i+1})
      -(rho_{i-1} b_{i-1} + rho_i b_i)]/(2 dx)

(L_h v)_m_i =
    -rho_i (a_{i+1}-a_{i-1})/(2 dx)
    -m_i (b_{i+1}-b_{i-1})/(2 dx)
    -[0.5(m_i b_i + m_{i+1} b_{i+1})
      -0.5(m_{i-1} b_{i-1} + m_i b_i)]/dx .
```

The matrix rows are built from

```text
M(x)' v = D(x) v + (Theta/dx) L_h(x) v .
```

Equivalently, using the skewness of the reversible operator,
`M(x) = D(x) - (Theta/dx)L_h(x)`. The capillary modification enters through the
free energy and therefore through `s_rho`; applying the same reversible operator
to the capillary score gives the conservative Korteweg force.

The constant baseline is

```text
Phi = <M_true(x)>
```

estimated directly from 80,000 post-burnin observed states.

## Analytic Phi-forward artifacts

- Script: `FHDChainCapillaryN16/code/fit_Phi_analytic.jl`
- Forward HDF5: `FHDChainCapillaryN16/data/phi_analytic_forward_langevin.h5`
- Statistics figure: `FHDChainCapillaryN16/figures/phi_analytic_forward_stats.png`
- Dynamic correlation figure: `FHDChainCapillaryN16/figures/phi_analytic_forward_cmn.png`
- Metrics: `FHDChainCapillaryN16/logs/fit_Phi_analytic_metrics.txt`

The accepted Phi-forward run used:

- `Phi` samples: `80000`
- `dt = 2.5e-3`
- total forward time: `200 t_D = 4840`
- burn-in: `20 t_D = 484`
- saved forward states shape: `(17963, 16, 2, 72)`
- minimum symmetric eigenvalue of `Phi`: `-5.55e-17` numerical zero

Accepted analytic Phi-forward metrics:

- `rho_pdf_rel_l2 = 3.1923e-2`
- `m_pdf_rel_l2 = 2.9555e-2`
- `u_pdf_rel_l2 = 3.3055e-2`
- `eta_pdf_rel_l2 = 3.7023e-2`
- `covariance_corr = 0.999127`
- `covariance_rel_rmse = 7.1258e-2`
- `rho_acf_rel_l2 = 1.0967e-1`
- `m_acf_rel_l2 = 9.7227e-2`
- selected `C_mn(t)` rel.RMSE: `1.3784e-1`
- selected `C_mn(t)` corr: `0.991398`

The one-dimensional PDFs now overlay very closely. The remaining finite-lag
correlation error is expected because the forward model replaces the full
state-dependent `M(x)` by the constant baseline `<M_true>`.

## Failed attempts and lessons

- Initial capillary runs copied the original `FHDChainN16` strong-compressibility
  parameter scale. With the corrected LLNS noise amplitude, those runs reached
  rare near-vacuum/nonfinite states and were rejected.
- Increasing `kappa` alone did not fix the instability in that strong regime.
  The stable accepted run instead uses a weakly compressible setting with
  `c_s=2.0`, `Theta=0.025`, and `eta0=0.08`.
- A first analytic Phi-forward run with `dt=1e-2` produced visibly too-broad
  PDFs: PDF relative L2 errors were about `0.13-0.18`, and covariance relative
  RMSE was `0.466`. That was a Langevin discretization artifact, not an analytic
  score/Phi issue. Reducing the forward step to `2.5e-3` fixed the invariant
  density agreement.
- Full `states_flat` was not materialized in memory during simulation. It is
  streamed to HDF5, while `/trajectories/states` is stored as Float32. This was
  necessary to keep the large production run memory-safe.
- Long production loops print heartbeats; otherwise long no-output Julia runs
  are too hard to monitor.

## No-cheating audit

Simulation-only data generation used the physical SDE, analytic score only for
metadata and ex-post capillary verification. No training loss was minimized.

The analytic Phi-forward task intentionally used analytic quantities because the
user explicitly requested no NN training and direct use of known analytic score,
`M_true(x)`, and `Phi=<M_true>`. No learned model, DSM label, conditional score,
or target-fitting loss was used. The analytic comparison is labeled as such and
should not be confused with a data-only Step 2 pipeline.

## Next action

If a later agent starts data-only Step 2, do not reuse this analytic score or
analytic `M_true` in any minimized training target. They are allowed only for
ex-post diagnostics or for this explicitly analytic baseline. For a true
data-only benchmark step, train DSM score and estimate Phi from data following
the repository protocol.
