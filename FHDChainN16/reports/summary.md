# FHDChainN16 Phi-Only Benchmark Summary

## Scope

This benchmark uses the strong fluctuating-hydrodynamic chain from `system.txt`
with `N=16`, `L=2π`, `rho0=1`, `c_s=0.6`, `Theta=0.04`, `eta0=0.08`, and
`zeta=3`.  It stops at the constant-mobility model `M=Phi`: no conditional
score, no joint score, and no learned `M_theta` were trained or used.

## Simulation

- Dataset: `data/fhd_chain_n16.h5`
- State order: `(rho_1,...,rho_16,m_1,...,m_16)`
- Trajectories: `72`
- Saved frames per trajectory: `277987`
- Integration `dt`: `0.000625`
- Saved `save_dt`: `0.166875`
- Pilot decorrelation time used for the production schedule: `t_D = 16.7`
- Production budget from pilot: `T/t_D = 2777.779266`
- Saved resolution from pilot: `t_D/save_dt = 100.074906`
- Conservation: max mass drift `5.33e-15`, max momentum drift `5.89e-16`
- Minimum saved density: positive but underflow-scale, `1e-323`

Important caveat: the final saved-window density ACF does not decorrelate on the
configured diagnostic window. The metadata final check gives
`t_D_final_check = 166.875`, exactly the maximum plotted lag, with density ACF
still about `0.0903`. A separate extended check out to lag `6675` still had
density ACF about `0.0554`. Thus the production schedule matches the pilot
budget exactly, but the stationary density field has a much slower tail than the
pilot estimate. This is a real limitation of this N=16 strong-regime dataset.

The strict conservative Euler-Maruyama density update requested in the original
plan was also tested in a pilot-only run. It failed positivity even after
recursive substeps down to `1.5e-7`, because centered density flux can drive
near-vacuum cells negative. The saved dataset therefore uses the same
positivity-preserving exponential density update used by the existing N=8 FHD
benchmark, with exact mass renormalization.

## Stationary Score

- Code/config: `code/score.jl`, `configs/score.toml`
- Checkpoint: `models/score_sigma005.bson`
- Training: DSM only, `sigma=0.05`, no BatchNorm, GPU request `GPU:1`
- Resolved device: RTX 5070 (`nvidia-smi` index 1)
- Final DSM loss: `9.350978e-01`
- Ex-post Stein relative error: `4.312714e-02`
- Ex-post analytic safe-subset score rel.RMSE: `6.085385e-01`
- Ex-post analytic safe-subset cosine: `0.795813`

Score-only Langevin validation:

- Trajectory: `data/score_langevin.h5`
- Figure: `figures/score_langevin_validation.png`
- Mean PDF relative L2: `2.435288e-02`
- Density PDF relative L2: `3.460757e-02`
- Momentum PDF relative L2: `2.538064e-02`
- Velocity PDF relative L2: `1.856813e-02`
- Viscosity PDF relative L2: `1.885517e-02`
- Covariance relative RMSE: `1.049587e-01`

The score reproduces one-point PDFs well, but covariance and analytic-score
diagnostics are not perfect. This is consistent with the near-vacuum samples and
slow density tail.

## Phi Recovery and Phi-Only Validation

- Code/config: `code/fit_Phi.jl`, `configs/fit_Phi.toml`
- Artifacts: `models/fit_Phi_artifacts.bson`
- Metrics: `logs/fit_Phi_metrics.txt`
- Phi figure: `figures/phi_recovery.png`
- Cdot figure: `figures/phi_cdot_gfdt.png`

Data-only Phi recovery:

- `Phi` vs `<M_true>` rel.RMSE: `3.37409026e-02`
- `Phi` vs `<M_true>` correlation: `9.99439270e-01`
- Tangent eigenvalue range of `sym(Phi)`: `-1.26488838e-18` to `9.16655945e-02`

Phi-GFDT Cdot diagnostic:

- Formula used: `Cdot_mx^Phi(t) = <phi_m(x_t) s_theta(x_0)'> Phi`
- Conditional score used: none
- Relative RMSE vs data Cdot: `1.83217862e-01`
- Correlation vs data Cdot: `9.84435256e-01`

Phi forward Langevin validation:

- Trajectory: `data/phi_forward_langevin.h5`
- Statistics figure: `figures/phi_forward_stats.png`
- Cmn figure: `figures/phi_forward_cmn.png`
- Minimum tangent diffusion eigenvalue: `-1.40430040e-18`
- Forward `Cmn` Phi vs observation rel.RMSE: `4.20459019e-01`
- Forward `Cmn` Phi vs observation correlation: `9.29362705e-01`

The constant `Phi` model reproduces stationary one-point statistics well and
recovers the true mean mobility very accurately. It does not fully reproduce the
finite-lag dynamical correlations, especially in the slow density sector.

## No-Cheating Audit

Analytic score, true mobility, simulator generator formulas, and true model
internals were not used in DSM losses, `Phi` construction, `Cdot_data`, the
Phi-GFDT estimator, or forward-model fitting/tuning. Analytic score and true
mobility were used only for labeled ex-post diagnostics and validation figures.
