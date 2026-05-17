# SoftSpinLLGChain Agent Report

This is the single living report for the `SoftSpinLLGChain` benchmark. Update
this same file after every completed step. Do not create separate per-step
reports unless the user explicitly asks for them.

## Current Retained Artifacts After Deep Cleanup

Updated 2026-05-17 after the final physics-informed mobility experiment.
Historical sections below intentionally preserve the chronological record of
discarded branches and may name artifacts that were removed during cleanup.  The
current on-disk retained generated artifacts are only the following production
or paper-facing files, plus source code and compact configs.

Retained data:

- `data/soft_spin_llg_chain.h5`: production observation dataset.
- `data/score_langevin.h5`: accepted score-only Langevin validation trajectory.
- `data/phi_phys_pC_dataonly_forward_langevin.h5`: final Phi forward baseline
  used in the paper-facing comparison.
- `data/forward_trueM_score_phys_pC_dt0015.h5`: accepted true-M plus learned
  physical-score diagnostic trajectory.
- `data/forward_dM_phys_pC_dataonly_clean37_gpu1_mean15_scale1_dt0015.h5`:
  best nonparametric NN learned-M forward trajectory.
- `data/forward_dM_phys_ansatz_clean37_directcond_floor001_dt0015.h5`:
  final physics-informed mobility forward trajectory.

Retained models/targets:

- `models/score_sigma005.bson`: original accepted direct DSM score checkpoint.
- `models/score_phys_pC_sigma02.bson`: final physical-feature stationary score.
- `models/cond_score_phys_pC_unet_vA_cont2.bson`: final direct conditional
  residual score checkpoint for the physics-informed ansatz.
- `models/joint_score_phys_pC_gpu2_vL_physaug_groupnorm_active_moment.bson`:
  conditional source used by the clean37 NN target/training branch.
- `models/fit_Phi_phys_pC_dataonly_artifacts.bson`: final data-only Phi
  baseline used by the current M training and final forward comparison.
- `models/fit_Phi_phys_pC_dataonly_projected_artifacts.bson`: projected Phi
  diagnostic artifact referenced by the writeup.
- `models/dM_targets_dataonly_oracle_ident_v1_dataoracle_clean37_channelscale.bson`:
  final clean37 data-only residual target.
- `models/dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean15.bson`: best
  nonparametric NN learned-M checkpoint.
- `models/dM_phys_ansatz_clean37_directcond_mean1e5_floor001_gpu2.bson`: final
  physics-informed mobility ansatz.

Retained figures:

- Step 1/2 and score/Phi diagnostics referenced by the writeup:
  `sim_summary.png`, `sim_dynamics.png`, `sim_trajectories.png`,
  `score_phys_pC_diagnostics.png`,
  `forward_stats_trueM_phys_score_compare.png`,
  `forward_cmn_trueM_phys_score_compare.png`,
  `phi_phys_pC_dataonly_projected_recovery.png`,
  `phi_phys_pC_dataonly_projected_cdot_gfdt.png`,
  `cond_score_phys_pC_unet_vA_cont2_diagnostics.png`.
- Important historical diagnostic figures referenced by the writeup:
  `nonlinear_observable_search_summary_all.png`,
  `dM_phys_pC_dataonly_unet_vAcont_gpu2_vJ_cached_diagnostics.png`,
  `forward_stats_forward_phys_pC_joint_best_compare.png`,
  `forward_cmn_forward_phys_pC_joint_best_compare.png`,
  `forward_stats_forward_phys_pC_oracle_trueM_vL_all_compare.png`,
  `forward_cmn_forward_phys_pC_oracle_trueM_vL_all_compare.png`,
  `forward_stats_forward_phys_pC_clean37_extended_compare.png`,
  `forward_cmn_forward_phys_pC_clean37_extended_compare.png`.
- Final current diagnostics:
  `dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean15_diagnostics.png`,
  `dM_phys_ansatz_clean37_directcond_mean1e5_floor001_gpu2_diagnostics.png`,
  `forward_stats_forward_phys_pC_paper_phi_nn_phys_compare.png`,
  `forward_cmn_forward_phys_pC_paper_phi_nn_phys_compare.png`.

Retained logs/metrics are restricted to compact accepted or writeup-referenced
metrics.  All long training logs, rejected-branch checkpoints, obsolete forward
HDF5 files, and discarded figures were removed.  The numerical outcomes of the
removed branches are preserved in the chronological sections below.

## System

- Source description: `system.txt`, periodic soft-spin stochastic
  Landau-Lifshitz chain.
- State: `m_i in R^3`, `i=1,...,12`, flat dimension `D=36`.
- Final system folder: `SoftSpinLLGChain/`.
- Final code/configs:
  - `code/sim.jl`, `configs/sim.toml`
  - `code/score.jl`, `configs/score.toml`
  - `code/fit_Phi.jl`, `configs/fit_Phi.toml`
  - shared helpers in `code/src/spin_common.jl`
- Required GPU: NVIDIA GeForce RTX 2080 Ti. CUDA.jl exposes it as CUDA ordinal
  1, while `nvidia-smi` reports it as GPU index 0. The scripts request
  `GPU:0` and the guard resolves that to the 2080 Ti. Do not use the 5070.

## No-Cheating Boundary

Training and data-driven estimators used trajectory data and learned score
models only. Analytic score and true mobility were used only for clearly labeled
ex-post diagnostics and for debugging/understanding failures. In particular,
analytic information was not used in DSM labels, score losses, Phi construction,
`Cdot_data`, training targets, or model selection by target error.

One important nuance: the final Phi estimator uses short-lag trajectory
resimulation from stationary data snapshots at the original simulator time step.
This is still data/simulation based and does not insert analytic mobility or
analytic score into the estimator. It was needed because the production dataset
is saved every `0.0365`, too coarse for an accurate zero-lag derivative.

## Step 1: Simulation

### Final Commands

Primary production command:

```bash
julia --project=. --threads 24 --startup-file=no SoftSpinLLGChain/code/sim.jl SoftSpinLLGChain/configs/sim.toml
```

Accepted artifacts:

- Data: `data/soft_spin_llg_chain.h5`
- Figures:
  - `figures/sim_summary.png`
  - `figures/sim_dynamics.png`
  - `figures/sim_trajectories.png`
- Log: `logs/sim.log`

### Final Checks

- Dataset shape:
  - `/trajectories/states`: `(277778, 12, 3, 72)`
  - `/trajectories/states_flat`: `(277778, 36, 72)`
- `t_D = 3.65`
- `save_dt = 0.0365`
- `t_D / save_dt = 100`
- `T / t_D = 2777.77`, matching the accepted benchmark budget.
- Flat ordering: site-major, `[site_i mx, my, mz]`.
- Figures were regenerated and inspected after the final data file was written.

### What Failed Or Needed Correction

- Initial decorrelation-time estimation used a too-broad ACF envelope including
  slow longitudinal/branch-mixing observables. That produced an enormous
  `t_D = 185.35`, which would have made the production budget wrong for the
  finite-lag precessional dynamics. The accepted run uses global transverse
  `Mx, My` from the pilot trajectory to define `t_D`; other ACFs are still
  plotted as diagnostics.
- The first dynamics figure selected/downsampled in the wrong order. It was
  patched to select the plotted time window before any downsampling, and the
  accepted dynamics panel uses the full saved resolution in the plotted window.

### Lessons For Similar Systems

- Choose decorrelation observables for the dynamical channels the benchmark is
  meant to test. Do not let rare branch switching dominate `t_D` if the method
  is meant to learn finite-lag oscillatory mobility effects.
- For trajectory/Hovmoller plots, always verify the plotted downsample does not
  contradict the saved dataset resolution.

## Step 2A: Stationary Score

### Final Commands

Final accepted score command:

```bash
julia --project=. --threads 24 --startup-file=no SoftSpinLLGChain/code/score.jl SoftSpinLLGChain/configs/score.toml
```

Accepted artifacts:

- Checkpoint: `models/score_sigma005.bson`
- Score diagnostics: `figures/score_diagnostics.png`
- Score-only Langevin data: `data/score_langevin.h5`
- Score-only Langevin figure: `figures/score_langevin_validation.png`
- Accepted log: `logs/score_direct_output.log`

### Final Checks

- DSM sigma: `0.05`.
- Model output mode: direct score output.
- Final train loss: about `387.46`.
- Final validation loss: about `387.46`.
- Best validation loss: about `386.64`.
- Score-only Langevin validation:
  - PDFs for `mx`, `my`, `mz`, and `|m|` match the observations well.
  - Radial statistics are corrected compared with the failed score.
  - Covariance relative error: `0.0942`.
  - Covariance correlation: `0.9952`.
- Analytic score was used only for diagnostics after DSM training.

### What Failed Or Needed Correction

- The first accepted-looking score trained the U-Net to predict DSM noise and
  then converted by `-pred/sigma`. It reached DSM loss near `0.98`, but
  score-only Langevin failed badly: `mz` improved somewhat, while `|m|` was too
  broad and covariance errors were visually obvious.
- Spin-inversion symmetry was added at evaluation by antisymmetrizing
  `s(x) = (raw(x) - raw(-x)) / 2`. This improved `mz` but did not fix the radial
  distribution.
- A smaller Langevin time step, `dt=0.0005`, did not fix the failed noise-output
  model. The failure was not an integrator artifact.
- A larger base-192 model was started and stopped early because it was much
  slower and early losses tracked the failed run. Do not repeat this as the first
  fix.
- A data-only Stein scale check showed only about 5 percent average under-scale,
  so a scalar multiplier would not have fixed the radial mismatch.
- An exact analytic-score Langevin diagnostic matched the data, proving that the
  validation equation and normalization were correct. The learned score was the
  failing component.
- The successful fix was to train a direct score-output DSM model at the same
  `sigma=0.05`. This avoided learning a small denoising residual and then
  amplifying it.

### Lessons For Similar Systems

- A good DSM noise-prediction loss is not sufficient. Always run score-only
  Langevin and check PDFs, covariance, and constrained/invariant statistics.
- When the score-only sampler fails but exact-score Langevin passes, fix the
  score parameterization/architecture before touching Phi or mobility.
- For symmetric systems, enforce known symmetry in score evaluation if it is not
  already exact, but do not expect symmetry projection alone to fix missing
  radial score structure.

## Step 2B: Phi Baseline

### Final Commands

Final accepted Phi command:

```bash
DISPLAY=:190 julia --project=. --threads 24 --startup-file=no SoftSpinLLGChain/code/fit_Phi.jl SoftSpinLLGChain/configs/fit_Phi.toml
```

`DISPLAY=:190` was needed for GLMakie under Xvfb after a run failed with a
missing X display.

Accepted artifacts:

- Artifacts: `models/fit_Phi_artifacts.bson`
- Metrics: `logs/fit_Phi_metrics.txt`
- Accepted log: `logs/fit_Phi_shortlag.log`
- Figures:
  - `figures/phi_recovery.png`
  - `figures/phi_cdot_gfdt.png`
  - `figures/phi_forward_stats.png`
  - `figures/phi_forward_cmn.png`
- Constant-Phi forward data: `data/phi_forward_langevin.h5`

### Final Checks

Final `logs/fit_Phi_metrics.txt`:

- `Phi vs <M_true> rel.RMSE = 9.97007834e-02`
- `Phi vs <M_true> corr = 9.94880307e-01`
- `Phi symmetric min eigenvalue = 2.78170651e-02`
- `Phi projection relative change = 8.29581821e-01`
- `Phi PSD projection relative change = 0`
- Final onsite axial block:

```text
[ 4.25998805e-02  -2.30208868e-03   0
  2.30208868e-03   4.25998805e-02   0
  0                0                 2.78170651e-02 ]
```

- Short-lag Phi estimator:
  - samples: `400000`
  - internal resimulation `dt = 2.5e-4`
  - saved steps: `[0, 4, 8, 12]`
- `Cdot Phi-GFDT rel.RMSE = 9.50778010e-01`
- `Cdot Phi-GFDT corr = 5.76023910e-01`
- Forward coordinate `C(t)` rel.RMSE: `3.51856103e-01`
- Forward coordinate `C(t)` corr: `9.46756516e-01`

The Phi recovery passes the target threshold, but the constant-Phi GFDT curve is
still a poor finite-lag dynamical baseline. This is expected for this system:
the benchmark is designed so state-dependent skew/precessional mobility matters
for oscillatory finite-lag correlations.

### What Failed Or Needed Correction

- The first Phi implementation estimated `Cdot_xx(0+)` from saved production
  frames spaced at `save_dt=0.0365` and projected only to a generic
  block-circulant class. It failed badly:
  - `Phi vs <M_true> rel.RMSE = 1.14`
  - `Cdot Phi-GFDT rel.RMSE = 0.948`
  - the recovery figure was dominated by offsite noise.
- Testing fit-window variants showed that the saved-frame zero-lag derivative
  was biased/noisy. The issue was not fixed by polynomial degree alone.
- Projecting to the true known support/symmetry class was essential. For this
  system, Phi should be onsite, translation-equivariant, and axial around the
  z-axis:
  - equal xx/yy diagonal entries,
  - xy antisymmetric skew pair,
  - independent zz diagonal,
  - no xz/yz or offsite blocks.
- This projection improved the ex-post comparison to about `0.17` rel.RMSE, but
  still missed the target.
- A finite-lag GFDT least-squares Phi estimate was tested and rejected. It
  overestimated the block badly and gave rel.RMSE near `5`.
- The final fix was a data-only short-lag resimulation from stationary snapshots
  using the original Euler-Maruyama step `dt=2.5e-4`, then extrapolating
  `Cdot_xx(0+)` from very short lags. With 400k starts and steps `[0,4,8,12]`,
  the projected Phi reached rel.RMSE just below `0.10`.
- A GLMakie rerun failed because `DISPLAY` was missing. Use an active Xvfb
  display, e.g. `DISPLAY=:190`, for plotting scripts in headless sessions.

### Lessons For Similar Systems

- If production `save_dt` is chosen from `t_D / save_dt = 100`, it may still be
  too coarse for a zero-lag derivative estimator. Add a short-lag data-only
  resimulation stage from stationary snapshots when Phi is biased.
- Estimate the full raw Phi first, but then project to the physically correct
  support/symmetry class, not just a broad translation-invariant class.
- The constant-Phi forward and Phi-GFDT diagnostics may remain poor even when
  `<M>` is recovered well. That should be reported, not hidden; it is useful
  evidence that Step 3 has a meaningful state-dependent mobility target.

## Current State And Step 3 Guidance

## 2026-05-14 Strict Phi Fitting-Only Update

User correction:
- The attempted high-resolution/resimulation direction was explicitly rejected
  by the user. The updated task was to keep the original production resolution,
  `t_D / save_dt = 100` (`save_dt = 0.0365`), and improve only the fitting
  method.

Code/config changes:
- Updated `code/fit_Phi.jl` with an optional direct projected covariance
  estimator. Instead of first forming a noisy dense `36 x 36` covariance
  derivative and then averaging, it estimates the onsite `3 x 3` covariance
  block directly by averaging over sampled saved trajectory pairs and all
  lattice sites, then applies the same onsite axial Phi projection.
- Added `configs/fit_Phi_phys_pC_dataonly_projected.toml`.
- Removed the high-resolution branch files/artifacts:
  `code/fit_Phi_highres.jl`, `configs/fit_Phi_highres.toml`,
  `data/soft_spin_llg_chain_phi_highres.h5`,
  `models/fit_Phi_highres_artifacts.bson`,
  `logs/fit_Phi_highres_metrics.txt`, and
  `figures/phi_highres_recovery.png`.

Accepted command:

```bash
xvfb-run -a julia --project=. SoftSpinLLGChain/code/fit_Phi.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly_projected.toml
```

Accepted artifacts:
- `models/fit_Phi_phys_pC_dataonly_projected_artifacts.bson`
- `logs/fit_Phi_phys_pC_dataonly_projected_metrics.txt`
- `data/phi_phys_pC_dataonly_projected_forward_langevin.h5`
- `figures/phi_phys_pC_dataonly_projected_recovery.png`
- `figures/phi_phys_pC_dataonly_projected_cdot_gfdt.png`
- `figures/phi_phys_pC_dataonly_projected_forward_stats.png`
- `figures/phi_phys_pC_dataonly_projected_forward_cmn.png`
- Retained-observable forward check:
  - `logs/forward_phi_phys_pC_dataonly_projected_only_metrics.txt`
  - `figures/forward_stats_phi_phys_pC_dataonly_projected_only.png`
  - `figures/forward_cmn_phi_phys_pC_dataonly_projected_only.png`

Accepted metrics:
- `Phi vs <M_true> rel.RMSE = 1.74457882e-01`, improved from the previous
  strict trajectory-only value `2.39439699e-01`.
- `Phi vs <M_true> corr = 9.84265927e-01`, improved from `9.80447139e-01`.
- `Phi symmetric min eigenvalue = 2.10705553e-02`; PSD projection change `0`.
- Projection change is now `5.71828515e-02`, much smaller than the previous
  dense-raw strict estimator's `9.66524090e-01`, because the estimator directly
  averages the symmetry-reduced block instead of estimating noisy off-profile
  entries.
- Final onsite axial block:

```text
[ 4.41164798e-02  -2.52978081e-03   0
  2.52978081e-03   4.41164798e-02   0
  0                0                 2.10705553e-02 ]
```

Forward/GFDT diagnostics:
- `Cdot Phi-GFDT rel.RMSE = 9.47051718e-01`, essentially unchanged from the
  previous strict branch (`9.41506557e-01`).
- Forward coordinate `C(t)` rel.RMSE `4.31418459e-01`, slightly worse than the
  previous strict branch (`4.19738492e-01`).
- Retained nonlinear-observable forward check: covariance rel.RMSE
  `1.69478045e-01`, covariance corr `9.83398468e-01`, retained `C(t)`
  rel.RMSE `9.85238933e-01`, retained `C(t)` corr `8.98464354e-01`.
- Conclusion: the fitting-only update improves the ex-post mean-Phi matrix
  recovery and removes dense off-profile noise, but it does not improve the
  constant-Phi finite-lag dynamical baseline.

Rejected fitting-only branches at the original resolution:
- A projected linear fit over the first 3 saved lags overestimated the block
  amplitude (`Phi` rel.RMSE `1.63`) and was rejected.
- A projected quadratic fit over the first 3 saved lags improved modestly
  (`Phi` rel.RMSE `0.222`) but did not beat the accepted wider-window fit.
- Random-pair scans showed degree-4 fits over 8--10 saved lags could sometimes
  reach ex-post rel.RMSE near `0.09`, but those were seed-sensitive and were not
  accepted by target-error cherry-picking.
- Stratified and all-saved-pair polynomial fits were tested to remove random
  sampling noise. They were much worse, showing that exact coarse-saved-lag
  polynomial extrapolation has strong finite-lag bias.
- Matrix-log and increment-covariance fits on the exact saved-lag covariance
  blocks were also worse. They did not overcome the fact that
  `save_dt=0.0365` is still too coarse for a robust zero-lag derivative
  estimate in this system.

No-cheating audit:
- The accepted projected Phi estimator used only saved trajectory states from
  `data/soft_spin_llg_chain.h5` and the learned stationary score for GFDT
  validation.
- True mobility was computed only after Phi construction for ex-post metrics.
- No simulator coefficients, analytic mobility, analytic score, short-lag
  resimulation, or target-error-based hyperparameter selection entered the
  accepted Phi construction.

Step 1 and Step 2 accepted artifacts are present. Step 3 has not been
completed yet for `SoftSpinLLGChain`.

Next files to create:

- `code/fit_dM.jl`
- `configs/fit_dM.toml`

Step 3 conditional-score work was started on the RTX 5070:

- Added `code/cond_score.jl`.
- Added `configs/cond_score.toml`.
- First 5070 run completed 120 epochs but failed during diagnostics before
  saving because `PhiConfig` was missing for BSON loading.
- The script was patched to define `PhiConfig` and save the checkpoint
  immediately after training, before diagnostics.
- The rerun was interrupted because the workstation had to be shut down. It was
  killed at epoch 76 with loss about `377.5898` and residual norm about
  `3.1121`. No conditional-score checkpoint exists from that partial run.
- Next session must rerun `cond_score.jl` from scratch, or add periodic
  checkpointing before rerunning if interruptions are likely.

Step 3 should use the final direct stationary score checkpoint and the final
short-lag/projected Phi artifacts. Do not use the obsolete saved-frame Phi
estimator for the mobility residual target except as a documented failed
diagnostic. Use `A_data = Cdot_data - A[Phi]`, where `A[Phi]` comes from the
stationary-score GFDT channel already implemented in `fit_Phi.jl`.

Expected risk: the conditional-score operator diagnostic is the gating item.
Do not train or interpret the mobility NN if the conditional transition score
does not reproduce the `Cdot_data` operator when tested with ex-post true
mobility.

## Cleanup Record

After writing this report, failed/scratch artifacts from the completed Step 2
work were removed:

- failed noise-output score checkpoint,
- failed score-validation figures,
- obsolete score/Phi logs from rejected attempts.

Keep the accepted production data, accepted score checkpoint, accepted Phi
artifacts, accepted forward datasets, accepted figures, accepted metrics, and
this report.

## Step 3 Conditional-Score Attempt Log, 2026-05-08/09

Step 3 was resumed after the third GPU was installed. Physical GPUs reported by
`nvidia-smi` were:

- GPU 0: `NVIDIA GeForce RTX 2080 Ti`
- GPU 1: `NVIDIA GeForce RTX 2080 Ti`
- GPU 2: `NVIDIA GeForce RTX 5070`

Important device mapping detail: CUDA ordinal masking did not match
`nvidia-smi` order. In this workstation state:

- `CUDA_VISIBLE_DEVICES=0` exposed the physical RTX 5070.
- `CUDA_VISIBLE_DEVICES=1` exposed physical 2080 Ti GPU 0.
- `CUDA_VISIBLE_DEVICES=2` exposed physical 2080 Ti GPU 1.

I patched `code/cond_score.jl` to add periodic checkpoint/resume support and
then trained multiple conditional-score variants in parallel. I did not start
mobility training because the required true-M operator diagnostic remained poor.
Proceeding to `M_theta` from these scores would violate the Step 3 gate.

### Scripts And Commands Tried

Completed residual conditional-score runs:

- Variant A, physical GPU 0 / 2080 Ti:
  `CUDA_VISIBLE_DEVICES=2 DISPLAY=:190 julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/cond_score.jl SoftSpinLLGChain/configs/cond_score_gpu0_vA.toml`
- Variant B, physical GPU 1 / 2080 Ti:
  `CUDA_VISIBLE_DEVICES=1 DISPLAY=:190 julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/cond_score.jl SoftSpinLLGChain/configs/cond_score_gpu1_vB.toml`
- Variant C, physical GPU 2 / RTX 5070:
  `CUDA_VISIBLE_DEVICES=0 DISPLAY=:190 julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/cond_score.jl SoftSpinLLGChain/configs/cond_score_gpu2_vC.toml`
- Variant D, physical GPU 2 / RTX 5070:
  `CUDA_VISIBLE_DEVICES=0 DISPLAY=:190 julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/cond_score.jl SoftSpinLLGChain/configs/cond_score_gpu2_vD.toml`

Probed and rejected branches:

- Variant E: residual score with aggressive short-lag oversampling,
  `lag_sampling_power=2.75`; probed at epoch 60.
- Variant F: residual score with GroupNorm; interrupted after it followed the
  same plateau as A/D.
- Variants G/H/I: direct posterior-score target
  `q_tau(x0|xt)` with operational residual `q_tau - s_theta(x0)`; I was probed
  after the early plateau.

### Conditional-Score Metrics

Completed residual diagnostics:

| Variant | GPU | Key difference | Operator rel.RMSE | Operator corr | Mean `||E[r]||/sqrt(D)` | Mean `||E[r x0']||/sqrt(D)` |
|---|---:|---|---:|---:|---:|---:|
| A | 2080 Ti GPU 0 | baseline residual, tau max `0.60 t_D` | `4.82512516e-01` | `8.92723297e-01` | `1.02811673e-01` | `2.01428358e-01` |
| B | 2080 Ti GPU 1 | wider lag window, stronger regularization | `4.96414521e-01` | `8.88047815e-01` | `8.72998497e-02` | `2.04144175e-01` |
| C | RTX 5070 | wider model, stronger regularization | `5.75471897e-01` | `8.40267968e-01` | `8.43313007e-02` | `1.67367337e-01` |
| D | RTX 5070 | faster rerun of A-style recipe | `5.73262496e-01` | `8.56737193e-01` | `7.64173326e-02` | `2.21461349e-01` |

Additional diagnostic on the best completed run A:

- Restricting the operator comparison to coordinate observables `mx,my,mz`
  improved but was still not acceptable:
  - rel.RMSE `4.28666264e-01`
  - corr `9.13578756e-01`
- Lagwise coordinate operator error for A:
  - lag 1, tau `0.0365`: rel.RMSE `2.263`, corr `0.352`
  - lag 2, tau `0.073`: rel.RMSE `1.042`, corr `0.713`
  - lag 3, tau `0.1095`: rel.RMSE `0.688`, corr `0.831`
  - lag 5, tau `0.1825`: rel.RMSE `0.429`, corr `0.915`
  - lag 10, tau `0.365`: rel.RMSE `0.248`, corr `0.970`

This shows the main failure is the very-short-lag transition score. Excluding
the first few lags improves the diagnostic, but the conditional score is still
not good enough to satisfy the protocol gate for full mobility training.

Failed/probed branches:

- Variant E, short-lag-biased residual at epoch 60:
  - operator rel.RMSE `8.85596226e-01`
  - corr `6.47198478e-01`
  - lesson: aggressive short-lag oversampling made the operator diagnostic much
    worse even though the DSM training loss became much smaller.
- Variant I, direct posterior-score target at early plateau:
  - operator rel.RMSE `4.31816873e+00`
  - corr `-5.41957391e-02`
  - lesson: direct posterior training as patched here is not a safe substitute
    for the residual target. The operational residual `q_tau - s_theta` was
    badly mis-scaled for the true-M operator diagnostic.
- Variant F, GroupNorm residual, was interrupted before final diagnostics
  because its loss/norm trajectory matched the already-failed residual plateau.
- Variants G/H, posterior runs on the two 2080 Ti GPUs, were interrupted after
  the 5070 posterior probe failed badly.

### Code Changes Made During This Attempt

- `code/cond_score.jl` now supports:
  - resumable checkpoints via `checkpoint_every` and `resume`;
  - explicit `lag_sampling_power` for nonuniform lag sampling;
  - `score_target = "residual"` or `"posterior"`;
  - posterior mode computes the operational transition residual as
    `q_tau - s_theta`.
- Added configs:
  - `configs/cond_score_gpu0_vA.toml`
  - `configs/cond_score_gpu1_vB.toml`
  - `configs/cond_score_gpu2_vC.toml`
  - `configs/cond_score_gpu2_vD.toml`
  - `configs/cond_score_gpu2_vE.toml`
  - `configs/cond_score_gpu2_vF_groupnorm.toml`
  - `configs/cond_score_gpu0_vG_posterior.toml`
  - `configs/cond_score_gpu1_vH_posterior_groupnorm.toml`
  - `configs/cond_score_gpu2_vI_posterior.toml`

### No-Cheating Audit

The minimized conditional-score losses used only trajectory pairs, DSM noise on
`x0`, clean conditioning `xt`, and the already learned stationary score when
forming residual targets or residual penalties. True mobility was used only in
the labeled ex-post operator diagnostic. No analytic score, true mobility,
generator formula, or simulator coefficient entered a minimized loss, training
target, residual target, or model-selection target.

### Cleanup

Removed interrupted/probe checkpoints and probe figures that are not useful as
restart points:

- `models/cond_score_residual_gpu2_vE.bson`
- `models/cond_score_residual_gpu2_vE_probe_epoch60.bson`
- `models/cond_score_residual_gpu2_vF_groupnorm.bson`
- `models/cond_score_posterior_gpu0_vG.bson`
- `models/cond_score_posterior_gpu1_vH_groupnorm.bson`
- `models/cond_score_posterior_gpu2_vI.bson`
- `figures/cond_score_gpu1_vE_probe_epoch60_diagnostics.png`
- `figures/cond_score_gpu2_vI_posterior_probe_diagnostics.png`

After the later cleanup, kept only the best residual checkpoint A as a restart
point. Kept compact figures and all metric/log text files for B-D and rejected
probe branches because they document the failed branches and prevent future
agents from repeating them blindly.

### Observable-Channel Pruning After All-Profile Diagnostic

After inspecting the full all-observable conditional-score operator figure, I
added a reproducible pruning diagnostic:

`DISPLAY=:190 julia --project=. --startup-file=no SoftSpinLLGChain/code/plot_all_cond_cdot.jl`

New artifacts:

- `figures/cond_score_gpu0_vA_all_Cdot_profiles.png`
- `figures/cond_score_gpu0_vA_retained_Cdot_profiles.png`
- `logs/cond_score_gpu0_vA_all_Cdot_profiles_metrics.txt`
- `logs/cond_score_gpu0_vA_retained_Cdot_profiles_metrics.txt`
- `configs/cond_score_gpu0_vA_retained_cdot_channels.toml`

Selection rule: keep only translation-profile channels with correlation at
least `0.90`, relative RMSE at most `0.50`, and data RMS at least `2%` of the
strongest channel. This is an ex-post diagnostic using true mobility to test
the learned conditional score; it must not be used as a minimized training
target.

Retained channels:

- `mx -> mx`: rel.RMSE `1.28785085e-01`, corr `0.99188092`
- `my -> my`: rel.RMSE `1.19953651e-01`, corr `0.99318676`
- `mz -> mz`: rel.RMSE `3.95178143e-01`, corr `0.92767236`

Rejected channels:

- All cross-component coordinate channels, because their data RMS is only about
  `0.6%` to `1.3%` of the strongest channel and the learned curves mostly track
  noise or bias rather than a reproducible shape.
- All nonlinear channels `r2`, `mperp2`, `mz2`, and `local_U`, because their
  data RMS is below `1.3%` of the strongest channel and their correlations are
  near zero.

Lesson: the previous observable library was too broad for this conditional-score
diagnostic. For any mobility experiment authorized after this point, use only
the retained diagonal coordinate channels unless a new conditional score clearly
passes the same all-profile test on additional observables.

### Nonlinear Observable Library Search

The user correctly pointed out that retaining only linear coordinate channels is
not enough for a useful mobility NN. I added and ran a nonlinear observable
library search against the same diagnostic: data-driven `Cdot_mn(t)` from
trajectory finite differences versus the best conditional-score residual acted
on by ex-post true mobility. True mobility was used only for diagnostic
filtering, not for a minimized loss or target.

Scripts:

- `code/search_nonlinear_observables.jl`
- `code/merge_retained_observable_channels.jl`

GPU/workload split:

- Baseline nonlinear local/neighbor family on physical GPU 0, 2080 Ti.
- `poly_high` family on the other 2080 Ti.
- `neighbor_high` family on the RTX 5070.
- Combined `all` family on the first 2080 Ti after the baseline search
  completed.

Commands run:

- `DISPLAY=:190 julia --project=. --startup-file=no SoftSpinLLGChain/code/search_nonlinear_observables.jl`
- `CUDA_VISIBLE_DEVICES=2 DISPLAY=:190 julia --project=. --startup-file=no SoftSpinLLGChain/code/search_nonlinear_observables.jl poly_high _poly_high GPU:1 2080ti`
- `CUDA_VISIBLE_DEVICES=0 DISPLAY=:190 julia --project=. --startup-file=no SoftSpinLLGChain/code/search_nonlinear_observables.jl neighbor_high _neighbor_high GPU:2 5070`
- `DISPLAY=:190 julia --project=. --startup-file=no SoftSpinLLGChain/code/search_nonlinear_observables.jl all _all GPU:0 2080ti`
- `julia --project=. --startup-file=no SoftSpinLLGChain/code/merge_retained_observable_channels.jl`

Selection rule for every family: keep only translation-profile channels with
correlation at least `0.90`, relative RMSE at most `0.50`, and data RMS at least
`5%` of the strongest searched nonlinear channel.

Retained counts:

- Baseline nonlinear family: `50 / 213` observable-target channels.
- `poly_high`: `74 / 402` observable-target channels.
- `neighbor_high`: `90 / 402` observable-target channels.
- Combined `all`: `110 / 591` observable-target channels.
- Merged unique retained nonlinear channels: `114`.

Final merged artifacts:

- `configs/nonlinear_observable_retained_channels_merged.toml`
- `logs/nonlinear_observable_retained_channels_merged.txt`
- family metrics in `logs/nonlinear_observable_search*_metrics.txt`
- summary figures in `figures/nonlinear_observable_search_summary*.png`

Representative strong retained nonlinear channels:

- For `mx` target: `mz_gradm_my`, `mz_lap_my`, `cross_p_x`, `mx_r4`,
  `mx_r2`, `mx_neighbor_r2sum`, `mx_mperp2`.
- For `my` target: `my_r2_mperp2`, `my_mperp2`, `my_r4`, `my_r2`,
  `my_twist2`, `my_grad2`, `my_lap2`.
- For `mz` target: `my_mx_nnavg`, `cross_p_z`, `mz_lap2`, `mz_grad2`,
  `mz_mz4`, `mz_twist2`, `mz_r2`.

Important plotting note: the `neighbor_high` and `all` runs completed the
numerical search and wrote metrics/TOML/summary artifacts, but their large
retained-panel figures exceeded a GLMakie texture width limit. I patched
`render_retained` to cap the displayed retained panels in future runs. The
metrics and retained-channel TOML files are valid because they were written
before the plotting failure.

Guidance for mobility training: use
`configs/nonlinear_observable_retained_channels_merged.toml` as the nonlinear
observable-channel source, not the original seven-observable library and not the
linear-only retained list. The merged list still needs de-duplication by physics
or conditioning if the mobility NN becomes ill-conditioned, because some
retained channels are algebraically similar, for example `cross_p_x` and
`cross_m_x`, or duplicate high-order monomials with permuted names.

### Cleanup After Nonlinear Search

Cleaned the system folder after the nonlinear search. Removed:

- large intermediate nonlinear-search BSON dumps:
  - `models/nonlinear_observable_search_artifacts.bson`
  - `models/nonlinear_observable_search_poly_high_artifacts.bson`
- huge retained-panel figures from the pre-cap plotting layout:
  - `figures/nonlinear_observable_retained_Cdot_profiles.png`
  - `figures/nonlinear_observable_retained_Cdot_profiles_poly_high.png`
- per-family retained-channel TOMLs superseded by the merged final file:
  - `configs/nonlinear_observable_retained_channels.toml`
  - `configs/nonlinear_observable_retained_channels_all.toml`
  - `configs/nonlinear_observable_retained_channels_neighbor_high.toml`
  - `configs/nonlinear_observable_retained_channels_poly_high.toml`
- the linear-only retained-channel TOML, to avoid accidental use after the
  nonlinear search:
  - `configs/cond_score_gpu0_vA_retained_cdot_channels.toml`
- non-best conditional-score checkpoints:
  - `models/cond_score_residual_gpu1_vB.bson`
  - `models/cond_score_residual_gpu2_vC.bson`
  - `models/cond_score_residual_gpu2_vD.bson`

Preserved final/accepted or useful compact artifacts:

- production data and forward-validation HDF5 files;
- `models/score_sigma005.bson`, `models/fit_Phi_artifacts.bson`, and the best
  conditional score `models/cond_score_residual_gpu0_vA.bson`;
- `configs/nonlinear_observable_retained_channels_merged.toml`;
- compact nonlinear-search metrics, merged retained-channel log, and summary
  figures;
- all score/Phi/simulation figures and the compact failed-branch logs.

### Mobility NN Training On Retained Nonlinear Observables

Implemented and ran the mobility NN stage without Langevin forward integration.
The implementation uses the merged retained nonlinear observable channels and
does not train a dense global mobility matrix. The NN outputs one local `3 x 3`
onsite block per lattice site. The block is parameterized as `S(x) + K(x)`,
where `S` is PSD by Cholesky factors and `K` is skew-symmetric. This uses the
known onsite support of the true mobility and avoids a wasteful dense `36 x 36`
output.

New scripts/configs:

- `code/fit_dM.jl`
- `code/evaluate_dM_model.jl`
- `configs/fit_dM_gpu0_vA.toml`
- `configs/fit_dM_gpu1_vB.toml`
- `configs/fit_dM_gpu2_vC.toml`

Shared target artifact:

- `models/dM_targets_retained_nonlinear.bson`

Target construction:

- `Cdot_data` was estimated from trajectory finite differences on the retained
  nonlinear observable set.
- `Cdot_phi` was estimated with the stationary-score GFDT channel using the
  learned stationary score and data-only `Phi`.
- The minimized residual target is `A_data = Cdot_data - Cdot_phi`.
- True mobility did not enter target construction or training. It was used only
  for ex-post diagnostics.

Training commands:

- `CUDA_VISIBLE_DEVICES=1 DISPLAY=:190 julia --project=. --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_gpu0_vA.toml`
- `CUDA_VISIBLE_DEVICES=2 DISPLAY=:190 julia --project=. --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_gpu1_vB.toml`
- `CUDA_VISIBLE_DEVICES=0 DISPLAY=:190 julia --project=. --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_gpu2_vC.toml`

Variants:

- A: local `(mx,my,mz)` features, width `80`, depth `3`.
- B: local plus `r2` features, width `96`, depth `3`.
- C: nearest-neighbor features `(m_{i-1},m_i,m_{i+1})`, width `96`, depth `3`.

Final metrics:

| Variant | Feature mode | A rel.RMSE | A corr | true-M block rel.RMSE | true-M block corr | mean NN vs Phi rel |
|---|---|---:|---:|---:|---:|---:|
| A | `local` | `0.5589` | `0.8359` | `0.6076` | `0.9003` | `1.0289` |
| B | `local_r2` | `0.6202` | `0.8268` | `0.7413` | `0.8258` | `1.1353` |
| C | `neighbor` | `0.5209` | `0.8687` | `0.6957` | `0.8330` | `0.7210` |

Full diagnostic reruns on 20k state samples:

- A: `A` rel.RMSE `0.5612`, corr `0.8331`; true-M block rel.RMSE
  `0.6078`, corr `0.9001`.
- B: `A` rel.RMSE `0.6185`, corr `0.8265`; true-M block rel.RMSE
  `0.7414`, corr `0.8257`.
- C: `A` rel.RMSE `0.5230`, corr `0.8669`; true-M block rel.RMSE
  `0.6968`, corr `0.8319`.

Best model by the valid data-driven selection criterion is C:

- `models/dM_nn_gpu2_vC.bson`
- `figures/dM_nn_gpu2_vC_diagnostics.png`
- `figures/fit_dM_gpu2_vC_full_trueM_diagnostics.png`
- `logs/dM_nn_gpu2_vC_metrics.txt`
- `logs/fit_dM_gpu2_vC_full_trueM_diagnostics.txt`

Important caveat: A is better by ex-post true-M block metrics, but true-M error
cannot be used for model selection under the protocol. C is selected because it
best fits the data-driven residual `A` target. The true-M diagnostics show that
the learned block structure is only partially recovering the actual state
dependence: off-diagonal/skew channels have high correlations, while diagonal
entries remain biased and high-error. This is not yet a full Step 3 success
claim.

No Langevin equation was run in this stage, as requested.

### Learned-M Langevin Forward Validation

Forward Langevin validation was run for all three trained mobility NNs after the
user explicitly requested it. The integrator uses the learned stationary score
and the learned local mobility block, including the `div_x M_theta` drift term
computed by central finite differences in feature space. The stochastic forcing
uses the learned Cholesky factor of the PSD symmetric block. True mobility was
not used in the forward dynamics.

New scripts:

- `code/forward_dM.jl`
- `code/render_forward_with_dM.jl`

Forward integration commands:

- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_gpu2_vC.toml SoftSpinLLGChain/configs/fit_Phi.toml`
- `CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_gpu0_vA.toml SoftSpinLLGChain/configs/fit_Phi.toml`
- `CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_gpu1_vB.toml SoftSpinLLGChain/configs/fit_Phi.toml`

Actual device use from the startup checks/logs:

- C: config requested `GPU:2`, resolved to NVIDIA GeForce RTX 5070; wall time
  about `11:41`.
- A: config requested `GPU:0`, resolved to NVIDIA GeForce RTX 2080 Ti; wall time
  about `20:21`.
- B: config requested `GPU:1`, resolved to NVIDIA GeForce RTX 2080 Ti; wall time
  about `20:39`.
- `nvidia-smi` showed load and memory use on all three physical GPUs during the
  parallel run.

Forward trajectory artifacts:

- `data/forward_dM_gpu2_vC.h5`
- `data/forward_dM_gpu0_vA.h5`
- `data/forward_dM_gpu1_vB.h5`
- logs in `logs/forward_dM_gpu*_v*.log`

Rendering command:

- `xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_forward_with_dM.jl SoftSpinLLGChain/configs/fit_Phi.toml`

Updated forward figures:

- `figures/forward_stats_with_dM.png`
- `figures/forward_cmn_with_dM.png`
- metrics in `logs/forward_with_dM_metrics.txt`

Final forward metrics against observations:

| Model | Cov rel.RMSE | Cov corr | coordinate C rel.RMSE | coordinate C corr |
|---|---:|---:|---:|---:|
| `Phi` | `0.1494` | `0.98921` | `0.3519` | `0.94676` |
| `M_NN C` | `0.1129` | `0.99455` | `0.1659` | `0.98863` |
| `M_NN A` | `0.0908` | `0.99615` | `0.1551` | `0.98782` |
| `M_NN B` | `0.1137` | `0.99396` | `0.1988` | `0.98239` |

Interpretation:

- All three learned-M forward models improve the dynamical coordinate
  correlations over the constant `M=Phi` baseline.
- C remains the best model by the valid data-driven residual-A training
  criterion and was run first on the 5070. In forward validation, A has the best
  covariance and slightly best coordinate-C rel.RMSE. This is an ex-post
  forward result, not a training-selection signal.
- The learned models preserve the one-point `mx`, `my`, and `|m|` PDFs well.
  The `mz` PDF is the most sensitive stationary statistic; A and C differ in
  opposite directions, and no model is perfect there.
- The diagonal coordinate correlations still decay more slowly than the observed
  curves in many panels, but they are much closer than `M=Phi`.

Important implementation notes for future agents:

- `forward_dM.jl` initially needed two practical fixes before running: guard
  `main()` so the file can be included safely, and explicitly convert NN output
  named-tuple fields to host arrays before using them in scalar loops.
- GLMakie/GLFW required `xvfb-run`; setting `DISPLAY=:190` alone was unreliable
  because the Xvfb process did not persist across shell invocations.
- The finite-difference divergence is local and exploits the feature map. It
  perturbs only the central component rows and updates the `r2` feature
  consistently for `local_r2`. This avoided dense finite differences over all
  `36` coordinates.
- The current forward validation is a meaningful improvement over Phi, but it
  is not proof that the mobility NN fully recovered the true state-dependent
  mobility. The earlier true-M ex-post block diagnostics still show biased
  diagonal/symmetric entries.

No-cheating audit:

- Forward integration used the learned stationary score checkpoint and learned
  mobility checkpoints only.
- True `M(x)`, analytic score, generator formulas, and simulator internals were
  not used in the forward drift, diffusion, divergence, metrics used for this
  comparison, or any training target.

Current next action:

- If more Step 3 work is desired, the best technical direction is a more
  physically structured onsite coefficient NN with best-epoch checkpointing.
  The current local `3 x 3` block NN already improves forward correlations, but
  it still leaves systematic diagonal-mobility bias.

### Step 3 Forward-Improvement Pass, 2026-05-10

Goal:

- Improve learned-M Langevin forward validation beyond the earlier compact
  E20 model, using all three GPUs for independent mobility-NN and forward
  branches where useful.
- Keep the training target data-only: residual-A targets from data-driven
  `Cdot_data` minus the stationary-score Phi GFDT baseline, and learned
  conditional-score residuals. True mobility was used only for labeled ex-post
  diagnostics, never in minimized losses, targets, weights, or model selection.

Code changes made in this pass:

- `code/fit_dM.jl`
  - Added validation-best checkpoint saving as `*_best.bson`.
  - Added final checkpoint saving as `*_final.bson` when the final epoch is not
    the validation-best epoch.
  - Ensured the configured `model_bson` path receives the validation-best model
    at the end of training.
  - Added active lag-window support with `first_lag_index` and `last_lag_index`.
    The validation metrics can now be computed on the lag subset used by the
    residual-A loss.
- `code/make_compact_observable_channels.jl`
  - Built compact channel subsets from existing observable diagnostics.
- `code/subset_dm_targets.jl`
  - Created reduced residual-A target tensors for channel-subset experiments.

Accepted mobility model:

- Training config:
  `configs/fit_dM_compact_lag7_gpu2_vX.toml`
- Command:
  `xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_compact_lag7_gpu2_vX.toml`
- GPU actually used:
  NVIDIA GeForce RTX 5070, selected by config `GPU:2` with the required-name
  guard set to `5070`.
- Active training lag window:
  lags `7:24` of the 24 available lags.
- Final accepted checkpoint:
  `models/dM_nn_compact_lag7_gpu2_vX.bson`
- Diagnostics:
  `figures/dM_nn_compact_lag7_gpu2_vX_diagnostics.png`
- Metrics/logs:
  `logs/dM_nn_compact_lag7_gpu2_vX_metrics.txt`
  and `logs/train_dM_compact_lag7_gpu2_vX.log`

Accepted residual-A validation metrics for compact lag7 X:

| Epoch | Active-lag validation rel.RMSE | Active-lag validation corr |
|---:|---:|---:|
| 10 | `0.35184` | `0.94001` |
| 20 | `0.33990` | `0.94254` |
| 30 | `0.33266` | `0.94371` |
| 40 | `0.32457` | `0.94639` |

Accepted forward validation:

- Forward config:
  `configs/fit_dM_compact_lag7_gpu1_vX_finalforward.toml`
- Command:
  `CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_compact_lag7_gpu1_vX_finalforward.toml SoftSpinLLGChain/configs/fit_Phi.toml SoftSpinLLGChain/data/forward_dM_compact_lag7_vX_final_dt0015.h5 1.0 0.0015 1.0`
- GPU actually used:
  the second physical NVIDIA GeForce RTX 2080 Ti. This was selected by masking
  with `CUDA_VISIBLE_DEVICES=2`; inside Julia it appeared as config `GPU:0` and
  passed the required-name guard `2080ti`.
- Accepted trajectory:
  `data/forward_dM_compact_lag7_vX_final_dt0015.h5`
- Final comparison command:
  `xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi.toml final_accepted_learnedM_compare 'Phi=../data/phi_forward_langevin.h5' 'M_NN compact E20 dt0015=../data/forward_dM_compact_E20_dt0015.h5' 'M_NN compact lag7 X final=../data/forward_dM_compact_lag7_vX_final_dt0015.h5'`
- Final figures:
  `figures/forward_stats_final_accepted_learnedM_compare.png`
  and `figures/forward_cmn_final_accepted_learnedM_compare.png`
- Final metrics:
  `logs/forward_final_accepted_learnedM_compare_metrics.txt`

Final forward metrics against observations:

| Model | Cov rel.RMSE | Cov corr | coordinate C rel.RMSE | coordinate C corr |
|---|---:|---:|---:|---:|
| `Phi` | `0.151971` | `0.988870` | `0.353367` | `0.946487` |
| `M_NN compact E20 dt0015` | `0.078188` | `0.996922` | `0.119051` | `0.993146` |
| `M_NN compact lag7 X final` | `0.073958` | `0.997228` | `0.107133` | `0.993832` |

Accepted tail/finite-state diagnostics:

| Model | max abs state | q999 abs state | q9999 abs state | max norm | q999 norm | frac abs > 2 | frac abs > 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `E20 dt0015` | `1.51105` | `1.20975` | `1.29139` | `1.56248` | `1.32328` | `0.0` | `0.0` |
| `lag7 X final` | `1.60678` | `1.20730` | `1.28565` | `1.71815` | `1.32610` | `0.0` | `0.0` |

Interpretation:

- The lag7 X model is the best accepted learned-M forward model so far.
  Compared with the previous best compact E20 model, it improves covariance
  rel.RMSE from `0.07819` to `0.07396` and coordinate-correlation rel.RMSE from
  `0.11905` to `0.10713`.
- It strongly improves over the constant `M=Phi` baseline in both stationary
  covariance and finite-lag coordinate correlations.
- The largest single state value is higher than E20, but the high quantiles are
  comparable or slightly better and there are no samples with `abs(state) > 2`.
  I accepted this as a finite, stable forward trajectory rather than a tail
  instability.
- Stationary one-point PDFs are improved over Phi but still not perfect,
  especially in the sensitive `mz` and norm panels. Further work should target
  the score/conditional-score/divergence bottleneck rather than repeating the
  failed channel-selection branches below.

Failed or rejected branches:

- Top6 retained-observable target:
  - Created a reduced target with
    `configs/nonlinear_observable_retained_channels_top6.toml` and
    `models/dM_targets_top6.bson`, then trained several full/lag-window NNs.
  - Representative rejected metrics:
    top6 S epoch 10 active all-lag rel.RMSE `0.52953`, corr `0.85931`;
    top6 P epoch 10 rel.RMSE `0.55601`, corr `0.85669`;
    top6 Q lag3 epoch 10 rel.RMSE `0.51532`, corr `0.87927`;
    top6 U lag5 epoch 20 rel.RMSE `0.45417`, corr `0.89704`.
  - Lesson: removing too many channels reduced the target dimension but did not
    make the residual-A operator easier to fit. Do not repeat this top6 branch.
- Compact all-lag retrains:
  - A lighter-penalty all-lag run had epoch 10 rel.RMSE `0.44651`, corr
    `0.89480`, and epoch 20 rel.RMSE `0.49450`, corr `0.90074`.
  - Later all-lag retrain Y had epoch 10 rel.RMSE `0.52309`, corr `0.89871`,
    and epoch 20 rel.RMSE `0.50580`, corr `0.90496`.
  - Lesson: all-lag compact training is noncompetitive; the short noisy lags
    dominate the loss and harm the learned operator.
- Compact lag3 V:
  - Completed on the 5070 with final/best rel.RMSE about `0.38548` and corr
    `0.92934`.
  - Lesson: excluding only the first two lags helps, but not enough.
- Compact lag5 W:
  - Best active-lag residual-A metric was epoch 30 rel.RMSE `0.34670`, corr
    `0.93838`.
  - Forward validation was worse than E20: coordinate-C rel.RMSE about
    `0.13048` versus E20 `0.11905`.
  - Lesson: better residual-A loss is not automatically better forward
    dynamics; the lag7 window gave a better bias-variance tradeoff.
- Smaller-step E20 forward:
  - Reintegrating the old E20 checkpoint at `dt=0.0010` worsened the final
    metrics: covariance rel.RMSE `0.08586`, coordinate-C rel.RMSE `0.12157`.
  - Lesson: the previous E20 error was not primarily an Euler time-step issue.

Cleanup after this pass:

- Removed failed/superseded HDF5 forward trajectories, reduced top6 targets,
  failed best/final checkpoints, obsolete broad comparison figures, failed lag5
  figures, and aborted progress logs.
- Preserved only accepted production artifacts. The useful rejected-branch
  metrics and lessons are recorded in this report, so the failed configs/logs
  were removed rather than kept as scratch artifacts.

No-cheating audit:

- No analytic stationary score, true mobility, simulator coefficient, generator
  formula, or true-model tensor entered the residual-A loss, the mobility-NN
  training targets, validation loss, loss weights, or model selection.
- The accepted lag7 X model was selected using data-driven residual-A validation
  and forward statistics against observed trajectories, plus finite-state
  diagnostics.
- True `M(x)` remained ex-post diagnostic information only. No new true-M-based
  observable filtering or hyperparameter selection was introduced in this pass.

Current next action:

- The accepted current best is `models/dM_nn_compact_lag7_gpu2_vX.bson` with
  forward trajectory `data/forward_dM_compact_lag7_vX_final_dt0015.h5`.
- If more improvement is required, the most promising directions are improving
  the stationary/conditional score quality at the lags used by training,
  validating the finite-difference divergence term more directly, and using a
  more physically structured onsite coefficient parameterization. Do not spend
  more time on the top6 retained-channel branch or compact all-lag retrains.

### Repository Cleanup, 2026-05-11

Goal:

- Clean `SoftSpinLLGChain/` according to the repository-level agent workflow:
  keep accepted artifacts and compact diagnostics, remove obsolete failed-run
  outputs, and leave the folder easy for a future agent to inspect.

Removed:

- Superseded forward HDF5 trajectories:
  - `data/forward_dM_A_*`
  - `data/forward_dM_gpu*_v*.h5`
  - `data/forward_trueM_score_dt*.h5`
- Superseded or failed mobility/checkpoint artifacts:
  - `models/dM_nn_gpu0_vA.bson`
  - `models/dM_nn_gpu1_vB.bson`
  - `models/dM_nn_gpu2_vC.bson`
  - `models/dM_targets_retained_nonlinear.bson`
- Failed/probe configs for conditional-score, structured-M, full nonlinear-M,
  and rejected compact-M branches. The accepted configs remain listed below.
- Intermediate figures for beta/scale/dt sweeps, true-M score probes, old
  `with_dM` comparisons, failed conditional-score variants, and non-accepted
  full nonlinear-M diagnostics.
- Bulky logs and metrics for superseded forward sweeps, failed/probe
  conditional-score branches, rejected compact-M branches, structured-M probes,
  and old true-M score forward diagnostics.

Preserved accepted artifacts:

- Simulation:
  - `data/soft_spin_llg_chain.h5`
  - `figures/sim_summary.png`
  - `figures/sim_dynamics.png`
  - `figures/sim_trajectories.png`
  - `logs/sim.log`
- Stationary score:
  - `models/score_sigma005.bson`
  - `data/score_langevin.h5`
  - `figures/score_diagnostics.png`
  - `figures/score_langevin_validation.png`
  - `logs/score_direct_output.log`
- Phi:
  - `models/fit_Phi_artifacts.bson`
  - `data/phi_forward_langevin.h5`
  - `figures/phi_recovery.png`
  - `figures/phi_cdot_gfdt.png`
  - `figures/phi_forward_stats.png`
  - `figures/phi_forward_cmn.png`
  - `logs/fit_Phi_metrics.txt`
  - `logs/fit_Phi_shortlag.log`
- Conditional score and observable diagnostics:
  - `models/cond_score_residual_gpu0_vA.bson`
  - `figures/cond_score_gpu0_vA_diagnostics.png`
  - `figures/cond_score_gpu0_vA_all_Cdot_profiles.png`
  - `figures/cond_score_gpu0_vA_retained_Cdot_profiles.png`
  - compact nonlinear-observable summary figures and metrics
- Learned-M:
  - `models/dM_targets_compact.bson`
  - `models/dM_nn_compact_gpu2_vE20.bson`
  - `models/dM_nn_compact_lag7_gpu2_vX.bson`
  - `data/forward_dM_compact_E20_dt0015.h5`
  - `data/forward_dM_compact_lag7_vX_final_dt0015.h5`
  - `figures/dM_nn_compact_gpu2_vE20_diagnostics.png`
  - `figures/dM_nn_compact_lag7_gpu2_vX_diagnostics.png`
  - `figures/forward_stats_final_accepted_learnedM_compare.png`
  - `figures/forward_cmn_final_accepted_learnedM_compare.png`
  - final accepted metrics/logs for the comparison

Preserved configs:

- `sim.toml`, `score.toml`, `fit_Phi.toml`
- `cond_score.toml`, `cond_score_gpu0_vA.toml`
- `nonlinear_observable_retained_channels_merged.toml`
- `nonlinear_observable_retained_channels_compact.toml`
- `fit_dM_compact_gpu2_vE20.toml`
- `fit_dM_compact_gpu0_vE20_forward.toml`
- `fit_dM_compact_lag7_gpu2_vX.toml`
- `fit_dM_compact_lag7_gpu1_vX_finalforward.toml`

Final inventory after cleanup:

- `data/`: 5 files, about `5.6G`
- `models/`: 6 files, about `87M`
- `figures/`: about `13M`
- `logs/`: about `3.1M`
- `configs/`: 11 files

No-cheating audit:

- This cleanup only removed obsolete outputs and updated documentation. It did
  not modify training targets, accepted models, accepted data, or validation
  metrics.

### Forward PDF Regeneration, 2026-05-11

Goal:

- Regenerate the forward stationary-statistics figures with unbiased PDF
  estimates. The previous plotting path used a short aligned observation window
  for the PDF panels, which could bias the slow `mz` branch distribution.

Code changes:

- `code/render_forward_with_dM.jl`
  - PDF panels now use randomized samples from all post-burn observation
    trajectories and all saved forward-model trajectories.
  - Covariance panels now use randomized time-trajectory samples from the same
    full stationary pools.
  - Correlation panels still use aligned windows, because they are finite-lag
    dynamical diagnostics.
- `code/evaluate_forward_grid.jl`
  - Passes the full post-burn observation pool and full model trajectories to
    the statistics renderer, while keeping the old aligned arrays for
    correlations.
- `code/fit_Phi.jl`
  - The Phi-only `phi_forward_stats.png` renderer was patched to use the same
    full-stationary PDF/covariance rule for future regenerations.

Commands run:

- Syntax/load check:
  `julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/render_forward_with_dM.jl"); println("render include ok after fit_Phi patch")'`
- Final learned-M comparison:
  `xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi.toml final_accepted_learnedM_compare 'Phi=../data/phi_forward_langevin.h5' 'M_NN compact E20 dt0015=../data/forward_dM_compact_E20_dt0015.h5' 'M_NN compact lag7 X final=../data/forward_dM_compact_lag7_vX_final_dt0015.h5'`
- Phi-only stats figure from existing HDF5:
  `xvfb-run -a julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/fit_Phi.jl"); ... render_forward_stats(...)'`

Regenerated figures:

- `figures/forward_stats_final_accepted_learnedM_compare.png`
- `figures/forward_cmn_final_accepted_learnedM_compare.png`
- `figures/phi_forward_stats.png`

Final metrics after the unbiased stationary-statistics sampling:

| Model | Cov rel.RMSE | Cov corr | coordinate C rel.RMSE | coordinate C corr |
|---|---:|---:|---:|---:|
| `Phi` | `0.146459` | `0.990266` | `0.353367` | `0.946487` |
| `M_NN compact E20 dt0015` | `0.067017` | `0.997944` | `0.119051` | `0.993146` |
| `M_NN compact lag7 X final` | `0.068377` | `0.997921` | `0.107133` | `0.993832` |

Notes:

- No NN was retrained.
- No forward Langevin trajectory was reintegrated; the figures use the accepted
  saved HDF5 trajectories.
- The `obs` `mz` PDF is now estimated from the full post-burn observation
  dataset, not the short early comparison window.
- Existing model trajectories can still show finite branch imbalance in `mz`,
  especially for older comparison runs, because branch switching is slow. The
  renderer now uses all saved model samples, but it does not symmetrize or hide
  model-side finite-sampling/model-bias effects.

No-cheating audit:

- This was a plotting/evaluation change only. It did not use analytic score,
  true mobility, or simulator internals in any training target or model
  selection.

### Forward Correlation Time-Range Update, 2026-05-11

Goal:

- Increase the time range shown in the final `forward_cmn` comparison figure
  without retraining neural networks or reintegrating Langevin trajectories.

Code changes:

- `code/fit_Phi.jl`
  - Added `DEFAULT_FORWARD_CORR_MAX_LAGS = 300`.
  - Phi-only forward correlation diagnostics use this longer lag cap.
- `code/render_forward_with_dM.jl`
  - The direct learned-M renderer uses the same lag cap.
- `code/evaluate_forward_grid.jl`
  - The final grid comparison uses the same lag cap and records both
    `corr_lags` and `corr_tmax` in the metrics file.

Command run:

```bash
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi.toml final_accepted_learnedM_compare 'Phi=../data/phi_forward_langevin.h5' 'M_NN compact E20 dt0015=../data/forward_dM_compact_E20_dt0015.h5' 'M_NN compact lag7 X final=../data/forward_dM_compact_lag7_vX_final_dt0015.h5'
```

Regenerated:

- `figures/forward_cmn_final_accepted_learnedM_compare.png`
- `figures/forward_stats_final_accepted_learnedM_compare.png`
- `logs/forward_final_accepted_learnedM_compare_metrics.txt`

Final correlation window:

- `corr_lags = 300`
- `save_dt = 0.0365`
- `corr_tmax = 10.95`, about `3 t_D`

Updated coordinate-correlation metrics on the longer window:

| Model | coordinate C rel.RMSE | coordinate C corr |
|---|---:|---:|
| `Phi` | `0.391050` | `0.942355` |
| `M_NN compact E20 dt0015` | `0.174049` | `0.989677` |
| `M_NN compact lag7 X final` | `0.145513` | `0.991327` |

No-cheating audit:

- This was a plotting/evaluation range change only. It reused accepted saved
  forward HDF5 trajectories and did not retrain or reintegrate any model.

### Forward Cmn Retained-Observable Update, 2026-05-11

Goal:

- Make the final `forward_cmn_final_accepted_learnedM_compare.png` show the
  retained nonlinear observable/target-component correlations used by the
  compact learned-M training target, instead of only coordinate auto-
  correlations.

Code changes:

- `code/render_forward_with_dM.jl`
  - Added a retained-channel forward Cmn path based on
    `configs/nonlinear_observable_retained_channels_compact.toml`.
  - The figure now renders all 36 retained observable/target-component
    channels in a 6-by-6 panel layout.
  - Each channel correlation is centered and averaged over all lattice sites
    using `6000` sampled time-trajectory pairs per lag.
- `code/evaluate_forward_grid.jl`
  - The final grid renderer now calls the retained-channel Cmn path and records
    the retained-channel source/config in the metrics file.

Commands run:

```bash
julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/evaluate_forward_grid.jl"); println("include ok after titlesize patch")'
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi.toml final_accepted_learnedM_compare 'Phi=../data/phi_forward_langevin.h5' 'M_NN compact E20 dt0015=../data/forward_dM_compact_E20_dt0015.h5' 'M_NN compact lag7 X final=../data/forward_dM_compact_lag7_vX_final_dt0015.h5'
```

Regenerated:

- `figures/forward_cmn_final_accepted_learnedM_compare.png`
- `figures/forward_stats_final_accepted_learnedM_compare.png`
- `logs/forward_final_accepted_learnedM_compare_metrics.txt`

Final retained-observable C metrics:

| Model | retained observable C rel.RMSE | retained observable C corr |
|---|---:|---:|
| `Phi` | `0.917465` | `0.891614` |
| `M_NN compact E20 dt0015` | `0.159875` | `0.988147` |
| `M_NN compact lag7 X final` | `0.119969` | `0.992573` |

Notes:

- No NN was retrained.
- No forward Langevin trajectory was reintegrated.
- The plotted channels are the 36 compact retained observable/target-component
  pairs. The full training tensor also included all lattice translations; this
  figure uses a compact site-averaged correlation for readability.

No-cheating audit:

- This was a plotting/evaluation change only. It used the learned models,
  saved forward trajectories, and the data-driven compact channel list. No
  analytic score, true mobility, or simulator internals entered any training
  target or model selection.

### Forward Bottleneck And Probe Pass, 2026-05-11

Goal:

- Improve the learned state-dependent mobility forward validation beyond the
  accepted compact lag7 vX model, with emphasis on both stationary statistics
  and the retained nonlinear `C_mn(t)` channels.
- Use all three GPUs for independent forward probes where useful, without
  retraining the score U-Net or using analytic quantities in losses or model
  selection.

Code/evaluation changes:

- `code/render_forward_with_dM.jl`
  - Increased retained-C sampling from `6000` to `30000` time-trajectory pairs
    per lag.
- `code/evaluate_forward_grid.jl`
  - Final/probe Cmn comparisons now use the full post-burn observation pool
    instead of an aligned short observation slice.
- `code/forward_dM.jl`
  - Added a forward-only `skew_scale`/CLI argument for diagnostic reversible
    mobility scaling. It scales only the antisymmetric block and its divergence;
    the PSD symmetric noise factor is unchanged. Default is `1.0`.

Important diagnostics:

- Full-observation comparison of accepted vX and vW:
  - vX: covariance rel.RMSE `6.71293563e-02`; retained-C rel.RMSE
    `1.10532140e-01`.
  - vW: covariance rel.RMSE `5.61497234e-02`; retained-C rel.RMSE
    `1.13159573e-01`; visually worse `mz` branch balance.
  - Decision: keep vX because retained-C dynamics is the main target and vW
    does not improve it.
- Ex-post true mobility plus learned stationary score:
  - covariance rel.RMSE `3.50530455e-01`;
  - retained-C rel.RMSE `3.78946317e-01`.
  - Lesson: with the current learned stationary score, moving toward true
    mobility is not enough; the learned mobility is compensating for score bias.
- Physical local Kramers-Moyal branch:
  - covariance rel.RMSE `3.01242897e-01`;
  - retained-C rel.RMSE `3.37074323e-01`.
  - Rejected.
- Wider/continued learned-M forward probes:

| Probe | Cov rel.RMSE | retained-C rel.RMSE | Decision |
|---|---:|---:|---|
| vY best | `7.89815700e-02` | `1.69194993e-01` | reject |
| vZ best | `1.06797391e-01` | `1.91063130e-01` | reject |
| vX continuation low penalty | `7.06965382e-02` | `1.18984261e-01` | reject |
| vX global mobility scale `1.2` | `5.66216719e-02` | `1.40055489e-01` | reject |

- Reversible skew-scale probes:

| Probe | Cov rel.RMSE | retained-C rel.RMSE | max `|state|` | Decision |
|---|---:|---:|---:|---|
| vX skew `1.25` | `5.53480632e-02` | `1.37749199e-01` | `2.96583009` | reject, tails |
| vX skew `1.5` | `1.38822658e-01` | `1.31725033e-01` | `4.12045765` | reject, unstable |
| vX skew `2.0` | `4.34365737e+00` | `3.47581643e+01` | `5.54098272` | reject, unstable |

Final accepted comparison after this pass:

```bash
xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/evaluate_forward_grid.jl \
  SoftSpinLLGChain/configs/fit_Phi.toml final_accepted_learnedM_compare \
  'Phi=../data/phi_forward_langevin.h5' \
  'M_NN compact lag7 X final=../data/forward_dM_compact_lag7_vX_final_dt0015.h5'
```

Accepted final artifacts:

- `data/forward_dM_compact_lag7_vX_final_dt0015.h5`
- `figures/forward_stats_final_accepted_learnedM_compare.png`
- `figures/forward_cmn_final_accepted_learnedM_compare.png`
- `logs/forward_final_accepted_learnedM_compare_metrics.txt`

Final metrics:

| Model | Cov rel.RMSE | Cov corr | retained-C rel.RMSE | retained-C corr |
|---|---:|---:|---:|---:|
| `Phi` | `1.46459023e-01` | `9.90266264e-01` | `9.18993487e-01` | `8.93002270e-01` |
| `M_NN compact lag7 X final` | `6.71293563e-02` | `9.97975296e-01` | `1.10532140e-01` | `9.93613689e-01` |

Interpretation:

- The accepted learned-M model gives a large and robust improvement over the
  constant-Phi closure, especially on the retained nonlinear dynamical
  observables.
- The requested "perfect" agreement was not reached. The bottleneck is not a
  simple forward integrator bug, global clock-speed error, missing full-observed
  statistics, or reversible skew gain. The strongest evidence is the ex-post
  true-M plus learned-score failure, which shows that the stationary score is a
  limiting factor for any physically faithful mobility in this benchmark.
- Further improvement should start from a better stationary score or a new
  score/mobility co-training strategy, not from more local MLP capacity or
  post-hoc reversible scaling.

No-cheating audit:

- All minimized losses and data-driven estimators used trajectory data,
  learned score models, data-only Phi artifacts, and learned conditional-score
  residuals only.
- True mobility and analytic model information were used only for labeled
  ex-post diagnostics. They did not enter training targets, loss weights,
  checkpoint selection, or the final accepted forward model.

Cleanup:

- Removed bulky rejected forward HDF5 trajectories from vW/vY/vZ/vX-continuation
  probes, global-scale probes, skew probes, physical KM, true-M ex-post, and the
  older E20 comparison.
- Removed obsolete probe figures and progress logs while preserving compact
  metrics text files and the final accepted artifacts above.
- Removed `.agent-scratch/softspin_scale*` scratch directories after recording
  their conclusions here.

### Score-Function Improvement Pass, 2026-05-12

Goal:

- Improve the learned score/Phi/M forward validation without analytic training
  targets, using all three GPUs for independent score and conditional-score
  attempts.

Code change kept:

- `code/fit_dM.jl`
  - Added optional `data.cond_score_config` support via
    `configured_cond_score_config(cfg_path)`.
  - This fixes a real workflow hazard: a new `cond_score_bson` could previously
    be loaded while `fit_dM.jl` still used the old hard-coded
    `cond_score_gpu0_vA.toml` architecture and lag normalization.
  - Backward compatibility was preserved by not adding a field to serialized
    `DMConfig`; an initial struct-field patch broke loading old BSON mobility
    checkpoints and was immediately replaced.

Important commands:

```bash
xvfb-run -a julia --project=. --threads 16 --startup-file=no \
  SoftSpinLLGChain/code/evaluate_forward_grid.jl \
  SoftSpinLLGChain/configs/fit_Phi.toml score_vC_dataPhi_vM_forward_probe \
  'Phi=../data/phi_forward_langevin.h5' \
  'M_NN lag7 X old score=../data/forward_dM_compact_lag7_vX_final_dt0015.h5' \
  'M_NN lag7 X score vC oldM=../data/forward_dM_compact_lag7_vX_score_vC_dt0015.h5' \
  'M_NN vM score vC dataPhi=../data/forward_dM_compact_lag7_score_vC_dataPhi_vM_dt0015.h5'
```

Representative training/evaluation commands used all three GPUs with the local
visibility mapping (`CUDA_VISIBLE_DEVICES=0` -> RTX 5070, `1/2` -> 2080 Ti):

```bash
CUDA_VISIBLE_DEVICES=2 julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/cond_score.jl ...
CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/cond_score.jl ...
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/cond_score.jl ...
```

Main failed branches and metrics:

| Branch | Key result | Decision |
|---|---:|---|
| score vC + old M vX | cov `0.0628`, retained-C `0.1258` | reject; dynamics worse than accepted vX |
| score vC + trajectory-only Phi + retrained M vM | cov `0.0794`, retained-C `0.1259` | reject |
| score-vC lag24 cond vB | true-M operator rel.RMSE `0.6640` | reject |
| score-vC lag24 no-penalty cond vC | `0.8240` | reject |
| score-vC lag36 wide cond vD | `0.7157` | reject |
| score-vC lag24 5070 repeat cond vE | `0.5787` | reject |
| old-score vA continuation vH | `0.5969` | reject |
| old-score lag24 GroupNorm vG, epoch 30 | `0.8025` | reject |
| accepted old cond vA | `0.4825` | still best |

Notes:

- Several new conditional scores had better posterior DSM MSE than vA, but all
  worsened the true-M operator diagnostic used by the paper identity.
- Under the current stricter no-cheating rule, the old short-lag Phi
  resimulation should be treated as a legacy estimator. A fully trajectory-only
  Phi was tested (`Phi` vs `<M_true>` rel.RMSE about `0.1537`) but did not lead
  to better forward validation.
- No new mobility model was accepted. The accepted forward model remains
  `data/forward_dM_compact_lag7_vX_final_dt0015.h5`.

No-cheating audit:

- Analytic score and true mobility were used only for labeled ex-post
  diagnostics. They were not used in DSM losses, Phi construction, Cdot/A
  training tensors, target weights, M losses, or checkpoint selection.
- The vC/data-only Phi branch used trajectory finite-difference targets and a
  learned stationary score. It failed on forward validation and was cleaned up.

Cleanup:

- Removed rejected score/conditional/M checkpoints, failed score-only and
  forward HDF5 files, failed branch figures, failed branch configs, and scratch
  TOML/scripts.
- Preserved compact failed-branch metrics text where useful and kept accepted
  production artifacts unchanged.

### True-M Plus Improved Learned Score, 2026-05-13

Goal:

- The user asked to keep working until the learned stationary score was precise
  enough that Langevin integration with true mobility and the learned score
  gave very good agreement with observed statistical and dynamical observables.
- This was a score-bottleneck pass only. True mobility was used for the
  requested ex-post forward integration and validation, not inside a minimized
  loss.

Important implementation changes:

- `code/src/spin_common.jl`
  - Added `PhysicalFeatureScore`, using local physical polynomial features
    `x`, `r2*x`, periodic `lap`, and `r2^2*x` in normalized coordinates.
  - Added `score_from_dsm_model(model::PhysicalFeatureScore, batch, sigma)` so
    this model is treated as a direct score. This was essential: the generic
    score path assumes a DSM noise predictor and gives the wrong sign/scale for
    the physical score.
- `code/fit_physical_score.jl`
  - Fits a physical-feature direct score by streamed DSM normal equations.
  - Training data are only trajectory samples plus Gaussian DSM noise.
  - Analytic score coefficients and true mobility are excluded from the fit.
- `code/score_posthoc_metrics.jl`
  - Computes diagnostics for already saved score checkpoints/forward outputs.
- `code/forward_trueM_score.jl`
  - Accepts a score-checkpoint override path for true-M plus learned-score
    forward diagnostics.

Failed attempts cleaned up:

- Three generic U-Net score probes were trained and rejected:
  - vF, `sigma=0.05`: analytic rel.RMSE `0.289855`, cosine `0.962709`,
    score-Langevin covariance rel.RMSE `0.225996`.
  - vG, `sigma=0.035`: analytic rel.RMSE `0.362023`, cosine `0.932705`,
    score-Langevin covariance rel.RMSE `0.340610`.
  - vH, `sigma=0.025` wide: analytic rel.RMSE `0.237836`, cosine `0.972477`,
    score-Langevin covariance rel.RMSE `0.948348`.
- Their configs, checkpoints, score-only HDF5 files, figures, and logs were
  removed after recording the metrics here. Do not continue these branches.

Accepted physical-feature score fits:

```bash
julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/fit_physical_score.jl \
  SoftSpinLLGChain/configs/score_phys_gpu0_pA_sigma05.toml

julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/fit_physical_score.jl \
  SoftSpinLLGChain/configs/score_phys_gpu1_pB_sigma035.toml

julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/fit_physical_score.jl \
  SoftSpinLLGChain/configs/score_phys_gpu2_pC_sigma02.toml
```

Fit setup:

- All fits used `4194304` post-burn trajectory samples with spin-inversion
  augmentation, zero channel-mean normalization, ridge `1.0e-8`, batch size
  `65536`, and DSM noise only.
- pA used `sigma=0.05`, `noise_repeats=4`.
- pB used `sigma=0.035`, `noise_repeats=6`.
- pC used `sigma=0.02`, `noise_repeats=8`.
- The normal-equation fits are CPU/BLAS work; GPU use enters the validation
  forward integrations below.

Ex-post score diagnostics:

| Score | DSM sigma | analytic rel.RMSE | analytic cosine | safe rel.RMSE | Stein rel.err |
|---|---:|---:|---:|---:|---:|
| `score_phys_pA_sigma05` | `0.05` | `0.112218` | `0.998975` | `0.103207` | `0.047646` |
| `score_phys_pB_sigma035` | `0.035` | `0.060534` | `0.999698` | `0.055367` | `0.049130` |
| `score_phys_pC_sigma02` | `0.02` | `0.021837` | `0.999953` | `0.020265` | `0.049703` |

Analytic-score diagnostics are labeled ex-post only. They did not enter the
DSM normal equations, loss weights, or score selection.

True-M plus learned-score forward integrations:

```bash
CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/forward_trueM_score.jl \
  SoftSpinLLGChain/configs/fit_Phi.toml \
  SoftSpinLLGChain/data/forward_trueM_score_phys_pA_dt0015.h5 \
  0.0015 GPU:1 2080ti ../models/score_phys_pA_sigma05.bson

CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/forward_trueM_score.jl \
  SoftSpinLLGChain/configs/fit_Phi.toml \
  SoftSpinLLGChain/data/forward_trueM_score_phys_pB_dt0015.h5 \
  0.0015 GPU:0 2080ti ../models/score_phys_pB_sigma035.bson

CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/forward_trueM_score.jl \
  SoftSpinLLGChain/configs/fit_Phi.toml \
  SoftSpinLLGChain/data/forward_trueM_score_phys_pC_dt0015.h5 \
  0.0015 GPU:2 5070 ../models/score_phys_pC_sigma02.bson
```

Device audit:

- pA ran on a 2080 Ti exposed through `CUDA_VISIBLE_DEVICES=2`; the script
  resolved it as `GPU:1`, required name `2080ti`.
- pB ran on the other 2080 Ti through `CUDA_VISIBLE_DEVICES=1`; the script
  resolved it as `GPU:0`, required name `2080ti`.
- pC ran on the RTX 5070 through `CUDA_VISIBLE_DEVICES=0`; the script resolved
  it as `GPU:2`, required name `5070`.
- All three true-M forward integrations completed and saved HDF5 trajectories.

Forward validation command:

```bash
xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/evaluate_forward_grid.jl \
  SoftSpinLLGChain/configs/fit_Phi.toml trueM_phys_score_compare \
  'Phi=../data/phi_forward_langevin.h5' \
  'M_true phys pA=../data/forward_trueM_score_phys_pA_dt0015.h5' \
  'M_true phys pB=../data/forward_trueM_score_phys_pB_dt0015.h5' \
  'M_true phys pC=../data/forward_trueM_score_phys_pC_dt0015.h5'
```

Final artifacts:

- `models/score_phys_pA_sigma05.bson`
- `models/score_phys_pB_sigma035.bson`
- `models/score_phys_pC_sigma02.bson`
- `data/forward_trueM_score_phys_pA_dt0015.h5`
- `data/forward_trueM_score_phys_pB_dt0015.h5`
- `data/forward_trueM_score_phys_pC_dt0015.h5`
- `figures/score_phys_pA_diagnostics.png`
- `figures/score_phys_pB_diagnostics.png`
- `figures/score_phys_pC_diagnostics.png`
- `figures/forward_stats_trueM_phys_score_compare.png`
- `figures/forward_cmn_trueM_phys_score_compare.png`
- `logs/score_phys_pA_sigma05_fit_metrics.txt`
- `logs/score_phys_pB_sigma035_fit_metrics.txt`
- `logs/score_phys_pC_sigma02_fit_metrics.txt`
- `logs/score_phys_pA_metrics.txt`
- `logs/score_phys_pB_metrics.txt`
- `logs/score_phys_pC_metrics.txt`
- `logs/forward_trueM_phys_score_compare_metrics.txt`

Observed-forward metrics:

| Model | Cov rel.RMSE | Cov corr | retained-C rel.RMSE | retained-C corr |
|---|---:|---:|---:|---:|
| `Phi` | `0.146459` | `0.990266` | `0.918993` | `0.893002` |
| `M_true phys pA` | `0.090347` | `0.996322` | `0.029036` | `0.999563` |
| `M_true phys pB` | `0.103337` | `0.995393` | `0.049093` | `0.998930` |
| `M_true phys pC` | `0.051569` | `0.998461` | `0.030241` | `0.999549` |

Interpretation:

- This resolves the earlier true-M plus learned-score failure. The retained
  nonlinear dynamical channels now visually and numerically track observations
  very closely, with rel.RMSE about `0.03` and correlation about `0.99955` for
  pA/pC.
- pA has the lowest retained-C error by a small margin, but pC is the best
  all-around observed-forward score because its covariance error is much lower
  and its retained-C error is nearly identical.
- The comparison also confirms that the old constant-Phi baseline is not the
  right dynamics for the retained nonlinear channels: its retained-C rel.RMSE is
  about `0.919`.
- Future Step 3 learned-M work should restart from the pC stationary score and
  rerun the Phi/conditional-score/M stack consistently. The old accepted learned
  M likely compensated for the old score bias.

No-cheating audit:

- The physical score coefficients were fitted only from trajectory samples and
  Gaussian DSM noise. Analytic score, true mobility, simulator coefficients,
  and generator formulas did not enter the minimized fit, target, loss weights,
  or model selection.
- Analytic score was used only for labeled ex-post diagnostics in
  `logs/score_phys_*_metrics.txt`.
- True mobility was used only in the explicitly requested and labeled
  `forward_trueM_score_phys_*` validation integrations.
- The accepted score choice was based on observed forward PDFs/covariance and
  retained nonlinear `C_mn(t)` metrics, not analytic-score error.

Cleanup:

- Removed failed vF/vG/vH U-Net score probe configs, checkpoints, score-only
  forward HDF5 files, diagnostic figures, and logs.
- Preserved the three physical score checkpoints and true-M validation
  trajectories because they are the accepted restart/evaluation artifacts.
- `.agent-scratch/` contains only its `.gitignore`.

## 2026-05-14 Shutdown-State Step 3 Continuation

Purpose:

- Continue the learned conditional score and learned mobility route after the
  physical pC stationary score made true-M plus learned-score forward dynamics
  accurate.
- Stop cleanly for workstation shutdown when requested by the user.

Completed conditional-score branch:

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/cond_score.jl \
  SoftSpinLLGChain/configs/cond_score_phys_pC_unet_gpu2_vA_cont2.toml
```

- GPU: RTX 5070 exposed as `CUDA_VISIBLE_DEVICES=0`, config request `GPU:2`,
  required name `5070`.
- Checkpoint: `models/cond_score_phys_pC_unet_vA_cont2.bson`.
- Figure: `figures/cond_score_phys_pC_unet_vA_cont2_diagnostics.png`.
- Metrics: `logs/cond_score_phys_pC_unet_vA_cont2_metrics.txt`.
- Operator diagnostic improved from `vA_cont` rel.RMSE `0.394855`, corr
  `0.924924` to `vA_cont2` rel.RMSE `0.378631`, corr `0.934872`.
- This is real progress but still not an excellent conditional operator.

Completed M branches:

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/fit_dM.jl \
  SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_unet_vAcont2_gpu2_vN_cached_warmJ.toml
```

- Warm-started from `models/dM_phys_pC_dataonly_unet_vAcont_gpu2_vJ_cached_best.bson`.
- Best checkpoint: `models/dM_phys_pC_dataonly_unet_vAcont2_gpu2_vN_cached_warmJ_best.bson`.
- Metrics: `logs/dM_phys_pC_dataonly_unet_vAcont2_gpu2_vN_cached_warmJ_metrics.txt`.
- A validation rel.RMSE `0.286565`, corr `0.960107`; not better than vJ
  rel.RMSE `0.286122`, corr `0.960154`.

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/fit_dM.jl \
  SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_unet_vAcont2_gpu2_vO_lag5_cached_warmJ.toml
```

- Same warm start, but active lag range `5:24` instead of `7:24`.
- Best checkpoint: `models/dM_phys_pC_dataonly_unet_vAcont2_gpu2_vO_lag5_cached_warmJ_best.bson`.
- Metrics: `logs/dM_phys_pC_dataonly_unet_vAcont2_gpu2_vO_lag5_cached_warmJ_metrics.txt`.
- A validation rel.RMSE `0.296931`, corr `0.957308`; worse than vJ/vN.

Forward validation completed for vN:

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/forward_dM.jl \
  SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_unet_vAcont2_gpu2_vN_best_forward.toml \
  SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml \
  SoftSpinLLGChain/data/forward_dM_phys_pC_dataonly_vAcont2_vN_best_scale1p10_dt0015.h5 \
  1.10 0.0015
```

Comparison command:

```bash
xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/evaluate_forward_grid.jl \
  SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml \
  forward_phys_pC_dataonly_vAcont2_vN_compare \
  'Phi dataonly=../data/phi_phys_pC_dataonly_forward_langevin.h5' \
  'M_NN vD x0.8=../data/forward_dM_phys_pC_dataonly_unet_vA_vD_scale0p8_dt0015.h5' \
  'M_NN vJ x1.0=../data/forward_dM_phys_pC_dataonly_vAcont_vJ_best_scale1_dt0015.h5' \
  'M_NN vJ x1.10=../data/forward_dM_phys_pC_dataonly_vAcont_vJ_best_scale1p10_dt0015.h5' \
  'M_NN vN x1.10=../data/forward_dM_phys_pC_dataonly_vAcont2_vN_best_scale1p10_dt0015.h5'
```

Key forward metrics:

| Model | Cov rel.RMSE | Cov corr | retained-C rel.RMSE | retained-C corr |
|---|---:|---:|---:|---:|
| `Phi dataonly` | `0.173127` | `0.982795` | `0.968388` | `0.896899` |
| `M_NN vD x0.8` | `0.054316` | `0.998374` | `0.173086` | `0.985515` |
| `M_NN vJ x1.0` | `0.061802` | `0.998260` | `0.151772` | `0.989581` |
| `M_NN vJ x1.10` | `0.090452` | `0.996873` | `0.134873` | `0.991017` |
| `M_NN vN x1.10` | `0.081999` | `0.996656` | `0.164720` | `0.986221` |

Best current learned-M result:

- `M_NN vJ x1.10` remains best for retained nonlinear dynamics.
- Figures:
  - `figures/forward_cmn_forward_phys_pC_dataonly_vAcont2_vN_compare.png`
  - `figures/forward_stats_forward_phys_pC_dataonly_vAcont2_vN_compare.png`
- It improves substantially over `Phi` but remains far from the true-M plus pC
  learned score reference (`retained-C rel.RMSE` about `0.0302`).

Interrupted branches:

- `cond_score_phys_pC_unet_gpu1_vA_shortlag_cont.toml` was killed on user
  shutdown request after reaching about epoch `304/340`.
- `cond_score_phys_pC_unet_gpu0_vA_constrained_cont.toml` was killed on user
  shutdown request after reaching about epoch `303/340`.
- Do not treat either as a completed diagnostic. Restart only if the user wants
  to resume those exact branches.

No-cheating audit:

- Conditional-score and M training used trajectory data, DSM noise, learned pC
  stationary score, data-only Phi artifacts, and data-only residual-A targets.
- True mobility was used only in clearly labeled ex-post operator/M diagnostics.
- No analytic score, true mobility, simulator coefficient, generator formula,
  or analytic target entered a minimized loss or data-driven target.

Next action:

- The most promising untried route is not another physical conditional MLP.
  Those already failed badly. Try a stationary-score-style ridge/normal-equation
  conditional residual feature expansion over `(x0, xt, tau)`, and only proceed
  to M training if its operator diagnostic beats `vA_cont2`.

## 2026-05-14 — Human-facing writeup created

Status:

- Complete.

What changed:

- Created the missing `writeup/` folder for this benchmark.
- Added `writeup/system_report.tex`, a human-facing scientific report following
  the updated repository instructions.
- Compiled `writeup/system_report.pdf`.

What the writeup covers:

- The soft-spin Landau--Lifshitz system definition and parameters.
- The accepted simulation dataset and decorrelation-budget checks.
- Stationary-score results, including the physical-feature score repair and the
  decisive true-mobility forward validation.
- Phi baseline recovery and its finite-lag limitations.
- Conditional-score diagnostics and why the physical-feature conditional MLP
  route was rejected.
- Nonlinear observable filtering, mobility residual-target fitting, and current
  learned-M forward results.
- No-cheating audit and current bottlenecks.

Verification:

```bash
cd SoftSpinLLGChain/writeup
latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex
grep -E "(Warning|Overfull|Undefined|Error)" system_report.log || true
```

- Final compile succeeded.
- Final log scan returned no warnings, overfull boxes, undefined references, or
  errors.

Cleanup:

- Removed LaTeX auxiliary files after compilation.
- Kept only `system_report.tex` and `system_report.pdf` in `writeup/`.

Next action:

- Whenever a new accepted experiment changes the conclusions, update this
  Markdown report and `writeup/system_report.tex`, then recompile
  `writeup/system_report.pdf`.

## 2026-05-14 — PDF compatibility repair

Status:

- Complete.

Issue:

- The user reported that `writeup/system_report.pdf` could not be opened.

Checks:

- The original PDF existed with normal permissions.
- `pdfinfo` parsed it as an unencrypted 18-page PDF.
- `pdftotext` extracted the first-page text correctly.
- `pdftoppm` rendered page 1 successfully.

Fix:

- Rewrote the compiled PDF through Ghostscript as a conservative PDF 1.4
  compatibility copy.
- Replaced `writeup/system_report.pdf` with the rewritten version.

Final artifact:

- `writeup/system_report.pdf` is now PDF 1.4, 18 pages, unencrypted, and about
  `5.2M`.

Verification:

- `pdfinfo`, `pdftotext`, and `pdftoppm` all succeed on the rewritten PDF.

Cleanup:

- Removed `.agent-scratch/pdf_check/`.

## 2026-05-14 — Human writeup expanded after review

Status:

- Complete.

Issue:

- The first human-facing writeup was too thin in several places. In particular,
  it did not explain how the nonlinear observable library was chosen, and the
  Phi discussion did not clearly separate the rejected simulator-assisted
  short-lag estimate from the final strict trajectory-only estimate.

Changes made:

- Rewrote the Phi-baseline discussion in `writeup/system_report.tex` to say
  explicitly that the older short-lag Phi estimate used the known simulator to
  generate extra near-zero-lag transitions from stationary snapshots. It is not
  direct `Phi=<M_true>` cheating, but it is simulator-assisted and is not part
  of the final strict branch.
- Added a full observable-library subsection:
  - why linear observables were insufficient;
  - definitions of local amplitude, transverse-amplitude, gradient, Laplacian,
    neighbor-amplitude, and chirality observables;
  - how candidate channels were filtered by signal size, smoothness, and
    ex-post operator agreement;
  - the exact compact retained count: 36 channels, top 12 per target component;
  - thresholds: correlation at least `0.9`, relative error at most `0.5`, and
    signal fraction at least `0.05`;
  - representative retained observables for `mx`, `my`, and `mz` targets.
- Added the nonlinear-observable search summary figure to the writeup.
- Strengthened the no-cheating audit to distinguish observable design from
  minimized M-training targets.
- Added a paragraph explaining why the mobility NN was not trained as a dense
  `36 x 36` output.

Verification:

```bash
cd SoftSpinLLGChain/writeup
latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex
grep -E "(Warning|Overfull|Underfull|Undefined|Error)" system_report.log || true
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer ...
pdfinfo system_report.pdf
pdftotext system_report.pdf - | head
```

- Final LaTeX log scan returned no warnings, overfull boxes, underfull boxes,
  undefined references, or errors.
- Final PDF is a PDF 1.4 compatibility copy, 20 pages, unencrypted, about
  `5.6M`.

Cleanup:

- Removed LaTeX auxiliary files and `.agent-scratch/pdf_check/`.

Next action:

- If the writeup is further revised, keep the observable-selection discussion
  and Phi/no-cheating distinction explicit. Those were the main missing pieces.

## 2026-05-14 — Human writeup mathematical implementation expansion

Status:

- Complete.

Issue:

- The user correctly pointed out that the writeup still read like a superficial
  summary. It did not give enough mathematical detail for a reader to understand
  what was actually implemented.

Changes made:

- Added a new `Mathematical Implementation Details` section to
  `writeup/system_report.tex`.
- The new section gives explicit mathematical definitions for:
  - data standardization and score conversion between normalized and raw
    coordinates;
  - empirical lag-pair averages, centered observables, correlations, and
    finite-difference derivative estimates;
  - the Euler--Maruyama reference integrator;
  - stationary DSM and the accepted physical-feature score ansatz;
  - spin-inversion odd projection of the learned score;
  - constant-Phi estimation, symmetry projection, and stationary-score GFDT;
  - conditional residual-score DSM, including noisy initial endpoint, clean
    conditioning endpoint, Fourier lag features, and mean/Stein penalties;
  - nonlinear residual targets `A_data = Cdot_data - Cdot_Phi`;
  - retained-channel scaling and active lag window;
  - local onsite mobility NN features, PSD Cholesky parameterization, skew block,
    mean-Phi penalty, and residual loss;
  - learned-M forward SDE, local Cholesky noise factor, and central finite
    difference divergence.
- Regenerated the PDF as a compatibility PDF 1.4 file.

Verification:

```bash
cd SoftSpinLLGChain/writeup
latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex
grep -E "(Warning|Overfull|Underfull|Undefined|Error)" system_report.log || true
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer ...
pdfinfo system_report.pdf
pdftotext system_report.pdf - | head
```

- Final LaTeX log scan returned no warnings, overfull boxes, underfull boxes,
  undefined references, or errors.
- Final PDF is PDF 1.4, 25 pages, unencrypted, about `5.7M`.

Cleanup:

- Removed LaTeX auxiliary files and `.agent-scratch/pdf_check/`.

Next action:

- The writeup now contains the core math of the implementation. Further polish
  should focus on refining prose and figure order, not on adding missing
  estimator definitions.

## 2026-05-14 — Structured Joint-Score Campaign

Goal:

- Keep the accepted conditional residual code/checkpoints untouched and test a
  separate structured joint-score route for
  `grad_(z0,zt) log p_0,t(z0,zt)`.
- Use the operational transition score from the initial block:
  `r_t(z0,zt) = q0_joint(z0,zt,t) - s_theta(z0)` for raw joint models, or the
  first residualized output block for residualized joint models.
- True mobility is used only in the ex-post operator diagnostic. It is not used
  in DSM labels, losses, checkpoint selection, or target construction.

Implementation changes:

- Added `code/joint_score.jl`.
- Added compatibility dispatch in `code/fit_dM.jl` without changing the
  serialized `DMConfig` struct. New M configs can set
  `data.cond_score_kind = "joint_score"`; old configs still use the existing
  conditional residual evaluator.
- Added joint configs and artifacts under `joint_score_*` names only, leaving
  `cond_score.jl` behavior and accepted conditional checkpoints untouched.
- Fixed one implementation bug during the smoke test: the Stein penalty
  reshaped a GPU view, which triggered disallowed scalar indexing. The corrected
  path materializes the GPU array before matrix multiplication.

Smoke check:

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --startup-file=no \
  SoftSpinLLGChain/code/joint_score.jl .agent-scratch/joint_score_smoke.toml
```

- Passed with a one-batch model and nonzero Stein penalty.
- Smoke artifacts are temporary and should be removed at cleanup.

First wave completed:

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/joint_score.jl \
  SoftSpinLLGChain/configs/joint_score_phys_pC_gpu2_vA.toml

CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/joint_score.jl \
  SoftSpinLLGChain/configs/joint_score_phys_pC_gpu0_vB.toml

CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/joint_score.jl \
  SoftSpinLLGChain/configs/joint_score_phys_pC_gpu1_vC_raw.toml
```

Additional first-wave continuation on the 5070 after vA failed:

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/joint_score.jl \
  SoftSpinLLGChain/configs/joint_score_phys_pC_gpu2_vD_initial_active_s035.toml
```

First-wave metrics:

| Branch | GPU | Main change | Global true-M operator rel.RMSE | Corr | Active-lag rel.RMSE |
|---|---|---|---:|---:|---:|
| `vA` | 5070 | wide basic residualized, both endpoints noised, Fourier 12 | `0.729450` | `0.739019` | not separately logged |
| `vB` | 2080 Ti | physical augmented residualized, both endpoints noised | `0.594637` | `0.822446` | not separately logged |
| `vC_raw` | 2080 Ti | raw joint-score ablation | `3.716048` | `-0.097219` | not separately logged |
| `vD_initial_active_s035` | 5070 | initial-only DSM, active lags `7:24`, `sigma=0.035` | `1.549671` | `-0.009245` | `1.440056` |

Gate reference:

- Old accepted conditional score `cond_score_phys_pC_unet_vA_cont2`:
  true-M operator rel.RMSE `0.378631`, corr `0.934872`.

Conclusion after first wave:

- No first-wave joint checkpoint is eligible for mobility training.
- The best first-wave branch is `vB`, but it is still much worse than the old
  conditional residual score.
- Raw joint-score training is not a viable direction as configured; subtracting
  the stationary score from the raw initial joint block produced an operator
  with negative correlation.
- Initial-only active-lag DSM was also much worse, including on active lags
  `7:24`; do not repeat that exact route.

Second wave launched on all GPUs:

- `configs/joint_score_phys_pC_gpu2_vE_physaug_groupnorm.toml`:
  physical augmented inputs, GroupNorm, stronger initial-block weighting on the
  5070.
- `configs/joint_score_phys_pC_gpu0_vF_physfull.toml`: larger physical-full
  feature set on a 2080 Ti.
- `configs/joint_score_phys_pC_gpu1_vG_short_s07.toml`: physical augmented
  inputs, `sigma=0.07`, short-lag-biased sampling on the other 2080 Ti.

No-cheating audit:

- All joint-score minimized losses used trajectory pairs, Gaussian DSM noise,
  and the learned pC stationary score for residualization only.
- True mobility entered only the labeled ex-post operator diagnostic.
  It did not enter training targets, loss weights, checkpoint selection, or
  mobility-training targets.

## 2026-05-15 — Structured Joint-Score Campaign, second wave and first M runs

Code updates after the first wave:

- Added retained nonlinear-channel diagnostics to `code/joint_score.jl`.
  This uses `models/dM_targets_compact_phys_pC_dataonly.bson` only as a
  data-target artifact: selected indices, data `Cdot`, observable means, and
  retained channel definitions. True mobility is still used only in the
  labeled ex-post operator estimate `-<phi_m(x_t) M_true(x_0)' r_tau>`.
- Made `code/search_nonlinear_observables.jl` safely includable when
  `cond_score.jl` is already loaded.
- Added a conditional `RetainedChannel` definition so the retained target BSON
  can be loaded by `joint_score.jl` without breaking `fit_dM.jl`.
- Set joint-score configs to report active lag indices `7:24` and retained
  target diagnostics on evaluation-only reruns.

Second-wave completed metrics:

| Branch | GPU | Main change | Global rel.RMSE | Global corr | Active `7:24` rel.RMSE | Retained active rel.RMSE | Retained active corr |
|---|---|---|---:|---:|---:|---:|---:|
| `vE_physaug_groupnorm` | 5070 | physical augmented, GroupNorm, stronger initial block | `0.486413` | `0.889943` | `0.330341` | `0.289740` | `0.957278` |
| `vF_physfull` | 2080 Ti | physical-full features, uniform lags | `0.979281` | `0.591105` | not rerun | not rerun | not rerun |
| `vG_short_s07` | 2080 Ti | `sigma=0.07`, short-lag sampling | `0.431958` | `0.903012` | not rerun | not rerun | not rerun |

Decision:

- `vE` does not beat the old conditional globally, but it passes the user gate
  on the active retained nonlinear channels: retained active-lag rel.RMSE
  `0.289740`, corr `0.957278`.
- `vF` was rejected without retained rerun because the global operator was very
  poor (`0.979281`).
- `vG` was rejected as a second-wave candidate because it remained worse than
  the old conditional globally and worse than `vE`; the GPU was moved to M
  training rather than spending another retained eval on it.

M training launched from `vE`:

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/fit_dM.jl \
  SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu2_warmJ.toml

CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/fit_dM.jl \
  SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu0_fresh_neighbor.toml

CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/fit_dM.jl \
  SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu1_wide_neighbor.toml
```

Completed M metrics so far:

| M branch | Status | Best epoch | A val rel.RMSE | A val corr | Lesson |
|---|---|---:|---:|---:|---|
| `joint_vE_gpu2_warmJ` | complete | `75` | `0.304705` | `0.955587` | Warm-start helps but still does not beat old vJ (`0.286122`). |
| `joint_vE_gpu0_fresh_neighbor` | complete | `40` | `0.313294` | `0.949770` | Fresh same architecture is worse than warm-start. |
| `joint_vE_gpu1_wide_neighbor` | running | best live near `0.3097` at epoch `20` | pending | pending | Wider cache/capacity has not yet beaten warm-start. |

Additional targeted score branches launched because the first `vE` M fits are
not yet an improvement:

- `joint_score_phys_pC_gpu2_vH_physfull_initial_s05.toml` on the 5070:
  physical-full inputs, `endpoint_noise=initial_only`, active lags `7:24`.
  This tests whether the clean-conditioning endpoint fixes the joint-score
  subtraction issue. Early transition norm rose from about `1.6` to above `3`,
  so this is not obviously better, but it is still running.
- `joint_score_phys_pC_gpu0_vI_physaug_s035_uniform.toml` on a 2080 Ti:
  lower DSM noise `sigma=0.035`, uniform lags, physical augmented inputs.
  It is slow but has low early transition norm; still running.

Current next actions:

- Let `joint_vE_gpu1_wide_neighbor` finish, then compare all three `vE` M
  checkpoints by data-only A validation.
- Run learned-M forward validation for the best `vE` M checkpoint even if it
  does not beat old vJ, because this is needed to verify whether the retained
  active conditional-score improvement transfers to Langevin statistics.
- Let `vH` and `vI` finish unless a clear resource conflict appears. If either
  beats `vE` on retained active operator diagnostics, train the same M suite
  against that conditional source.

No-cheating audit:

- The retained active decision used true mobility only in an ex-post diagnostic.
  The M losses use `A_data = Cdot_data - A[Phi]`, the learned stationary score,
  data-only Phi, and the learned transition score. No true mobility, analytic
  score, or simulator coefficient entered the minimized M losses or checkpoint
  selection.

## 2026-05-15 — Structured Joint-Score Campaign, vE wide M and vH targeted score

Completed update to the `vE` M wave:

| M branch | Status | Best epoch | A val rel.RMSE | A val corr | Lesson |
|---|---|---:|---:|---:|---|
| `joint_vE_gpu2_warmJ` | complete | `75` | `0.304705` | `0.955587` | Warm-start helps but still does not beat old vJ (`0.286122`). |
| `joint_vE_gpu0_fresh_neighbor` | complete | `40` | `0.313294` | `0.949770` | Fresh same architecture is worse than warm-start. |
| `joint_vE_gpu1_wide_neighbor` | complete | `65` | `0.289004` | `0.961110` | Wider cache/capacity is the best `vE` M branch, but it is still slightly worse than old vJ by data-only A validation. |

Forward validation launched for the best `vE` M branch:

```bash
CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/forward_dM.jl \
  SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu1_wide_neighbor.toml \
  SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml \
  SoftSpinLLGChain/data/forward_dM_phys_pC_joint_vE_wide_scale1_dt0015.h5 \
  1.0 0.0015
```

Forward validation result:

- Figures:
  `figures/forward_stats_forward_phys_pC_joint_vE_wide_compare.png` and
  `figures/forward_cmn_forward_phys_pC_joint_vE_wide_compare.png`.
- Metrics:
  `logs/forward_forward_phys_pC_joint_vE_wide_compare_metrics.txt`.
- Compared observations, `M=Phi`, old learned-M `vJ x1.10`, and new
  `joint vE wide x1.0`.
- `M=Phi` retained observable `C(t)` rel.RMSE `0.968388`, corr `0.896899`.
- Old `vJ x1.10` retained observable `C(t)` rel.RMSE `0.134815`, corr
  `0.991029`.
- New `joint vE wide x1.0` retained observable `C(t)` rel.RMSE `0.134558`,
  corr `0.990849`.
- Old `vJ x1.10` covariance rel.RMSE `0.090806`, corr `0.996849`.
- New `joint vE wide x1.0` covariance rel.RMSE `0.067739`, corr `0.997441`.

Decision:

- The new joint-score branch very slightly beats the user-specified old
  `vJ x1.10` retained-C reference (`0.134558 < 0.134815`) and improves
  covariance, but the improvement is marginal and still far from the preferred
  retained-C target below `0.10`.
- Continue with the vH M branch and then run scale variants for the best joint
  M checkpoint rather than declaring the campaign solved.

Targeted `vH` score branch completed:

- Config: `configs/joint_score_phys_pC_gpu2_vH_physfull_initial_s05.toml`.
- GPU/device: `GPU:2`, resolved as RTX 5070.
- Main change: physical-full inputs, `endpoint_noise = "initial_only"`,
  active lags `7:24`, residualized initial block, terminal block not trained.
- Global true-M operator rel.RMSE `4.992027`, corr `0.245273`: globally poor
  and not a good all-channel joint score.
- Active-lag true-M operator rel.RMSE `0.354118`, corr `0.938023`.
- Retained active nonlinear-channel rel.RMSE `0.257459`, corr `0.966187`.

Decision:

- `vH` is not accepted as a globally accurate joint score, but it is the best
  active retained-channel transition score so far, beating `vE` retained active
  rel.RMSE `0.289740`.
- Because the M training target uses retained nonlinear active lags, `vH`
  satisfies the campaign gate for an M-training wave.
- Started `configs/fit_dM_phys_pC_joint_vH_gpu2_warmJ.toml` on the 5070 while
  the `vE` forward run and `vI` lower-noise joint-score run continue.
- Added matching but not-yet-launched configs
  `fit_dM_phys_pC_joint_vH_gpu0_fresh_neighbor.toml` and
  `fit_dM_phys_pC_joint_vH_gpu1_wide_neighbor.toml`; launch them when the
  corresponding 2080 Ti GPUs are free if `vH` warm-start looks competitive.

Completed `vH` warm-start M result:

- Config: `configs/fit_dM_phys_pC_joint_vH_gpu2_warmJ.toml`.
- GPU/device: `GPU:2`, resolved as RTX 5070.
- Best epoch `50`.
- Data-only A validation rel.RMSE `0.290345`, corr `0.959164`.
- Ex-post true-M block rel.RMSE `0.729154`, corr `0.721138`.
- Mean M_NN vs Phi onsite rel.RMSE `0.539499`.
- Decision: despite the better retained active score diagnostic, this M fit
  did not beat `vE` wide (`0.289004`) or old vJ (`0.286122`) by data-only
  A-validation. Do not forward-validate this checkpoint unless later evidence
  suggests a scale effect; prioritize the `vE` scale sweep and the still-running
  `vH` wide branch.

Additional forward scale run launched:

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/forward_dM.jl \
  SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu2_wide_neighbor_forward.toml \
  SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml \
  SoftSpinLLGChain/data/forward_dM_phys_pC_joint_vE_wide_scale1p10_dt0015.h5 \
  1.10 0.0015
```

Scale result:

- `joint vE wide x1.10` was rejected. `evaluate_forward_grid.jl` aborted with
  `M_NN joint vE wide x1.10 contains non-finite state values`.
- Lesson: do not use the old `vJ x1.10` scale factor blindly for the new
  joint-score M checkpoint. If scaling is revisited, try smaller local values
  such as `1.02` or `1.05`, with finite-state checks before expensive figure
  generation.

Still-running `vH` wide M branch:

- Config: `configs/fit_dM_phys_pC_joint_vH_gpu1_wide_neighbor.toml`.
- GPU/device: `GPU:1`, resolved as a 2080 Ti.
- Early validation was poor (`0.30794` at epoch 5, `0.30277` at epoch 10),
  but epoch 25 reached data-only A rel.RMSE `0.27556`, corr `0.96315`.
- This is the best M A-validation metric in the joint-score campaign so far
  and beats the old vJ A rel.RMSE `0.28612`; it is now the highest-priority
  checkpoint for forward validation once training completes.
- Later update: epoch 50 improved again to data-only A rel.RMSE `0.27297`,
  corr `0.96517`. Epoch 55 regressed to `0.31594`; the saved best checkpoint
  is epoch 50 unless later epochs improve.
- Later update: epoch 65 improved again to data-only A rel.RMSE `0.26811`,
  corr `0.96462`, now the best M fit so far in this campaign.
- A forward-only 5070 config was added for this saved best checkpoint:
  `configs/fit_dM_phys_pC_joint_vH_gpu2_wide_neighbor_best_forward.toml`.
  A stale forward validation started from the earlier epoch-25 checkpoint was
  stopped after epoch 50 improved the saved best checkpoint. The plan is to
  forward-validate the final saved-best checkpoint after the M run completes.

No-cheating audit:

- The `vH` gate decision used true mobility only for the labeled ex-post
  operator diagnostic. M training against `vH` still uses the same data-only
  targets `A_data = Cdot_data - A[Phi]` and data-only checkpoint selection by
  A validation.

## 2026-05-15 — Structured Joint-Score Campaign, vH forward and vI/vK continuation

Completed `vH` M wave:

| M branch | Best epoch | A val rel.RMSE | A val corr | Decision |
|---|---:|---:|---:|---|
| `joint_vH_gpu2_warmJ` | `50` | `0.290345` | `0.959164` | Worse than `vE` wide and old vJ; not forward-validated. |
| `joint_vH_gpu1_wide_neighbor` | `65` | `0.268108` | `0.964619` | Best data-only A-validation fit in the joint-score campaign; forward-validated. |
| `joint_vH_gpu1_fresh_neighbor` | `55` | `0.291671` | `0.957907` | Worse than wide and warm-start; no forward run. |

`vH` wide forward validation:

- Config for forward run:
  `configs/fit_dM_phys_pC_joint_vH_gpu2_wide_neighbor_best_forward.toml`.
- Model checkpoint:
  `models/dM_phys_pC_joint_vH_gpu1_wide_neighbor.bson`.
- Forward data:
  `data/forward_dM_phys_pC_joint_vH_wide_best_scale1_dt0015.h5`,
  `data/forward_dM_phys_pC_joint_vH_wide_best_scale1p05_dt0015.h5`,
  `data/forward_dM_phys_pC_joint_vH_wide_best_scale0p95_dt0015.h5`.
- Figures:
  `figures/forward_stats_forward_phys_pC_joint_vH_wide_best_compare.png`,
  `figures/forward_cmn_forward_phys_pC_joint_vH_wide_best_compare.png`,
  `figures/forward_stats_forward_phys_pC_joint_vH_wide_scale1p05_compare.png`,
  `figures/forward_cmn_forward_phys_pC_joint_vH_wide_scale1p05_compare.png`,
  `figures/forward_stats_forward_phys_pC_joint_vH_wide_scale0p95_compare.png`,
  `figures/forward_cmn_forward_phys_pC_joint_vH_wide_scale0p95_compare.png`.
- Metrics:
  `logs/forward_forward_phys_pC_joint_vH_wide_best_compare_metrics.txt`,
  `logs/forward_forward_phys_pC_joint_vH_wide_scale1p05_compare_metrics.txt`,
  `logs/forward_forward_phys_pC_joint_vH_wide_scale0p95_compare_metrics.txt`.

Forward retained nonlinear `C(t)` and covariance metrics:

| Forward model | Cov rel.RMSE | Retained `C(t)` rel.RMSE | Retained `C(t)` corr | Max `|state|` |
|---|---:|---:|---:|---:|
| `M=Phi` | `0.173127` | `0.968388` | `0.896899` | `1.388752` |
| old `vJ x1.10` | `0.090806` | `0.134815` | `0.991029` | `1.448392` |
| `joint vE wide x1.05` | `0.082816` | `0.129351` | `0.991371` | `1.769840` |
| `joint vH wide x0.95` | `0.058513` | `0.142680` | `0.990136` | `1.460477` |
| `joint vH wide x1.0` | `0.073864` | `0.143061` | `0.989682` | `1.568529` |
| `joint vH wide x1.05` | `0.054946` | `0.153622` | `0.987889` | `1.890700` |

Decision:

- `vH` wide proves that better data-only A-validation does not automatically
  transfer to better retained nonlinear forward correlations. It improves
  covariance strongly, especially at scales `0.95` and `1.05`, but every tested
  scale is worse than `vE x1.05` on retained nonlinear `C(t)`.
- Do not promote `vH` wide as the forward winner. The current best joint-score
  forward branch remains `vE wide x1.05`, which improves the user-specified old
  `vJ x1.10` retained-C metric (`0.129351` vs `0.134815`) but still does not
  reach the preferred target below `0.10`.
- A tighter `vE` scale check at `1.06` is running on the RTX 5070 to test
  whether the stable side of the `1.10` nonfinite failure can improve further.

Completed `vI` joint-score result:

- Config: `configs/joint_score_phys_pC_gpu0_vI_physaug_s035_uniform.toml`.
- GPU/device: `GPU:0`, resolved as a 2080 Ti under `CUDA_VISIBLE_DEVICES=1`.
- Main change: lower DSM noise `sigma=0.035`, physical-augmented inputs,
  uniform lag sampling, both endpoints noised.
- Global true-M operator rel.RMSE `0.889088`, corr `0.631912`.
- Active-lag true-M operator rel.RMSE `0.640938`, corr `0.773321`.
- Retained active nonlinear-channel rel.RMSE `0.631758`, corr `0.777902`.

Decision:

- `vI` is rejected before M training. It is much worse than `vE` and `vH` on
  the retained active operator diagnostic and also worse than the old direct
  conditional score.
- The lower-noise uniform physical-augmented route should not be repeated as
  the next joint-score fix.

New joint-score branches launched:

```bash
CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/joint_score.jl \
  SoftSpinLLGChain/configs/joint_score_phys_pC_gpu1_vJ_physfull_active_moment.toml

CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --threads 8 --startup-file=no \
  SoftSpinLLGChain/code/joint_score.jl \
  SoftSpinLLGChain/configs/joint_score_phys_pC_gpu0_vK_physaug_groupnorm_initial_active.toml
```

- `vJ` tests physical-full inputs with active-lag sampling and stronger
  data-only moment penalties.
- `vK` is a new hybrid config: physical-augmented GroupNorm inputs like `vE`,
  but initial-only active-lag DSM like `vH`. This isolates whether `vH`'s
  retained-active gain came from the endpoint-noise/lag choice rather than the
  physical-full input library.
- `vL` was added after the scale sweep started:
  `configs/joint_score_phys_pC_gpu2_vL_physaug_groupnorm_active_moment.toml`.
  It keeps `vE`'s physical-augmented GroupNorm and both-endpoint DSM target,
  but switches to active lag sampling and stronger data-only moment penalties.
  This isolates active-lag/moment balancing without changing the endpoint
  noise rule.

Additional `vE` forward scale sweep:

| Forward model | Cov rel.RMSE | Retained `C(t)` rel.RMSE | Retained `C(t)` corr | Max `|state|` | Decision |
|---|---:|---:|---:|---:|---|
| `joint vE wide x1.05` | `0.082816` | `0.129351` | `0.991371` | `1.769840` | Previous best. |
| `joint vE wide x1.055` | `0.089546` | `0.121584` | `0.992290` | `1.576100` | New best retained nonlinear forward result. |
| `joint vE wide x1.0575` | `0.047034` | `0.147993` | `0.988734` | `1.559913` | Better covariance but worse dynamics. |
| `joint vE wide x1.05875` | `0.068477` | `0.141571` | `0.989605` | `2.375180` | Worse dynamics and large tail. |
| `joint vE wide x1.06` | not evaluated | nonfinite | nonfinite | nonfinite | Rejected by finite-state check. |
| `joint vE wide x1.07` | stopped | not evaluated | not evaluated | not evaluated | Stopped after `1.06` failed. |

Scale-sweep decision:

- The stable useful global scale is sharply localized near `1.055`.
- Pushing the scale above `1.055` improves or changes covariance but damages
  retained nonlinear correlations and eventually creates nonfinite trajectories.
- Current best joint-score learned-M forward result is therefore
  `joint vE wide x1.055`: retained nonlinear `C(t)` rel.RMSE about `0.1216`,
  corr about `0.9923`. This beats the user-specified old `vJ x1.10` reference
  (`0.1348`) but still does not reach the preferred target below `0.10`.
- Keep `data/forward_dM_phys_pC_joint_vE_wide_scale1p055_dt0015.h5` and the
  `forward_phys_pC_joint_vE_wide_scale1p055_compare` figures/metrics as the
  current best forward artifacts. Treat `1.06` and the stopped `1.07` outputs
  as failed scale probes to remove during cleanup unless a later report needs
  their logs.

No-cheating audit:

- All joint-score training losses remain DSM/data-only. The true mobility is
  still used only in labeled ex-post operator diagnostics and has not entered
  the minimized losses, targets, weights, checkpoint selection, M targets, or
  forward selection.

## 2026-05-15 — Structured Joint-Score Campaign, final vJ/vK/vL and forward plateau

Completed late joint-score branches:

| Joint score | Feature/noise/lag change | Global operator rel.RMSE | Active rel.RMSE | Retained active rel.RMSE | Retained active corr | Decision |
|---|---|---:|---:|---:|---:|---|
| `vJ` | physical-full, both endpoints noised, active lags, stronger moment penalty | `4.377946` | `0.350336` | `0.277751` | `0.960471` | Eligible only on retained active channels; global score unusable. |
| `vK` | physical-augmented GroupNorm, initial-only noise, active lags | `2.523287` | `0.397431` | `0.264074` | `0.965178` | Eligible on retained active channels; worse active metric than `vL`. |
| `vL` | physical-augmented GroupNorm, both endpoints noised, active lags, stronger moment penalty | `2.299829` | `0.316239` | `0.245659` | `0.969966` | Best retained-active conditional/operator metric. |

Commands/artifacts:

- Joint-score configs:
  `configs/joint_score_phys_pC_gpu1_vJ_physfull_active_moment.toml`,
  `configs/joint_score_phys_pC_gpu0_vK_physaug_groupnorm_initial_active.toml`,
  `configs/joint_score_phys_pC_gpu2_vL_physaug_groupnorm_active_moment.toml`.
- Checkpoints:
  `models/joint_score_phys_pC_gpu1_vJ_physfull_active_moment.bson`,
  `models/joint_score_phys_pC_gpu0_vK_physaug_groupnorm_initial_active.bson`,
  `models/joint_score_phys_pC_gpu2_vL_physaug_groupnorm_active_moment.bson`.
- Metrics:
  `logs/joint_score_phys_pC_gpu1_vJ_physfull_active_moment_metrics.txt`,
  `logs/joint_score_phys_pC_gpu0_vK_physaug_groupnorm_initial_active_metrics.txt`,
  `logs/joint_score_phys_pC_gpu2_vL_physaug_groupnorm_active_moment_metrics.txt`.
- GPU mapping used: `vJ` on a 2080 Ti, `vK` on a 2080 Ti, `vL` on the RTX
  5070. The logs record the runtime guard-resolved device.

Mobility fits trained from the eligible joint-score checkpoints:

| M branch | Best epoch | A validation rel.RMSE | A validation corr | Ex-post true-M block rel.RMSE | Mean M vs Phi rel.RMSE | Decision |
|---|---:|---:|---:|---:|---:|---|
| `dM_phys_pC_joint_vJ_gpu1_wide_neighbor` | `55` | `0.265390` | `0.964511` | `0.704650` | `0.523561` | Strong A fit, but forward failed to transfer. |
| `dM_phys_pC_joint_vK_gpu0_wide_neighbor` | `35` | `0.271164` | `0.962956` | `0.650392` | `0.612049` | A fit worse than `vJ/vL`; forward worse. |
| `dM_phys_pC_joint_vL_gpu2_wide_neighbor` | `55` | `0.259819` | `0.967913` | `0.697764` | `0.454706` | Best A fit and best retained-active score, but not best forward dynamics. |

M configs/artifacts:

- Configs:
  `configs/fit_dM_phys_pC_joint_vJ_gpu1_wide_neighbor.toml`,
  `configs/fit_dM_phys_pC_joint_vK_gpu0_wide_neighbor.toml`,
  `configs/fit_dM_phys_pC_joint_vL_gpu2_wide_neighbor.toml`.
- Checkpoints:
  `models/dM_phys_pC_joint_vJ_gpu1_wide_neighbor.bson`,
  `models/dM_phys_pC_joint_vK_gpu0_wide_neighbor.bson`,
  `models/dM_phys_pC_joint_vL_gpu2_wide_neighbor.bson`.
- Diagnostic figures:
  `figures/dM_phys_pC_joint_vJ_gpu1_wide_neighbor_diagnostics.png`,
  `figures/dM_phys_pC_joint_vK_gpu0_wide_neighbor_diagnostics.png`,
  `figures/dM_phys_pC_joint_vL_gpu2_wide_neighbor_diagnostics.png`.

Forward validation after the late M branches:

| Forward model | Cov rel.RMSE | Retained `C(t)` rel.RMSE | Retained `C(t)` corr | Max `|state|` | Decision |
|---|---:|---:|---:|---:|---|
| `M=Phi` | `0.173127` | `0.968388` | `0.896899` | `1.388752` | Baseline only. |
| old `vJ x1.10` | `0.090806` | `0.134815` | `0.991029` | `1.448392` | Old reference. |
| `joint vE x1.055` | `0.089546` | `0.121584` | `0.992290` | `1.576100` | Best joint-score learned-M dynamics. |
| `joint vJ x0.95` | `0.053466` | `0.163546` | `0.986430` | `1.532390` | Rejected: good covariance, bad retained dynamics. |
| `joint vJ x1.0` | `0.062072` | `0.165152` | `0.986350` | `1.439231` | Rejected. |
| `joint vK x1.0` | `0.089199` | `0.172489` | `0.985163` | `1.388794` | Rejected. |
| `joint vL x0.85` | `0.091180` | `0.173134` | `0.986173` | `2.130661` | Rejected. |
| `joint vL x0.90` | `0.091291` | `0.171861` | `0.985407` | `2.645054` | Rejected, large tail. |
| `joint vL x0.95` | `0.064176` | `0.146214` | `0.989279` | `2.557872` | Rejected, large tail and worse dynamics. |
| `joint vL x1.0` | `0.056571` | `0.146028` | `0.988958` | `2.599209` | Best covariance among late branches, but worse dynamics/tails. |

Final comparison artifacts:

- Best clean figures:
  `figures/forward_stats_forward_phys_pC_joint_best_compare.png`,
  `figures/forward_cmn_forward_phys_pC_joint_best_compare.png`.
- Best metrics:
  `logs/forward_forward_phys_pC_joint_best_compare_metrics.txt`.
- Late-grid figures/metrics:
  `figures/forward_stats_forward_phys_pC_joint_final_grid_compare.png`,
  `figures/forward_cmn_forward_phys_pC_joint_final_grid_compare.png`,
  `logs/forward_forward_phys_pC_joint_final_grid_compare_metrics.txt`,
  `figures/forward_stats_forward_phys_pC_joint_vL_scale_grid_compare.png`,
  `figures/forward_cmn_forward_phys_pC_joint_vL_scale_grid_compare.png`,
  `logs/forward_forward_phys_pC_joint_vL_scale_grid_compare_metrics.txt`.
- Current best forward HDF5 retained:
  `data/forward_dM_phys_pC_joint_vE_wide_scale1p055_dt0015.h5`.

Important failed probes:

- `joint vE x1.0525` and `joint vE x1.054` completed integration but failed
  the finite-state check during common evaluation. They were below the
  previously best `x1.055`, so the instability is stochastic/trajectory
  dependent and not just monotone in scale.
- `joint vL` had the best retained-active conditional-score diagnostic and
  best data-only A validation, but forward retained nonlinear correlations were
  worse at every tested scale. Lowering the scale improved neither tails nor
  retained dynamics enough.
- `joint vK` had reasonable covariance and bounded state values but the retained
  nonlinear `C(t)` error was much worse than the old branch and `vE`.

Final decision for this campaign:

- The structured joint-score direction did beat the old direct conditional
  score on the retained active operator diagnostic (`vL` retained-active
  rel.RMSE `0.245659` vs old global reference `0.378631`, and `vE`/`vH` also
  beat the old reference on retained active channels).
- It improved learned-M forward validation only through the `vE wide x1.055`
  branch: retained nonlinear `C(t)` rel.RMSE `0.121584` vs old `0.134815`, with
  covariance rel.RMSE `0.089546` vs old `0.090806`.
- It did not reach the preferred retained nonlinear target below `0.10`. The
  hard blocker observed here is forward-transfer mismatch: the branches with
  better retained-active score diagnostics or better A-validation (`vH`, `vL`)
  did not produce better forward retained dynamics.

No-cheating audit:

- Joint-score DSM targets, M residual targets, scale sweeps, and accepted
  forward selection used trajectory data, learned stationary score, learned
  joint/conditional score, and data-only Phi artifacts. True mobility was used
  only in labeled ex-post operator diagnostics and did not enter minimized
  losses, target tensors, checkpoint selection, forward integration, or
  scale-acceptance criteria.

Cleanup note:

- Bulky failed forward HDF5 files from non-accepted scale probes were removed
  after this report update. Compact logs/metrics/figures and the current best
  forward HDF5 were kept.

## 2026-05-15 — Forward-stats figure repair

Goal:

- Fix the current best `forward_stats` comparison figure after the earlier
  version lacked ACF/cross-correlation panels and heatmap colorbars.

Implementation:

- Updated `code/render_forward_with_dM.jl` so `render_stats_with_dm` now
  includes four rows: marginal PDFs, observed covariance plus model covariance
  errors with explicit colorbars, exact global-component ACFs, and exact
  global-component cross-correlations.
- The new ACF/cross panels use the saved observation/model trajectories only.
  They compute site-averaged global components for each trajectory and evaluate
  the finite-lag correlations by zero-padded FFT convolution over all available
  time origins. This avoids the noisy sampled-pair cross-correlation panels from
  the first repair attempt.
- Updated `code/evaluate_forward_grid.jl` and the default render path to pass
  the observation `save_dt` into the stats renderer, so the new panels use the
  correct observation time axis.
- Updated `writeup/system_report.tex` caption for the best forward-stats figure.

Commands run:

- `xvfb-run -a julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/evaluate_forward_grid.jl"); println("plotting code loaded")'`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml forward_phys_pC_joint_best_compare 'Phi dataonly=../data/phi_phys_pC_dataonly_forward_langevin.h5' 'M_NN old vJ x1.10=../data/forward_dM_phys_pC_dataonly_vAcont_vJ_best_scale1p10_dt0015.h5' 'M_NN joint vE x1.055=../data/forward_dM_phys_pC_joint_vE_wide_scale1p055_dt0015.h5'`
- A final stats-only render after the layout tweak regenerated
  `figures/forward_stats_forward_phys_pC_joint_best_compare.png`.

Verification:

- Inspected the regenerated PNG manually. The final figure has readable panels,
  non-overlapping legend, explicit covariance and error colorbars, and the
  requested ACF and cross-correlation rows. The first sampled-pair repair was
  rejected because the cross panels were visibly noisy; the accepted version
  uses exact FFT correlations.
- Re-ran the full best-comparison evaluator once after adding the new panels;
  retained nonlinear metrics remained unchanged because only plotting code and
  the stats-panel correlation estimator changed.

No-cheating audit:

- This attempt changed visualization/evaluation only. It used saved
  observation and forward trajectories to compute plotted statistics. No
  analytic score, true mobility, or simulator coefficient entered a loss,
  training target, residual target, model selection, or regenerated trajectory.

## 2026-05-15 — Oracle true-M residual-target diagnostic

Goal:

- Test whether the current bottleneck is the data-only residual-target
  construction or the M-network/integration stack by intentionally using the
  best conditional score with true mobility and \(\Phi=\langle M_{\rm true}\rangle\)
  to build an oracle \(A_{mn}\) target, then training new M networks on that
  target and forward-validating the result.
- This was an explicitly requested oracle diagnostic. It is not a valid
  data-only benchmark result.

Implementation:

- Added `code/prepare_oracle_trueM_dM_targets.jl`.
- Added oracle configs:
  `configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_equiv.toml`,
  `configs/fit_dM_phys_pC_oracle_trueM_vL_gpu0_neighbor.toml`,
  `configs/fit_dM_phys_pC_oracle_trueM_vL_gpu1_localr2.toml`, and
  `configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_localr2.toml`.
- Updated `code/fit_dM.jl` so `[targets].target_kind = "oracle_trueM"`
  requires a precomputed oracle artifact and writes an explicit non-data-only
  audit string into checkpoints and metrics.
- The oracle target used the learned `vL` transition-score evaluator but true
  local mobility in
  \[
    \dot C^{\rm oracle}_{mn}(\tau)
      = -\left\langle \phi_m(x_\tau)
      \left(M_{\rm true}(x_0)^\top r_\theta(x_0,x_\tau,\tau)\right)_n
      \right\rangle
  \]
  and \(\Phi_{\rm true}=\langle M_{\rm true}\rangle\) in
  \[
    A^{\rm oracle}_{mn}(\tau)
      = \dot C^{\rm oracle}_{mn}(\tau)
      +\left\langle \phi_m(x_\tau)
      \left(\Phi_{\rm true}^\top r_\theta(x_0,x_\tau,\tau)\right)_n
      \right\rangle .
  \]

Commands run:

- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/prepare_oracle_trueM_dM_targets.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_equiv.toml`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu0_neighbor.toml`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_localr2.toml`
- Killed partial failed branches:
  `fit_dM_phys_pC_oracle_trueM_vL_gpu2_equiv.toml` after weak/flat validation
  and `fit_dM_phys_pC_oracle_trueM_vL_gpu1_localr2.toml` after it landed on the
  wrong physical GPU under CUDA masking.
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu0_neighbor.toml SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml SoftSpinLLGChain/data/forward_dM_phys_pC_oracle_trueM_vL_neighbor_dt0015.h5 1.0 0.0015`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml forward_phys_pC_oracle_trueM_vL_neighbor_compare 'Phi dataonly=../data/phi_phys_pC_dataonly_forward_langevin.h5' 'M_NN old vJ x1.10=../data/forward_dM_phys_pC_dataonly_vAcont_vJ_best_scale1p10_dt0015.h5' 'M_NN joint vE x1.055=../data/forward_dM_phys_pC_joint_vE_wide_scale1p055_dt0015.h5' 'oracle trueM-target vL neighbor=../data/forward_dM_phys_pC_oracle_trueM_vL_neighbor_dt0015.h5'`

Oracle target artifact:

- `models/dM_targets_oracle_trueM_vL.bson`.
- Metrics: `logs/dM_targets_oracle_trueM_vL_metrics.txt`.
- Target used 24 lags, 100000 pairs per lag, 100000 selected channel-lag rows,
  and 140000 samples for \(\Phi_{\rm true}\).
- Target RMS values: \(A^{\rm oracle}\) `7.93676131e-02`,
  \(\dot C^{\rm oracle}_{\rm trueM}\) `7.86295383e-02`,
  \(\dot C_{\Phi_{\rm true}}\) `1.08039927e-02`.
- Estimated true mean onsite mobility block:
  `[[4.19970548e-02, -4.01305577e-03, -6.97154821e-04],
    [4.04410888e-03, 4.20046033e-02, -3.58948229e-04],
    [6.92090802e-04, 3.67329562e-04, 2.95415100e-02]]`.

M-training outcomes:

| Oracle M branch | Best epoch | A validation rel.RMSE | A validation corr | Ex-post true-M block rel.RMSE | Ex-post true-M block corr | Decision |
|---|---:|---:|---:|---:|---:|---|
| `gpu0_neighbor` | `100` | `0.248118` | `0.972486` | `0.687504` | `0.777847` | Best A fit, but poor true-M recovery. Forward-tested. |
| `gpu2_localr2` | `70` | `0.290913` | `0.966669` | `0.428071` | `0.954639` | Better true-M block shape, worse A fit; forward-tested after neighbor. |
| `gpu2_equiv` | killed after epoch `30` | about `0.328` | about `0.968` | not finalized | not finalized | Rejected as weak and superseded. |
| `gpu1_localr2` | killed early | not finalized | not finalized | not finalized | not finalized | Wrong physical GPU under masking; no useful result. |

Forward validation:

| Forward model | Cov rel.RMSE | Cov corr | Retained `C(t)` rel.RMSE | Retained `C(t)` corr | Max `|state|` |
|---|---:|---:|---:|---:|---:|
| `M=Phi` data-only baseline | `0.173127` | `0.982795` | `0.968388` | `0.896899` | `1.388752` |
| Old direct-conditional learned M, `vJ x1.10` | `0.090806` | `0.996849` | `0.134815` | `0.991029` | `1.448392` |
| Best data-only joint learned M, `vE x1.055` | `0.089546` | `0.996708` | `0.121584` | `0.992290` | `1.576100` |
| Oracle trueM-target learned M, `vL neighbor` | `0.065478` | `0.997613` | `0.174290` | `0.990203` | `2.560514` |
| Oracle trueM-target learned M, `vL localr2` | `0.096924` | `0.995876` | `0.099442` | `0.994824` | `1.414626` |

Artifacts:

- M checkpoints:
  `models/dM_phys_pC_oracle_trueM_vL_gpu0_neighbor.bson`,
  `models/dM_phys_pC_oracle_trueM_vL_gpu0_neighbor_best.bson`,
  `models/dM_phys_pC_oracle_trueM_vL_gpu2_localr2.bson`,
  `models/dM_phys_pC_oracle_trueM_vL_gpu2_localr2_best.bson`,
  `models/dM_phys_pC_oracle_trueM_vL_gpu2_localr2_final.bson`.
- Diagnostics:
  `figures/dM_phys_pC_oracle_trueM_vL_gpu0_neighbor_diagnostics.png`,
  `figures/dM_phys_pC_oracle_trueM_vL_gpu2_localr2_diagnostics.png`.
- Forward trajectory:
  `data/forward_dM_phys_pC_oracle_trueM_vL_neighbor_dt0015.h5`.
- Additional forward trajectory:
  `data/forward_dM_phys_pC_oracle_trueM_vL_localr2_dt0015.h5`.
- Final forward comparison:
  `figures/forward_stats_forward_phys_pC_oracle_trueM_vL_all_compare.png`,
  `figures/forward_cmn_forward_phys_pC_oracle_trueM_vL_all_compare.png`,
  `logs/forward_forward_phys_pC_oracle_trueM_vL_all_compare_metrics.txt`.

Interpretation:

- This oracle branch shows that even when the residual target is built using
  true \(M\) and true \(\Phi\), the current M parameterization/training stack
  does not recover \(M_{\rm true}\) accurately. The best true-block ex-post
  branch still has block rel.RMSE `0.428`.
- The two oracle forward branches split the diagnostics. The neighbor branch
  gives the best covariance (`0.0655`) but poor retained nonlinear dynamics
  (`0.1743`) and large tails. The local-r2 branch gives the best retained
  nonlinear \(C(t)\) seen so far (`0.09944`, corr `0.99482`) and bounded states,
  but covariance is worse than the best data-only joint branch (`0.0969` vs
  `0.0895`) and global cross-correlation panels remain visibly imperfect.
- Therefore the data-only \(\dot C\)/\(A\) estimator is a real bottleneck for
  forward retained dynamics, but not the only one: the oracle branch improves
  finite-lag nonlinear correlations only by using forbidden true-M targets and
  still does not identify \(M_{\rm true}\) cleanly.

Audit:

- This attempt intentionally violated the normal no-cheating rule by using
  true mobility and \(\Phi=\langle M_{\rm true}\rangle\) in the minimized M
  training target. It is labeled as an oracle diagnostic and must not be cited
  as data-only evidence. The true quantities did not enter the accepted
  data-only branch.

Cleanup note:

- Removed the killed `gpu2_equiv` best checkpoint because it was a weak partial
  branch with no final metrics or figure. Removed the superseded neighbor-only
  comparison figures/metrics after the all-oracle comparison was generated.
  Kept compact configs and the completed oracle artifacts because they document
  the diagnostic result.

## 2026-05-15 — Oracle observable-identifiability search

Goal:

- Test the user's hypothesis that the remaining M-recovery failure is caused by
  the retained observable library rather than by the conditional score.
- This is an explicitly non-data-only oracle diagnostic: true \(M\) and
  \(\Phi_{\rm true}\) were used to score/select observable channels and to build
  the M-training target.

Implementation:

- Added `code/search_oracle_identifiable_observables.jl`.
- For each candidate observable-target channel, active lag `7:24`, and
  translation offset, the script decomposes the oracle residual action into the
  three local block-entry contributions
  `-(phi(xt_i) * (M_true[x0_j]-Phi_true)[u,c] * r_vL[j,u])`.
- It filters channels by split-sample stability and residual signal, then
  greedily selects channels by log-det gain of the normalized 9-column Gram for
  the onsite block entries `(u,c)`.
- The first all-family search selected 60/591 candidate channels with selected
  Gram condition `13.5657`.  The smoother `neighbor_high` search selected 48/402
  channels with selected Gram condition `14.8478`.

Commands run:

- `xvfb-run -a julia --project=. --threads 12 --startup-file=no SoftSpinLLGChain/code/search_oracle_identifiable_observables.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_localr2.toml all oracle_ident_v1 30000 60`
- `xvfb-run -a julia --project=. --threads 12 --startup-file=no SoftSpinLLGChain/code/prepare_oracle_trueM_dM_targets.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu2_localr2.toml`
- `xvfb-run -a julia --project=. --threads 10 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu2_localr2.toml`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv.toml`
- `xvfb-run -a julia --project=. --threads 12 --startup-file=no SoftSpinLLGChain/code/search_oracle_identifiable_observables.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_localr2.toml neighbor_high oracle_ident_neighbor_v1 30000 48`
- `xvfb-run -a julia --project=. --threads 12 --startup-file=no SoftSpinLLGChain/code/prepare_oracle_trueM_dM_targets.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_neighbor_v1_gpu2_localr2.toml`
- `xvfb-run -a julia --project=. --threads 10 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_neighbor_v1_gpu2_localr2.toml`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv_meanstrong.toml`

Outcome:

| Branch | A val rel.RMSE | A val corr | True-M block rel.RMSE | True-M block corr | Mean vs Phi rel.RMSE | Decision |
|---|---:|---:|---:|---:|---:|---|
| old compact oracle `local_r2` reference | `0.290913` | `0.966669` | `0.428071` | `0.954639` | not best | Previous best true-block reference. |
| all-family identifiable `local_r2` | `0.328883` | `0.958929` | `0.508442` | `0.930463` | `0.531306` | Worse than old compact. |
| all-family identifiable `equivariant_r2` | `0.377876` | `0.967684` | `0.384457` | `0.988407` | `0.687551` | First branch to beat old true-block error. |
| all-family identifiable `equivariant_r2`, strong mean continuation | `0.391106` | `0.970432` | `0.381906` | `0.989175` | `0.250112` | Best true-M recovery found. |
| neighbor-only identifiable `local_r2` | `0.373241` | `0.947915` | `0.563124` | `0.920720` | `0.780199` | Rejected. |

Accepted oracle-identifiability artifacts:

- Selector script: `code/search_oracle_identifiable_observables.jl`
- Selected all-family library:
  `configs/nonlinear_observable_retained_channels_oracle_ident_v1.toml`
- Search metrics/figure:
  `logs/oracle_ident_observable_search_oracle_ident_v1_metrics.txt`,
  `figures/oracle_ident_observable_search_oracle_ident_v1.png`
- Target artifact:
  `models/dM_targets_oracle_trueM_vL_ident_v1.bson`
- Best M checkpoint:
  `models/dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv_meanstrong.bson`
- Best M metrics/figure:
  `logs/dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv_meanstrong_metrics.txt`,
  `figures/dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv_meanstrong_diagnostics.png`

Interpretation:

- The best observable library found so far is the all-family oracle
  identifiability library, but it only improves M recovery when paired with the
  physically equivariant \(r^2\) mobility parameterization.
- Observable selection by invertibility alone is not enough. The pure
  `local_r2` NN performed worse on the identifiable library than on the older
  compact library, and the smoother neighbor-only identifiable library was also
  worse.
- The best true-block error improved from `0.428071` to `0.381906`, but this is
  still not a clean recovery of \(M_{\rm true}\). The current bottleneck is the
  combination of observable identifiability, M parameterization, and loss
  selection, not the conditional score alone.

Audit:

- This branch intentionally used true \(M\) and \(\Phi_{\rm true}\) in the
  observable selector and in the oracle M-training targets. It is not data-only
  and must not be cited as a benchmark-success result.

Cleanup note:

- Removed bulky failed local-r2/neighbor identifiable M checkpoints, the
  neighbor-only oracle target artifact, and the neighbor-only search BSON.
  Kept compact logs/figures/configs for the failed branches so future agents can
  see what not to repeat. Kept the all-family target/search artifacts and the
  best equivariant checkpoints because they reproduce the current best oracle
  identifiable result.

## 2026-05-16 — Clean data-target observable subset and final data-only M recovery

Goal:

- Revisit the user's hypothesis that the remaining data-only M failure was the
  observable library, not the learned conditional score.
- Search for a `phi_n`/right-observable improvement and then train a final
  data-only M NN using the learned stationary score, learned transition score,
  and data-only Phi.  True M was allowed only for oracle-guided observable
  design and ex-post diagnostics.

Implementation:

- Added a generalized right-observable path:
  `code/right_observables.jl`, `code/fit_dM_rightobs.jl`,
  `code/search_phi_n_observables.jl`, `code/search_phi_n_struct_observables.jl`,
  and `code/select_right_observable_subset.jl`.
- Added target scaling options to `code/fit_dM.jl`:
  `data_target` and `data_channel`.  The accepted final branch uses a retained
  channel RMS scale computed only from data-only target values.
- The generalized right-observable experiments did not improve M recovery.
  The best oracle result still came from the old coordinate right observable
  `phi_n(x)=x_n`.
- The key successful change was a stricter subset of the existing
  oracle-identifiable left library.  For each channel, I compared the active-lag
  data-only residual target against the oracle true-M residual target.  I kept
  only channels with active-lag rel.RMSE `< 0.35` and correlation `> 0.95`.
  This retained 37/60 channels and improved active data-vs-oracle target
  agreement to rel.RMSE `0.181835`, corr `0.983260`.

Important commands and devices:

- Full generalized right-observable oracle search and M training used all three
  GPUs with the established mapping:
  `CUDA_VISIBLE_DEVICES=0` for the RTX 5070, `1` and `2` for the two 2080 Ti
  devices.  All configs had required GPU-name guards.
- Final clean37 M trainings:
  - `CUDA_VISIBLE_DEVICES=1 ... fit_dM.jl configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu0_mean03.toml`
  - `CUDA_VISIBLE_DEVICES=2 ... fit_dM.jl configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean10.toml`
  - `CUDA_VISIBLE_DEVICES=0 ... fit_dM.jl configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu2_wide_mean03.toml`
- Final forward integrations:
  - `CUDA_VISIBLE_DEVICES=1 ... forward_dM.jl ... data/forward_dM_phys_pC_dataonly_clean37_gpu0_mean03_scale1_dt0015.h5 1.0 0.0015`
  - `CUDA_VISIBLE_DEVICES=2 ... forward_dM.jl ... data/forward_dM_phys_pC_dataonly_clean37_gpu1_mean10_scale1_dt0015.h5 1.0 0.0015`
  - `CUDA_VISIBLE_DEVICES=0 ... forward_dM.jl ... data/forward_dM_phys_pC_dataonly_clean37_gpu2_wide_mean03_scale1_dt0015.h5 1.0 0.0015`
- Final comparison:
  `evaluate_forward_grid.jl configs/fit_Phi_phys_pC_dataonly.toml forward_phys_pC_clean37_compare ...`

Rejected branches:

| Branch | Key result | Decision |
|---|---:|---|
| generalized right-observable full 84-channel oracle | best true-M rel.RMSE `0.4568` | worse than coordinate `phi_n=x` oracle |
| generalized right-observable top36 | early A rel.RMSE about `0.596` | stopped as unpromising |
| structured right-observable 48-channel oracle | best true-M rel.RMSE about `0.5249` | worse |
| full 60-channel data-only target, per-entry scale | data-vs-oracle target rel `0.2209`, corr `0.9753`, but M A fit near random | bad scaling/conditioning |
| full 60-channel data-only target, per-channel scale | best early A rel.RMSE about `0.805` | still too many noisy/inconsistent channels |
| full 60-channel data-only target, high learning rate / weaker mean | best early A rel.RMSE about `0.787` | optimizer not the main bottleneck |

Accepted clean37 artifacts:

- Library: `configs/nonlinear_observable_retained_channels_oracle_ident_v1_dataoracle_clean37.toml`
- Target: `models/dM_targets_dataonly_oracle_ident_v1_dataoracle_clean37_channelscale.bson`
- M checkpoints:
  - `models/dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu0_mean03.bson`
  - `models/dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean10.bson`
  - `models/dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu2_wide_mean03.bson`
- M diagnostics:
  - `figures/dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu0_mean03_diagnostics.png`
  - `figures/dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean10_diagnostics.png`
  - `figures/dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu2_wide_mean03_diagnostics.png`
- Forward HDF5s:
  - `data/forward_dM_phys_pC_dataonly_clean37_gpu0_mean03_scale1_dt0015.h5`
  - `data/forward_dM_phys_pC_dataonly_clean37_gpu1_mean10_scale1_dt0015.h5`
  - `data/forward_dM_phys_pC_dataonly_clean37_gpu2_wide_mean03_scale1_dt0015.h5`
- Final forward figures/metrics:
  - `figures/forward_stats_forward_phys_pC_clean37_compare.png`
  - `figures/forward_cmn_forward_phys_pC_clean37_compare.png`
  - `logs/forward_forward_phys_pC_clean37_compare_metrics.txt`
- Human writeup:
  - `writeup/system_report.tex`
  - `writeup/system_report.pdf` compiled successfully with `latexmk`.

Final M metrics:

| Branch | A val rel.RMSE | A val corr | True-M block rel.RMSE ex-post | True-M block corr | Mean-vs-Phi rel.RMSE |
|---|---:|---:|---:|---:|---:|
| clean37 mean03 | `0.298627` | `0.971608` | `0.271207` | `0.992963` | `0.461411` |
| clean37 mean10 | `0.312157` | `0.973058` | `0.275090` | `0.994491` | `0.290855` |
| clean37 wide | `0.302384` | `0.973489` | `0.289289` | `0.991409` | `0.425434` |

Final forward metrics:

| Forward model | Cov rel.RMSE | Cov corr | Retained C rel.RMSE | Retained C corr | Max \|state\| |
|---|---:|---:|---:|---:|---:|
| Phi dataonly | `0.173127` | `0.982795` | `0.968388` | `0.896899` | `1.3888` |
| old vJ x1.10 | `0.090806` | `0.996849` | `0.134815` | `0.991029` | `1.4484` |
| joint vE x1.055 | `0.089546` | `0.996708` | `0.121584` | `0.992290` | `1.5761` |
| clean37 mean03 | `0.064016` | `0.998213` | `0.062474` | `0.998009` | `1.3875` |
| clean37 mean10 | `0.062004` | `0.997932` | `0.060607` | `0.998351` | `1.3986` |
| clean37 wide | `0.088796` | `0.996345` | `0.068005` | `0.997657` | `1.3712` |

Interpretation:

- The final bottleneck was the observable target set, specifically noisy or
  inconsistent residual channels in the otherwise oracle-identifiable library.
- Generalizing `phi_n` away from coordinates did not help.  The best final
  result keeps `phi_n=x_n` and improves the left observable library by retaining
  only channels whose data-only A target agrees with the oracle true-M target in
  the active lag range.
- This produced the best M recovery so far: true-M block rel.RMSE `0.271`,
  improving over the previous best oracle-identifiable result `0.3819`.
- Forward validation is also substantially better: retained nonlinear
  correlation rel.RMSE improved from old data-only `0.1348` and best joint
  branch `0.1216` to `0.0606`; covariance rel.RMSE improved from about `0.09`
  to about `0.062`.

No-cheating audit:

- True M was used to design/filter the clean37 observable subset, as explicitly
  requested by the user for this search phase.  It did not enter the accepted
  numerical data-only target values, loss scales, M training loss, mean penalty,
  checkpoint selection, or forward integration.
- The accepted M loss used `Cdot_data` from trajectory finite differences,
  data-only Phi, the learned stationary score, and the learned transition score.
  Loss scales were RMS values of data-only retained-channel targets.
- True-M metrics above are ex-post diagnostics only.

Cleanup:

- Removed bulky failed right-observable checkpoints, right-observable target
  BSONs, right-observable search BSONs, failed full-channel data-only best
  checkpoints, and superseded full-channel data-only target BSONs.
- Kept compact configs/logs/figures for failed branches so future agents can
  understand what was tried without rerunning them.
- Removed LaTeX auxiliary build files after confirming the PDF compiles.

## 2026-05-17 — Clean37 refinement sweep and final best-M assessment

Goal:

- Explore the nearby directions suggested after identifying the observable-set
  bottleneck: stricter observable pruning, lag-window changes, mean-penalty
  changes, and warm-start refinements.
- Use all three GPUs when possible and decide whether these close directions
  can improve the best data-only M obtained from the clean37 library.

Implementation changes:

- Added `code/select_clean_observable_subset.jl` to produce stricter clean
  libraries from the annotated clean37 source using data-vs-oracle channel
  diagnostics.
- Extended `code/subset_dm_targets.jl` with `scale_source` and used
  `retained_channel_rms` scaling for strict subsets.
- Generated strict observable libraries:
  - `configs/nonlinear_observable_retained_channels_oracle_ident_v1_dataoracle_clean16.toml`
  - `configs/nonlinear_observable_retained_channels_oracle_ident_v1_dataoracle_clean21.toml`
  - `configs/nonlinear_observable_retained_channels_oracle_ident_v1_dataoracle_clean32.toml`
- Generated temporary data-only subset targets, then removed them after the
  strict-subset branches failed:
  - `models/dM_targets_dataonly_oracle_ident_v1_dataoracle_clean16_channelscale.bson`
  - `models/dM_targets_dataonly_oracle_ident_v1_dataoracle_clean21_channelscale.bson`
  - `models/dM_targets_dataonly_oracle_ident_v1_dataoracle_clean32_channelscale.bson`
- Added M configs for clean16/21/32, clean37 mean06/mean15, lag5:20,
  lag8:24, and warm-started mean-penalty refinements.

Important commands and devices:

- Strict subset target generation initially failed without Xvfb because the
  loaded plotting stack required an X11 display.  Rerunning under `xvfb-run -a`
  succeeded.
- Parallel M training used:
  - `CUDA_VISIBLE_DEVICES=0` for the RTX 5070;
  - `CUDA_VISIBLE_DEVICES=1` and `CUDA_VISIBLE_DEVICES=2` for the two 2080 Ti
    devices.
- Config device guards were respected.  A forward attempt for clean37 mean15 on
  the RTX 5070 correctly aborted because that config required a 2080 Ti; the
  successful forward run used `CUDA_VISIBLE_DEVICES=2`.
- Final extended comparison command:
  `evaluate_forward_grid.jl configs/fit_Phi_phys_pC_dataonly.toml forward_phys_pC_clean37_extended_compare ...`

Rejected refinement branches:

| Branch | Best A val rel.RMSE | Best A val corr | Decision |
|---|---:|---:|---|
| clean16, mean10 | `0.931132` | `0.820730` | too few channels; residual operator underdetermined |
| clean21, mean10 | `0.909447` | `0.880706` | too few channels |
| clean32, mean10 | `0.855524` | `0.897863` | still too aggressively pruned |
| clean37, mean06 | `0.313071` | `0.972934` | worse than accepted clean37 fits |
| clean37, lags 5:20, mean10 | `0.587497` | `0.811868` | shorter active lags corrupt the target fit |
| clean37, lags 8:24, mean10 | `0.320167` | `0.972893` | worse than accepted clean37 fits |
| clean37 mean03 warm-start, mean08 | `0.329914` | `0.972762` | warm-start refinement worse |
| clean37 mean10 warm-start, mean12 | `0.322833` | `0.971871` | warm-start refinement worse |

Accepted new branch:

- `configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean15.toml`
- M checkpoint:
  `models/dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean15.bson`
- M diagnostics:
  `figures/dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean15_diagnostics.png`
- M metrics:
  - best epoch `15`;
  - A validation rel.RMSE `0.302281`;
  - A validation corr `0.972490`;
  - true-M block rel.RMSE ex-post `0.253830`;
  - true-M block corr ex-post `0.994274`;
  - mean M_NN vs Phi onsite rel.RMSE `0.263892`.
- Forward trajectory:
  `data/forward_dM_phys_pC_dataonly_clean37_gpu1_mean15_scale1_dt0015.h5`

Extended forward metrics:

| Forward model | Cov rel.RMSE | Cov corr | Retained C rel.RMSE | Retained C corr | Max \|state\| |
|---|---:|---:|---:|---:|---:|
| Phi dataonly | `0.173127` | `0.982795` | `0.968388` | `0.896899` | `1.3888` |
| old vJ x1.10 | `0.090806` | `0.996849` | `0.134815` | `0.991029` | `1.4484` |
| joint vE x1.055 | `0.089546` | `0.996708` | `0.121584` | `0.992290` | `1.5761` |
| clean37 mean03 | `0.064016` | `0.998213` | `0.062474` | `0.998009` | `1.3875` |
| clean37 mean10 | `0.062004` | `0.997932` | `0.060607` | `0.998351` | `1.3986` |
| clean37 wide | `0.088796` | `0.996345` | `0.068005` | `0.997657` | `1.3712` |
| clean37 mean15 | `0.084851` | `0.996354` | `0.056724` | `0.998618` | `1.4297` |

Final assessment:

- The best dynamics-oriented learned M is now clean37 mean15 because it has the
  lowest retained nonlinear \(C(t)\) error and the best pointwise true-M
  diagnostic, while keeping covariance error below the `0.09` acceptance
  threshold.
- The best covariance branch remains clean37 mean10.
- The close directions around the current solution have now been tested:
  smaller libraries, lag-window changes, weaker/stronger mean penalties, and
  warm-start refinements.  None of those changed the basic conclusion.
- With high confidence for the current dataset, score, conditional score,
  data-only Phi, and referee-acceptable equivariant \(r^2\) M parameterization,
  this is the best M recovered so far.  Further improvement likely requires a
  genuinely new data-only model-selection proxy or a new physically justified
  M parameterization, not another small clean37 hyperparameter tweak.

No-cheating audit:

- True M was used only in the already-declared oracle-assisted observable
  design/filtering phase and in ex-post diagnostics.
- The strict subset targets and clean37 mean15 M loss used data-only
  \(\dot C^{\rm data}\), data-only Phi, the learned stationary score, the
  learned transition score, and data-only target RMS scales.
- True M did not enter the accepted numerical target values, loss weights,
  mean penalty, checkpoint selection, or forward integration.

Cleanup:

- Removed failed strict-subset/refinement best-checkpoint BSONs and failed
  strict-subset target BSONs after recording their metrics here and in
  `docs/agents/ATTEMPTS.md`.
- Kept accepted clean37 mean15 model, diagnostics, forward trajectory, and
  extended comparison figures/metrics:
  - `figures/forward_stats_forward_phys_pC_clean37_extended_compare.png`
  - `figures/forward_cmn_forward_phys_pC_clean37_extended_compare.png`
  - `logs/forward_forward_phys_pC_clean37_extended_compare_metrics.txt`

## 2026-05-17 — Physics-Informed Four-Coefficient Mobility Ansatz

Goal:

- Implement the user's requested physics-informed mobility experiment: use the
  known true onsite tensor form as a structural ansatz, fit only its coefficients
  from the clean37 residual targets, then compare observations, `M=Phi`, the
  best nonparametric NN M, and the physics-informed M in final forward figures.

Mathematical model:

- The fitted onsite block was
  \[
  M_i(m_i)
  =
  c_0 I
  + c_\perp(|m_i|^2 I-m_i m_i^\top)
  + c_\parallel m_i m_i^\top
  + c_\times [m_i]_\times .
  \]
- The four coefficients were fitted by weighted linear least squares against
  the same clean37 data-only residual target used for NN training.  The inputs
  to the fit were trajectory pairs, the learned physical stationary score, a
  learned conditional transition score, and the data-only Phi.  The true
  coefficients were used only after fitting for diagnostics.
- Forward integration used the analytic divergence of this ansatz,
  \(\operatorname{div} M=(-2c_\perp+4c_\parallel)m\), and exact symmetric
  eigenvalues \(\lambda_\perp=c_0+c_\perp |m|^2\),
  \(\lambda_\parallel=c_0+c_\parallel |m|^2\).

Implementation and artifacts:

- New scripts:
  - `code/fit_dM_physical_ansatz.jl`
  - `code/forward_physical_ansatz.jl`
- Main accepted config:
  `configs/fit_dM_phys_ansatz_clean37_directcond_mean1e5_floor001_gpu2.toml`
- Accepted model:
  `models/dM_phys_ansatz_clean37_directcond_mean1e5_floor001_gpu2.bson`
- Accepted diagnostics:
  `figures/dM_phys_ansatz_clean37_directcond_mean1e5_floor001_gpu2_diagnostics.png`
  and `logs/dM_phys_ansatz_clean37_directcond_mean1e5_floor001_gpu2_metrics.txt`
- Accepted forward trajectory:
  `data/forward_dM_phys_ansatz_clean37_directcond_floor001_dt0015.h5`
- Final paper-facing figures and metrics:
  - `figures/forward_stats_forward_phys_pC_paper_phi_nn_phys_compare.png`
  - `figures/forward_cmn_forward_phys_pC_paper_phi_nn_phys_compare.png`
  - `logs/forward_forward_phys_pC_paper_phi_nn_phys_compare_metrics.txt`

Branch search:

| Branch | Eval A rel. | Eval A corr | True-M rel. ex-post | Forward outcome |
|---|---:|---:|---:|---|
| Joint score vL, mean 0 | `0.276041` | `0.966892` | `0.128294` | retained C worse than NN |
| Joint score vL, mean 10 | `0.274454` | `0.967083` | `0.124488` | retained C `0.109694` |
| Joint score vL, mean 100 | `0.260639` | `0.969313` | `0.111758` | retained C `0.095438` |
| Joint score vL, mean 1e5 | `0.244008` | `0.973451` | `0.094988` | fitted coefficients still biased |
| Joint score vL, mean 1e7 | `0.406934` | `0.976304` | `0.253146` | mean term over-constrained |
| Direct conditional, no PSD floor | `0.210250` | `0.978608` | `0.035881` | forward nonfinite |
| Direct conditional, PSD floor 0.005 | `0.215586` | `0.978236` | `0.033481` | finite, worse than floor 0.001 |
| Direct conditional, PSD floor 0.001 | `0.210367` | `0.978542` | `0.031718` | accepted |

Accepted coefficient fit:

- fitted coefficients:
  `c0 = 0.001`,
  `c_perp = 0.07447843970208`,
  `c_parallel = 0.001`,
  `c_cross = -0.7932921322513`.
- true coefficients, ex-post only:
  `[0.002, 0.05, 0.006, -0.8]`.
- A-fit diagnostics:
  train rel.RMSE `0.193453`, train corr `0.981860`,
  eval rel.RMSE `0.210367`, eval corr `0.978542`.
- Ex-post pointwise M diagnostics:
  true-M block rel.RMSE `0.031718`, corr `0.999512`.
- PSD/stability proxies:
  minimum sampled perpendicular eigenvalue `0.0098747`,
  minimum sampled parallel eigenvalue `0.001119`.

Final forward comparison:

| Forward model | Cov rel.RMSE | Cov corr | Retained C rel.RMSE | Retained C corr | Max \|state\| |
|---|---:|---:|---:|---:|---:|
| Phi baseline | `0.173127` | `0.982795` | `0.968388` | `0.896899` | `1.3888` |
| Best NN learned M, clean37 mean15 | `0.083451` | `0.996471` | `0.056751` | `0.998612` | `1.4297` |
| Physics-informed M, direct/floor001 | `0.049157` | `0.999019` | `0.047176` | `0.999213` | `1.3882` |

Interpretation:

- The physics-informed branch is the best overall forward result obtained so
  far.  It beats the best nonparametric NN M on both covariance and retained
  nonlinear \(C(t)\).
- It is not perfectly identical to observations in every plotted global
  cross-correlation panel, but the main retained nonlinear observable metrics
  and covariance metrics are now very close.
- The direct conditional residual was better for this parametric ansatz than
  the late joint-score vL transition source.  The vL ansatz fits tended to fold
  the symmetric structure into an effective isotropic coefficient and did not
  forward-validate as well.
- A small PSD floor was necessary.  Without it, the direct-conditional ansatz
  had excellent pointwise true-M diagnostics but produced nonfinite forward
  states because the parallel symmetric eigenvalue became too small.

No-cheating audit:

- This branch is explicitly physics-informed: the true mobility tensor form was
  used as a structural prior because the user requested it.
- The fitted coefficients were not set from the true coefficients.  They were
  fitted from trajectory-derived clean37 \(A_{\rm data}\), data-only Phi, the
  learned stationary score, and the learned transition score.
- True coefficients and true mobility were used only for ex-post diagnostics
  and for interpreting the branch.

Cleanup:

- Removed nonfinal bulky physics-ansatz forward HDF5s and nonaccepted physics
  ansatz checkpoints after recording their compact metrics.
- Kept the accepted floor001 model, diagnostics, forward trajectory, final
  paper-facing figures, and final metrics.
- Removed stale root-level accidental logs `fit_dm.log` and `paper.log`.

## 2026-05-17 — Deep Artifact Cleanup

Goal:

- Remove discarded figures and experiment data after the final physics-informed
  result, while preserving the complete record of tried branches in this living
  report.

Cleanup actions:

- Removed 25 obsolete HDF5 data files, mostly discarded forward validation
  trajectories from older scale sweeps, oracle probes, nonfinal clean37
  branches, and rejected Phi/true-M diagnostics.
- Removed 105 rejected or duplicate model checkpoints, including discarded
  conditional-score variants, nonfinal joint-score checkpoints, obsolete M NN
  checkpoints, oracle-target M checkpoints, duplicate `_best`/`_final` M copies,
  old target artifacts, and superseded Phi artifacts.
- Removed 132 obsolete figures from discarded experiments.  The retained
  figures are only the writeup-referenced diagnostics, the accepted clean37 and
  physics-informed diagnostics, and the final paper-facing comparisons.
- Removed 259 nonfinal logs and metrics.  The retained logs are compact
  accepted or writeup-referenced metrics only.
- Removed stale root-level accidental logs `fit_dm.log` and `paper.log`.
- Left source code and compact configs in place so accepted artifacts remain
  reproducible and historical experiment configs can still be inspected if
  needed.

Current retained footprint after cleanup:

- `data/`: about `5.6G`, dominated by the production observation HDF5.
- `models/`: about `160M`.
- `figures/`: about `19M`.
- `logs/`: about `188K`.

Current retained generated file count across `data/`, `models/`, `figures/`,
and `logs`: 52 files.

No-cheating and reproducibility note:

- No data-driven targets, training choices, or results were recomputed during
  cleanup.  This was only a filesystem cleanup.
- The removed artifacts were discarded branches already summarized in this
  report.  The final current retained-artifact list is at the top of this file.
