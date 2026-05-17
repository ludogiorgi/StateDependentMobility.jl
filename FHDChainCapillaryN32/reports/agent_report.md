# FHDChainCapillaryN32 Agent Report

## Status

Created a new `N=32` capillary fluctuating-hydrodynamic chain with physically
selected capillarity. This is a short validation trajectory, not a full
production benchmark dataset.

## Physical parameterization

The model is an isothermal stochastic Korteweg hydrodynamic chain under a heat
bath:

```text
F_kappa = dx sum_i [
  c_s^2 (rho_i log(rho_i/rho0) - rho_i + rho0)
  + m_i^2/(2 rho_i)
] + kappa/(2 dx) sum_i (rho_{i+1}-rho_i)^2 .
```

The capillary strength is no longer a small arbitrary value. It is computed
from a target density correlation length:

```text
xi = density_correlation_length_dx * dx
kappa = xi^2 c_s^2 / rho0
```

For this run:

- `N = 32`
- `L = 2pi`
- `dx = 0.19634954084936207`
- `rho0 = 1`
- `c_s = 1`
- `Theta = 0.005`
- `eta0 = 0.08`
- `zeta = 0.5`
- target `xi/dx = 4`
- `kappa = 0.6168502750680849`

This makes the capillary density correlation length resolved over multiple
cells, which is a physically motivated choice for a coarse finite-volume chain.
The sound speed was reduced from the earlier `c_s=2` trial to `c_s=1` while
holding `xi/dx=4`, so density fluctuations are visible but the trajectory
remains low-Mach and positive.

## Commands

Simulation:

```bash
julia --project=. --threads 16 FHDChainCapillaryN32/code/sim.jl FHDChainCapillaryN32/configs/sim.toml 2>&1 | tee FHDChainCapillaryN32/logs/sim.log
```

Two-panel Hovmoller figure:

```bash
julia --project=. FHDChainCapillaryN32/code/plot_hovmoller.jl 2>&1 | tee FHDChainCapillaryN32/logs/plot_hovmoller.log
```

Training-ready physical validation figure:

```bash
julia --project=. FHDChainCapillaryN32/code/plot_training_ready.jl 2>&1 | tee FHDChainCapillaryN32/logs/plot_training_ready.log
```

On SSH/headless sessions, run GLMakie scripts sequentially. A parallel plotting
attempt failed because two GLMakie processes collided with stale Xvfb lock
files. The display search range in `sim.jl` was expanded from `101:140` to
`101:240`, and the final accepted figures were regenerated sequentially.

## Artifacts

- Config: `FHDChainCapillaryN32/configs/sim.toml`
- Simulation code: `FHDChainCapillaryN32/code/sim.jl`
- Hovmoller plotting code: `FHDChainCapillaryN32/code/plot_hovmoller.jl`
- Training-ready validation code: `FHDChainCapillaryN32/code/plot_training_ready.jl`
- Short dataset: `FHDChainCapillaryN32/data/fhd_chain_capillary_n32_short.h5`
- Summary figure: `FHDChainCapillaryN32/figures/sim_summary.png`
- Dynamics figure: `FHDChainCapillaryN32/figures/sim_dynamics.png`
- Trajectory figure: `FHDChainCapillaryN32/figures/sim_trajectories.png`
- Static structure factor validation: `FHDChainCapillaryN32/figures/static_structure_factor.png`
- Two-panel Hovmoller figure: `FHDChainCapillaryN32/figures/rho_m_hovmoller_5tD.png`
- Training-ready validation figure: `FHDChainCapillaryN32/figures/training_ready_validation.png`

## Static Structure Factor Validation

The code computes

```text
S_rho(n) = <|delta rho_hat_n|^2>
```

from post-burnin density snapshots and compares its shape with the discrete
capillary prediction

```text
S_rho(n) proportional to 1 / [c_s^2/rho0 + kappa lambda_n]
lambda_n = 4 sin^2(pi n/N) / dx^2 .
```

The zero mode is excluded because total mass is conserved. For even `N`, the
Nyquist mode is also excluded from the fitted shape diagnostic.

Accepted short-run structure-factor metrics:

- shape relative RMSE: `1.205082e-2`
- shape correlation: `0.999908`

This is the strongest physical validation in this short run: the density
spectrum follows the Korteweg free-energy prediction.

## Short-Trajectory Checks

The run intentionally uses fallback `T/t_D = 6` with 100 saved snapshots per
pilot decorrelation time:

- pilot `t_D = 50.0` (hit configured pilot maximum)
- production `T = 300.0`
- `save_dt = 0.5`
- saved states shape: `(601, 32, 2, 16)`
- Hovmoller window: `501 x 32` for each channel, trajectory 1, 5 pilot `t_D`

Conservation and positivity:

- minimum saved density: `0.745983`
- max mass drift: `2.247042e-6`
- max momentum drift: `5.778526e-8`
- capillary force sum: `-1.110223e-15`
- capillary score term max error: `7.105427e-15`
- constrained score finite-difference relative error: `4.327865e-9`

Equilibrium and training-readiness checks:

- relative density std `std(rho-rho0)/rho0 = 0.04884493`
- velocity Mach RMS `std((m/rho)/c_s) = 0.15806947`
- empirical/theory density std ratio: `1.00441`
- empirical/theory momentum std ratio: `1.00451`
- channel-normalized inputs are close to standard normal
- normalized-coordinate score target stds:
  `std(sigma_rho*s_rho)=1.70874837`,
  `std(sigma_m*s_m)=0.97930698`

Density spatial correlations from the short post-burnin data:

- offset 1: `0.715594`
- offset 2: `0.473412`
- offset 3: `0.306862`
- offset 4: `0.158982`

This is consistent with choosing `xi = 4 dx`: neighboring densities are now
visibly and statistically coherent in the equilibrium density field. Momentum
spatial correlations remain near zero, as expected from the local kinetic part
of the equilibrium free energy.

## Figure and Parameter Audit

The first `N=32`, `xi=4dx` run used `c_s=2`, which was physically valid but too
stiff for a useful visual/training benchmark: relative density fluctuations were
only about `2.45%`, while momentum remained at its thermal scale. Reducing to
`c_s=1` while recomputing `kappa = xi^2 c_s^2/rho0` preserved the same physical
capillary correlation length and doubled the density fluctuation scale without
leaving the low-Mach regime.

The original summary figure also overlaid raw `rho` and raw `m`, which made the
density PDF look artificially thin. The final summary figure compares
`(rho-rho0)/rho0` and `u/c_s` in the first panel, and the separate
`training_ready_validation.png` figure checks channel-normalized inputs and
normalized-coordinate score targets. That is the relevant view for a U-Net score
model, because Step 2 should train in normalized constrained coordinates.

## Caveat

The pilot decorrelation estimate hit the configured pilot maximum
`pilot_max_decorrelation_time = 50`. The final short production diagnostic still
hits the available post-burnin window at `269.5`. This is acceptable for the
requested short trajectory, Hovmoller, structure-factor, and training-readiness
checks, but this folder should not be presented as a fully decorrelated
production dataset.

For a full benchmark-quality dataset, rerun with a much longer pilot and
production schedule after deciding the final physical parameters.

## No-Cheating Audit

This step is simulation and analytic validation only. No neural network was
trained. The static structure factor comparison is an ex-post validation of the
chosen Korteweg free energy, not a minimized training target.

## Step 2 Stationary Score And Phi Baseline

Status: completed through the Step 2 constant-mobility baseline only. No
conditional score network and no mobility NN were trained.

### Stationary score training

Three stationary score U-Nets were trained in parallel with projected DSM noise
only, using `sigma = 0.05`. Analytic scores were used only in ex-post diagnostic
panels and never as training labels, validation losses, target weights, or model
selection targets.

GPU mapping used:

- `CUDA_VISIBLE_DEVICES=1`: nvidia-smi index 0, RTX 2080 Ti, `score_gpu0.toml`
- `CUDA_VISIBLE_DEVICES=2`: nvidia-smi index 1, RTX 2080 Ti, `score_gpu1.toml`
- `CUDA_VISIBLE_DEVICES=0`: nvidia-smi index 2, RTX 5070, `score_gpu2.toml`

Commands:

```bash
CUDA_VISIBLE_DEVICES=1 julia --project=. FHDChainCapillaryN32/code/score.jl FHDChainCapillaryN32/configs/score_gpu0.toml
CUDA_VISIBLE_DEVICES=2 julia --project=. FHDChainCapillaryN32/code/score.jl FHDChainCapillaryN32/configs/score_gpu1.toml
CUDA_VISIBLE_DEVICES=0 julia --project=. FHDChainCapillaryN32/code/score.jl FHDChainCapillaryN32/configs/score_gpu2.toml
CUDA_VISIBLE_DEVICES=0 julia --project=. FHDChainCapillaryN32/code/score.jl FHDChainCapillaryN32/configs/score_gpu2_render.toml
```

Accepted checkpoint:

- `FHDChainCapillaryN32/models/score_gpu2.bson`
- diagnostics figure: `FHDChainCapillaryN32/figures/score_gpu2_diagnostics.png`
- score-only Langevin cache: `FHDChainCapillaryN32/data/score_gpu2_langevin.h5`

Accepted score diagnostics for `score_gpu2`:

- score-only mean PDF rel.L2: `3.39884796e-2`
- score-only covariance rel.RMSE: `1.31178382e-1`
- conserved mass drift: `9.537e-7`
- conserved momentum drift: `3.636e-7`
- ex-post analytic score rel.RMSE: `3.53135381e-1`
- ex-post analytic score cosine: `0.95720207`
- Stein relative error on full diagnostic pass: `3.32616415e-1`

Selection was by data-only score-only Langevin validation, not by analytic
score error. Later retrains produced one checkpoint with better analytic score
rel.RMSE (`~0.253`) but worse score-only stationary statistics and much worse
Phi-forward validation, so it was rejected.

### High-frequency Phi data

The original short trajectory has `save_dt = 0.5`, which is too coarse to
estimate the right derivative of the coordinate covariance. A high-frequency
continuation was generated from saved stationary states:

```bash
julia --project=. --threads 16 FHDChainCapillaryN32/code/sim_phi_highfreq.jl FHDChainCapillaryN32/configs/sim_phi_highfreq.toml
```

Artifact:

- `FHDChainCapillaryN32/data/fhd_chain_capillary_n32_phi_highfreq.h5`

Shape and cadence:

- states shape: `(4001, 32, 2, 16)`
- `save_dt = 0.005`
- total high-frequency window: `20.0`

This dataset was used only to estimate short-lag coordinate covariance
derivatives from observed trajectories.

### Phi estimation and forward validation

Accepted command:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=. FHDChainCapillaryN32/code/fit_Phi.jl FHDChainCapillaryN32/configs/fit_Phi.toml
```

Accepted artifacts:

- `FHDChainCapillaryN32/data/phi_baseline_artifacts.h5`
- `FHDChainCapillaryN32/data/phi_forward_trajectories.h5`
- `FHDChainCapillaryN32/figures/phi_recovery.png`
- `FHDChainCapillaryN32/figures/phi_forward_comparison.png`
- `FHDChainCapillaryN32/logs/fit_Phi_metrics.txt`

Configuration:

- score checkpoint: `score_gpu2.bson`
- Phi estimator: log-covariance short-lag estimator on normalized constrained
  coordinates
- fit lags: `0.005:0.005:0.1`
- block-circulant projection: enabled
- tangent PSD projection of the symmetric part: enabled
- learned-score Stein correction: enabled with the stable right-acting
  convention used by the accepted forward run
- forward integration: `dt = 5e-4`, `T = 80`, burn-in `20`,
  `save_dt = 0.5`, `24` trajectories

Accepted ex-post Phi diagnostic:

- `Phi` vs `<M_true>` rel.RMSE: `1.7122616139e-1`
- `Phi` vs `<M_true>` correlation: `9.8523955931e-1`
- tangent eig min/max of `sym(Phi)`: about `-1.2e-15 / 9.17`

Accepted forward metrics:

- true-score + `<M_true>` rho PDF rel.L2: `5.1700601979e-2`
- true-score + `<M_true>` m PDF rel.L2: `4.9900827768e-2`
- true-score + `<M_true>` rho ACF rel.L2: `6.1453560919e-2`
- true-score + `<M_true>` m ACF rel.L2: `3.7640628418e-2`
- U-Net-score + data-Phi rho PDF rel.L2: `1.2595998122e-1`
- U-Net-score + data-Phi m PDF rel.L2: `5.1261217674e-2`
- U-Net-score + data-Phi rho ACF rel.L2: `2.5891334302e-1`
- U-Net-score + data-Phi m ACF rel.L2: `1.0683074968e-1`
- learned-vs-true rho PDF rel.L2: `1.2055779988e-1`
- learned-vs-true m PDF rel.L2: `6.3534608639e-2`
- learned-vs-true covariance rel.RMSE: `4.4690177945e-1`

Interpretation: the true-score + `<M_true>` constant closure is good. The
trained U-Net + data-driven Phi reproduces the main marginal PDFs reasonably
and the momentum dynamics well, but the density covariance/spectrum remains too
broad. This should not be reported as "perfect" agreement. It is the best
non-cheating Step 2 result obtained here.

### Failed attempts and lessons

- Coarse-data polynomial Phi from the `save_dt = 0.5` trajectory failed:
  ex-post Phi vs `<M_true>` rel.RMSE was about `1`. Lesson: this system needs
  high-frequency pairs for the derivative at the origin.
- Raw projected Phi without learned-score Stein correction matched `<M_true>`
  better ex-post (`rel.RMSE ~ 0.067`) but made the learned-score forward
  unstable. Lesson: a Phi that is excellent against the analytic diagnostic is
  not sufficient if the learned score violates Stein identities.
- A left-acting Stein correction is the literal matrix orientation for
  `V=-<s x^T>` and improved ex-post Phi slightly, but the learned-score forward
  blew up. It is not the accepted baseline.
- Reducing the forward step to `2e-4` did not remove the density mismatch in a
  short test.
- A data-only Stein normalization of the score output and a channelwise density
  score scaling test did not improve the final three-way comparison.
- High-frequency score retraining plateaued and was stopped. A later
  best-checkpoint short-data retrain improved analytic-score RMSE, but worsened
  score-only Langevin and Phi-forward validation. Selection therefore stayed
  with `score_gpu2`.

### No-cheating audit for Step 2

No analytic score, analytic mobility, generator formula, simulator coefficient,
or true-model tensor entered any minimized training loss, DSM label,
data-driven Phi target, target-derived weight, or checkpoint selection. The
analytic score and `<M_true>` were used only for labeled ex-post diagnostics.
The selected score model was chosen by data-only score-only Langevin
statistics. The selected Phi was estimated from observed short-lag covariance
data and learned-score Stein diagnostics, then compared with `<M_true>` only
after construction.

## Data-Only Phi Estimator Improvement

Status: completed. This is an improved data-only estimate of the constant
mobility `Phi`, optimized for closeness to the short-lag covariance derivative
without using analytic score or true mobility in the estimator. It does not
replace the earlier forward-stabilized `Phi` artifact used in
`phi_forward_comparison.png`; it is the accepted diagnostic-quality data-only
estimate of `<M>`.

Accepted observation design:

```bash
julia --project=. --threads 16 FHDChainCapillaryN32/code/sim_phi_highfreq.jl FHDChainCapillaryN32/configs/sim_phi_burstfine4k.toml
```

This generated many independently initialized short stationary continuations,
rather than one long coarse trajectory:

- `ntrajectories = 4096`
- `t_total = 0.05`
- `save_dt = 0.0005`
- `dt = 5e-5`
- saved shape: `(101, 32, 2, 4096)`
- artifact: `FHDChainCapillaryN32/data/fhd_chain_capillary_n32_phi_burstfine4k.h5`

Accepted estimator command:

```bash
julia --project=. FHDChainCapillaryN32/code/fit_Phi_dataonly.jl FHDChainCapillaryN32/configs/fit_Phi_dataonly.toml
```

Accepted artifacts:

- `FHDChainCapillaryN32/code/fit_Phi_dataonly.jl`
- `FHDChainCapillaryN32/configs/sim_phi_burstfine4k.toml`
- `FHDChainCapillaryN32/configs/fit_Phi_dataonly.toml`
- `FHDChainCapillaryN32/data/phi_dataonly_burstfine4k_sweep.h5`
- `FHDChainCapillaryN32/figures/phi_dataonly_burstfine4k_sweep.png`
- `FHDChainCapillaryN32/logs/fit_Phi_dataonly_burstfine4k_metrics.txt`

The accepted data-only candidate was `poly_L1`, selected by split-half
trajectory stability plus observed small-lag covariance residual. The true
mobility was computed only after ranking all candidates.

The best ex-post row in the accepted sweep was slightly different
(`qv_plus_skew_poly_L1`, rel.RMSE `2.7087e-2`), but it was not selected because
the data-only criterion ranked `poly_L1` first. This distinction is intentional
and should be preserved.

Accepted diagnostics:

- data-only selection score: `4.1267441116e-2`
- split-half relative difference: `4.1148173872e-2`
- observed small-lag residual: `4.7706897432e-4`
- tangent minimum eigenvalue of `sym(Phi)`: `-5.93e-16`
- tangent maximum eigenvalue of `sym(Phi)`: `8.5190422282`
- ex-post `Phi` vs `<M_true>` rel.RMSE: `2.8667971724e-2`
- ex-post `Phi` vs `<M_true>` correlation: `9.9964925953e-1`

The same accepted `Phi` was also compared against `<M_true>` estimated from
the original short dataset and the original high-frequency continuation, not
only from the burst dataset:

- short dataset ex-post rel.RMSE/corr: `2.86676143e-2` / `9.99649265e-1`
- original high-frequency ex-post rel.RMSE/corr:
  `2.86691428e-2` / `9.99649241e-1`

Important failed/intermediate attempts:

- Original high-frequency continuation, `save_dt = 0.005`, selected `poly_L1`
  with ex-post rel.RMSE `1.0129e-1`; raw log-covariance rows were better
  ex-post, but they were not selected by the data-only criterion.
- Finer continuous continuation, `save_dt = 0.0005`, `32` trajectories and
  `T = 5`, still had split-half noise around `0.135` and ex-post rel.RMSE
  `9.396e-2`.
- `512` short bursts reduced the ex-post rel.RMSE to `6.116e-2`.
- `2048` short bursts reduced the ex-post rel.RMSE to `3.271e-2`.
- `4096` short bursts gave the accepted result, with ex-post rel.RMSE
  `2.867e-2`.

Lesson: for this reversible/dissipative chain, the coordinate covariance
derivative is a derivative-at-zero problem. Long trajectories at modest
`save_dt` were finite-lag and variance limited. Many independent, very short
stationary continuations give a much cleaner data-only estimate without using
the analytic score, true mobility, or simulator coefficients in the estimator.

No-cheating audit: candidate ranking used only observed trajectory covariance,
split-half stability, small-lag residuals, conservation projection, and
block-circulant/tangent PSD symmetry projections. `<M_true>` was evaluated only
after all candidates were ranked and saved as an ex-post diagnostic. No
analytic score, true mobility, generator tensor, or simulator coefficient
entered a minimized loss, data-driven target, residual target, weight, or
candidate-selection metric.

## Improved-Phi Forward Regeneration

Status: completed, but the learned-score forward model fails with the raw
improved data-only `Phi`.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 julia --project=. FHDChainCapillaryN32/code/fit_Phi.jl FHDChainCapillaryN32/configs/fit_Phi_improved.toml
```

Artifacts:

- config: `FHDChainCapillaryN32/configs/fit_Phi_improved.toml`
- Phi diagnostic: `FHDChainCapillaryN32/figures/phi_improved_recovery.png`
- forward comparison: `FHDChainCapillaryN32/figures/phi_improved_forward_comparison.png`
- forward trajectories: `FHDChainCapillaryN32/data/phi_improved_forward_trajectories.h5`
- metrics: `FHDChainCapillaryN32/logs/fit_Phi_improved_metrics.txt`

The forward run used the accepted U-Net score checkpoint
`score_gpu2.bson` and the raw improved external `Phi` from
`phi_dataonly_burstfine4k_sweep.h5`. No Stein correction was applied to `Phi`
in this primary regenerated figure.

Raw improved-`Phi` diagnostics:

- ex-post `Phi` vs `<M_true>` rel.RMSE: `2.8667971724e-2`
- ex-post `Phi` vs `<M_true>` correlation: `9.9964925953e-1`
- tangent minimum eigenvalue of `sym(Phi)`: about `-6.44e-16`

Forward result with learned score plus raw improved `Phi`:

- learned covariance rel.RMSE vs observations: `1.0515538684e9`
- learned rho PDF rel.L2: `2.1602732329`
- learned m PDF rel.L2: `2.6733724773`
- learned rho ACF rel.L2: `3.4084822350`
- learned m ACF rel.L2: `5.2923595958`

Interpretation: the raw improved `Phi` is physically and analytically much
better than the earlier forward-stabilized `Phi`, but the accepted learned
score is not compatible with it in forward Langevin integration. The
regenerated comparison figure intentionally exposes this failure. Do not
claim the learned-score/raw-improved-Phi forward model is good.

Failed quick alternatives, selected only by observed forward statistics:

- `score_gpu0.bson` with raw improved `Phi`: stable but poor, learned
  covariance rel.RMSE `3.0036`.
- `score_gpu1.bson` with raw improved `Phi`: stable but poor, learned
  covariance rel.RMSE `4.9534`.
- accepted `score_gpu2.bson` with improved raw `Phi` plus learned-score Stein
  correction to the forward `Phi`: stable but no longer uses the raw improved
  physical `Phi`; effective `Phi` ex-post rel.RMSE worsened to `0.4825`, and
  quick learned covariance rel.RMSE was `0.4762`.

Lesson: improving the data-only estimate of `<M>` does not automatically
improve the learned-score forward model. With a constant mobility close to the
true mobility, the stationary U-Net score error becomes the dominant failure
mode. The earlier accepted forward-stabilized `Phi` worked partly because it
compensated for learned-score Stein error, not because it was a more accurate
estimate of `<M_true>`.

No-cheating audit: the regenerated forward model used only the learned score
checkpoint and the data-only improved `Phi` for the learned trajectory.
`<M_true>` was used only for the separately labeled analytic reference and
ex-post diagnostics. The quick failed alternatives were judged by observed
PDF/covariance/ACF metrics, not by choosing the best `<M_true>` agreement.

## Data-Driven Score Repair

Status: completed under time constraint. The accepted repair is not a U-Net;
it is an empirical Gaussian stationary score estimated from observed covariance
in normalized constrained coordinates, then integrated exactly as an OU process
with the improved data-only `Phi`.

Accepted command:

```bash
julia --project=. FHDChainCapillaryN32/code/fit_Phi_gaussian_score.jl FHDChainCapillaryN32/configs/fit_Phi_gaussian_score.toml
```

Accepted artifacts:

- `FHDChainCapillaryN32/code/fit_Phi_gaussian_score.jl`
- `FHDChainCapillaryN32/configs/fit_Phi_gaussian_score.toml`
- `FHDChainCapillaryN32/data/phi_gaussian_score_forward_trajectories.h5`
- `FHDChainCapillaryN32/figures/phi_gaussian_score_forward_comparison.png`
- `FHDChainCapillaryN32/logs/fit_Phi_gaussian_score_metrics.txt`

Metrics for data Gaussian score + improved data-only `Phi`:

- rho PDF rel.L2: `3.0550567866e-2`
- m PDF rel.L2: `2.6079541432e-2`
- u PDF rel.L2: `3.0712407282e-2`
- covariance rel.RMSE: `1.1206932856e-1`
- rho ACF rel.L2: `2.9808187843e-1`
- m ACF rel.L2: `8.5887292323e-2`

This is the best non-cheating forward result with the accurate improved `Phi`.
The PDFs, spectra, and momentum dynamics are visually excellent in the accepted
figure. The remaining density ACF error is mostly late-lag amplitude mismatch
against a short observed trajectory; it is far smaller and more controlled than
the catastrophic U-Net forward failures.

U-Net repair attempts:

- Three DSM-only U-Net trainings were launched in parallel on all GPUs using
  `score_repair_gpu0.toml`, `score_repair_gpu1.toml`, and
  `score_repair_gpu2.toml`.
- The two RTX 2080 Ti runs were stopped early because the user needed to shut
  down soon and they were not competitive by validation DSM loss.
- The RTX 5070 run completed 80 epochs and saved
  `FHDChainCapillaryN32/models/score_repair_gpu2.bson`.
- The repaired U-Net improved ex-post analytic score diagnostics
  (`rel.RMSE 2.16464071e-1`, cosine `0.97698842`) but still had Stein error
  `3.73749103e-1`.
- Its quick improved-Phi forward check failed: learned covariance rel.RMSE
  `4.1929451432e7`, rho PDF rel.L2 `0.4141`, m PDF rel.L2 `0.4855`.

Lesson: for this near-Gaussian capillary equilibrium, a covariance-based
data-driven score is the right finite-time repair. The U-Net DSM score can look
reasonable against analytic score samples and still be unusable with an
accurate mobility because small global Stein/covariance errors are amplified by
the reversible part of `Phi`.

No-cheating audit: the empirical Gaussian score was estimated only from
observed sample covariance in normalized constrained coordinates. No analytic
score, true mobility, generator formula, or simulator coefficient entered the
score estimator. The improved `Phi` is the previously selected data-only
estimator. `<M_true>` appears only in labeled ex-post diagnostics and the
separate analytic reference trajectory.

## Coarse Original-Cadence Phi Re-Estimation

Status: completed. This experiment intentionally used only the original short
dataset with `save_dt = 0.5`, i.e. the original 100 saved snapshots per pilot
decorrelation time. No high-frequency continuation, burst data, simulator
coefficients, analytic score, or true mobility was used in estimator ranking.

Command:

```bash
julia --project=. FHDChainCapillaryN32/code/fit_Phi_coarse.jl FHDChainCapillaryN32/configs/fit_Phi_coarse.toml 2>&1 | tee FHDChainCapillaryN32/logs/fit_Phi_coarse.log
```

Artifacts:

- `FHDChainCapillaryN32/code/fit_Phi_coarse.jl`
- `FHDChainCapillaryN32/configs/fit_Phi_coarse.toml`
- `FHDChainCapillaryN32/data/phi_coarse_original_sweep.h5`
- `FHDChainCapillaryN32/figures/phi_coarse_original_sweep.png`
- `FHDChainCapillaryN32/logs/fit_Phi_coarse.log`
- `FHDChainCapillaryN32/logs/fit_Phi_coarse_metrics.txt`

Data actually used:

- input dataset: `FHDChainCapillaryN32/data/fhd_chain_capillary_n32_short.h5`
- cadence: `save_dt = 0.5`
- burn-in fraction: `0.10`
- post-burnin normalized state shape: `(541, 32, 2, 16)`
- no resimulation and no added high-resolution pairs

Estimator families tried:

- projected local polynomial covariance derivative, degrees `1:8`
- projected weighted covariance-logarithm fits
- projected VAR(1)-style finite-lag propagator fits followed by matrix log
- projected Euler-generator least squares
- Fourier-mode branch-unwrapped logarithms across wavenumber
- coarse quadratic-variation symmetric estimates plus extrapolated skew parts

Candidate ranking was data-only:

```text
score = split_rel
      + prediction_residual
      + 0.25 * one_step_residual
      + 0.02 * roughness
      + imag_penalty
```

where `split_rel` is odd/even trajectory stability,
`prediction_residual` is held-out saved-cadence lag covariance prediction
error, `one_step_residual` checks the first saved lag, `roughness` applies only
to Fourier branch unwrapping, and `imag_penalty` penalizes non-real logarithm
branches. True mobility was computed only after the table was ranked.

Accepted data-only row:

- estimator: `projected_varlog_L30`
- data score: `4.1270663750e-1`
- split-half relative difference: `3.8034078871e-1`
- held-out prediction residual: `2.5832267684e-2`
- one-step residual: `2.6134324423e-2`
- tangent eig min/max: `-2.19e-16 / 6.7307`
- ex-post `Phi` vs `<M_true>` rel.RMSE: `9.8637353419e-1`
- ex-post correlation: `2.0152866801e-1`
- norm ratio `||Phi|| / ||<M_true>||`: about `0.318`

Best ex-post row, not selected:

- estimator: `coarse_qv_plus_poly_skew_d3_L4`
- data score: `1.0416628966`
- held-out prediction residual: `0.7774220748`
- ex-post `Phi` vs `<M_true>` rel.RMSE: `0.9709634362`
- ex-post correlation: `0.2399718759`
- norm ratio `||Phi|| / ||<M_true>||`: about `0.259`

Important rejected branch:

- High-degree local polynomials can look attractive by split stability alone
  but have much worse held-out lag prediction and worse ex-post Phi. Example:
  `projected_poly_d7_L10` had data score `0.6522` under the final score,
  prediction residual `0.4208`, and ex-post rel.RMSE `1.1371`.

Conclusion:

- The original `save_dt = 0.5` cadence is not sufficient to recover the
  derivative-at-zero mobility for this capillary FHD chain. Better
  extrapolation improves saved-cadence covariance fits, but even the best
  ex-post candidate remains near relative error `0.97`.
- The earlier 4096-burst result (`rel.RMSE 0.02867`) is not merely a better
  fitting algorithm; it uses necessary near-zero-lag information.

No-cheating audit:

- Estimator ranking used only the original trajectory, empirical lag
  covariances, split-half trajectory stability, held-out saved-cadence
  covariance prediction, tangent/block-circulant projections, and data-only
  smoothness/logarithm diagnostics.
- `<M_true>` was evaluated only after all candidates were ranked and was used
  only in ex-post columns and the final failure diagnosis.

## Gaussian-Score Forward Sweep Across Phi Estimation Cadence

Status: completed. This experiment estimated `Phi` at several covariance
cadences, then paired each selected `Phi` with the same empirical Gaussian
stationary score and compared recovery of observed time correlations.

Command:

```bash
julia --project=. FHDChainCapillaryN32/code/fit_Phi_dt_sweep_gaussian.jl FHDChainCapillaryN32/configs/fit_Phi_dt_sweep_gaussian.toml 2>&1 | tee FHDChainCapillaryN32/logs/fit_Phi_dt_sweep_gaussian.log
```

Artifacts:

- `FHDChainCapillaryN32/code/fit_Phi_dt_sweep_gaussian.jl`
- `FHDChainCapillaryN32/configs/fit_Phi_dt_sweep_gaussian.toml`
- `FHDChainCapillaryN32/data/phi_dt_sweep_gaussian_forward.h5`
- `FHDChainCapillaryN32/figures/phi_dt_sweep_gaussian_forward.png`
- `FHDChainCapillaryN32/logs/fit_Phi_dt_sweep_gaussian.log`
- `FHDChainCapillaryN32/logs/fit_Phi_dt_sweep_gaussian_metrics.txt`

Design:

- Gaussian score: empirical covariance score estimated from
  `fhd_chain_capillary_n32_phi_burstfine4k.h5`, ridge `1.0e-8`.
- Forward integration: exact linear Gaussian-score/constant-Phi transition,
  output save interval `0.5`, `2048` trajectories, post-burn horizon `80` with
  `20` burn-in.
- Observation comparison: original short dataset
  `fhd_chain_capillary_n32_short.h5`, post-burnin start index from
  `burnin_fraction = 0.10`.
- `dt = 0.0005`: estimated from `fhd_chain_capillary_n32_phi_burstfine4k.h5`.
- `dt = 0.005`: estimated from `fhd_chain_capillary_n32_phi_highfreq.h5`.
- `dt = 0.05`: estimated from the same high-frequency data subsampled by
  stride `10`.

Data-only selected rows and forward RMSE:

```text
label       Phi dt    estimator  Phi rel.RMSE  rho ACF RMSE  m ACF RMSE  combined ACF RMSE  cov RMSE
dt_0p0005   0.0005   poly_L1    0.02866808    0.28541922    0.08122903  0.20983599         0.11333460
dt_0p005    0.005    poly_L1    0.10128973    0.61660122    0.18607110  0.45542261         0.11166656
dt_0p05     0.05     poly_L1    0.67820124    0.60081339    0.17931871  0.44335760         0.11316188
```

Ex-post best rows in the estimator sweeps:

- `dt_0p0005`: `qv_plus_skew_poly_L1`, rel.RMSE `0.02708727`.
- `dt_0p005`: `logcov_L12`, rel.RMSE `0.06090545`.
- `dt_0p05`: `logcov_L1`, rel.RMSE `0.06126634`.

These ex-post best rows were not used for forward comparison because selecting
them would use true mobility. They are useful diagnostics showing that the
data-only selection rule is not always the same as the ex-post best true-M row
at coarser cadence.

Conclusion:

- With the Gaussian score fixed, the finest `dt=0.0005` Phi gives the best
  observed time-correlation recovery: combined ACF RMSE `0.2098`.
- The coarser `dt=0.005` and `dt=0.05` selected Phi estimates have similar
  stationary covariance RMSE near `0.11`, but much worse time-correlation RMSE
  around `0.44-0.46`.
- This confirms that the near-zero-lag Phi estimate improves dynamics, not only
  ex-post agreement with `<M_true>`.

No-cheating audit:

- Each Phi was selected using observed covariance data at that cadence only.
- The Gaussian score was held fixed across all forward comparisons.
- `<M_true>` was evaluated only after selection, for diagnostic columns and
  interpretation; it did not select the forward models.

### Current next action

Do not proceed to Step 3 without explicit user approval. If Step 3 is approved,
the main warning is that this benchmark is demanding for an approximate
stationary U-Net score under a skew/reversible constant mobility. A conditional
score and learned state-dependent mobility should not be interpreted until the
conditional-score operator diagnostic is strong.
