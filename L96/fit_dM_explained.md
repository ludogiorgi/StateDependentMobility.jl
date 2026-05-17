# `L96/fit_dM.jl` — Detailed Explanation

This document describes what is implemented in [L96/fit_dM.jl](fit_dM.jl) and its
included module [L96/src/learned_mobility_pipeline.jl](src/learned_mobility_pipeline.jl).
It is meant to be read alongside [paper.tex](../paper.tex) (Section 2 *Method*
and the L96 subsection of *Results*) and [AGENTS.md](../AGENTS.md), which
fixes the conventions enforced in this code.

---

## 1. Repository context

The repository implements the score-based mobility-inference method of
`paper.tex` (Section *Method*) for the two systems in the *Results* section:

- **`2D/`** — the 2D affine multiplicative-noise benchmark (already done).
- **`L96/`** — the 40-dimensional stochastic Lorenz–96 with translation-invariant
  correlated additive noise (this directory).

The L96 pipeline runs in this order:

1. [L96/sim.jl](sim.jl) — integrate
   $\mathrm dx_i = f_i(x)\,\mathrm dt + \sqrt 2\,Q^{1/2}\,\mathrm dW_t$ with
   $f_i(x)=x_{i-1}(x_{i+1}-x_{i-2})-x_i+F$, $F=8$, $K=40$, store trajectories
   and statistics in HDF5.
2. [L96/score.jl](score.jl) — train the stationary score
   $s(x)=\nabla\log p_{\text{ss}}(x)$ as a 1D periodic UNet.
3. [L96/joint_score.jl](joint_score.jl) — train the lag-conditioned joint score
   $\nabla_{x_0,x_t}\log p(x_0,x_t,\tau)$ on stationary pairs at lags
   $\tau\in[\tau_{\min},\tau_{\max}]$.
4. [L96/fit_dM.jl](fit_dM.jl) — the script described here.

---

## 2. What `fit_dM.jl` actually does

`fit_dM.jl` is the end-to-end driver that turns the pretrained score networks
into an inferred mobility $M(x)=Q+R(x)$ and validates the resulting reduced
Langevin model. The four high-level stages are:

1. **Validate the conditional-score correlation-derivative identity** of the
   paper, namely
   $$\dot C_{m,n}(t) = -\bigl\langle\phi_m(x_t)\,s_{t|0}(x_t\mid x_0)^\top
   M(x_0)\,\nabla\phi_n(x_0)^\top\bigr\rangle,$$
   on a family of six observables, against a generator-based $\dot C^{\text{data}}$
   target.
2. **Fit a translation-equivariant local antisymmetric reference mobility**
   $M^{\text{ref}}=Q+R^{\text{fit}}$ from the score-form drift identity
   $F(x)=M(x)s(x)+\nabla\cdot M(x)$.
3. **Train a neural state-dependent correction $\delta M_\theta$** on the residual
   correlation constraint (eq. `deltaM_residual_constraint` in the paper), with
   a small translation-equivariant local-window MLP that outputs antisymmetric
   offset coefficients.
4. **Forward-validate** both the constant-$\Phi$ closure and the full
   learned-$M$ reduced Langevin model against the observed univariate / pair
   PDFs, ACFs, lattice cross-correlations and observable channel correlations.

Throughout, the conventions of [AGENTS.md](../AGENTS.md) are enforced:

- the diffusion is constant, $D(x)=Q$, so $M(x)=Q+R(x)$ with $R^\top=-R$;
- everything ($F$, $Q$, $s$, $R$, $\nabla\cdot R$) is expressed in **raw**
  $x$-coordinates — features may use standardized $u_i=(x_i-\mu)/\sigma_x$, but
  every derivative w.r.t. $u$ carries an explicit factor $1/\sigma_x$;
- in the conditional-score correlation identity the $R$-action enters with a
  **minus** sign (because $R^\top=-R$) and $\nabla\cdot R$ does **not** appear
  there — only the score-form drift identity uses the divergence;
- $\dot C^{\text{data}}(t)$ is computed via the generator
  $L\phi=F\cdot\nabla\phi+Q:\nabla\nabla\phi$, not from finite differences /
  spline derivatives of $C(t)$.

---

## 3. File layout

`fit_dM.jl` itself is ~1770 lines; the neural-mobility and forward-validation
machinery lives in `src/learned_mobility_pipeline.jl` (~1930 lines) and is
brought in via `include` near the bottom of `fit_dM.jl`.

The major sections of `fit_dM.jl`:

| Lines | Content |
|---|---|
| 1–82 | imports, package bootstrap, ScoreUNet1D includes, figure-style guard |
| 83–268 | parameter / cache structs (`FitDML96Params`, `LoadedModels`, `PairSampler`, `ReferenceMobilityResult`, `ManagedRunPaths`, …) |
| 270–411 | run-directory management, config copy, run-metadata TOML |
| 413–559 | `load_params`: parse `fit_dM.toml`, validate every field |
| 561–767 | data loading, score evaluation in raw $x$-coordinates |
| 769–875 | L96 drift, standardization, observable batches |
| 877–984 | smoothing splines / data-only smoothing selection (diagnostics) |
| 985–1055 | translation-channel accumulation, generator observables ($L\phi$) |
| 1058–1256 | local antisymmetric reference mobility fit |
| 1258–1429 | the central validation loop: $C$, generator-based $\dot C^{\text{data}}$, and the two predicted $\dot C$ channels |
| 1431–1502 | smoothing wrapper, circulant utilities, RMSE tables |
| 1504–1645 | diagnostics figure, BSON output, metrics text report |
| 1651 | `include("src/learned_mobility_pipeline.jl")` |
| 1653–1769 | `run_pipeline` orchestrator and entry point |

---

## 4. Configuration: `FitDML96Params`

`FitDML96Params` (lines 133–204) is a frozen struct mirroring the four sections
of [`fit_dM.toml`](fit_dM.toml):

- **`[data]`** — paths to the HDF5 trajectory file, the BSONs of the trained
  stationary and joint score networks, and a `burnin_fraction`.
- **`[evaluation]`** — `lag_stride`, `pairs_per_tau`, batch sizes for state /
  score / joint inference, the spline-smoothing grid for $C(t)$ and the lists
  of coordinate / nonlinear separations to evaluate and to plot.
- **`[reference]`** — `reference_mobility ∈ {Q_only, local_antisymmetric_fit}`,
  the antisymmetric pair `offsets`, the `feature_offsets` library, the number
  of fit samples / batch size / ridge.
- **`[mobility_nn]`** — the neural-mobility trainer: enable flag, offsets and
  `window_offsets`, `pairs_per_tau`, `tau_batch_size`, `anchor_states`, epochs,
  learning rate, weight decay, lag-weight power, zero-mean and anchor-RMS
  penalties (with annealing scales), `current_action_penalty`,
  `checkpoint_metric`, validation pair seeds and MLP `widths`.
- **`[forward_validation]`** — Euler–Maruyama dt, save stride, total/burn-in
  time, number of trajectories, common-random-numbers flag, support-clamp
  options, PDF bins / max samples, correlation stride / max time / threshold,
  $C_\phi$ window and stride, bivariate offsets and auxiliary sample cap.
- **`[figure]`** — figure size.
- **`[run]`** — device (`AUTO` / `GPU:i` / `CPU`), seed, runs root, label, notes.

`load_params` calls `require_condition` on every numeric / structural
constraint (e.g. all `mobility_nn.window_offsets` contain 0, all
`mobility_nn.offsets` are a subset, the forward-validation ring lags fit, etc.)
so that bad configs fail fast.

---

## 5. Run management

`managed_run_paths` (lines 318–367) returns a `ManagedRunPaths` containing
absolute paths for every artefact this script produces:

- `run_dir`, `config.toml`, `run_info.toml`,
- the figure directory (`l96_cdot_reference_comparison.png/pdf`,
  `fit_A.png`, `fit_cdot.png`, `fit_training.png`, `fit_mobility.png`,
  `forward_validation_stats.png`, `forward_validation_channels.png`),
- the data directory (`l96_cdot_reference_outputs.bson`,
  `l96_cdot_reference_metrics.txt`, `mobility_model.bson`,
  `forward_validation_metrics.txt`, `forward_validation_artifacts.bson`,
  `forward_validation_trajectories.h5`).

If `fit_dM.toml` already lives inside an existing `runs/run_NNN/`, that run is
re-used; otherwise a new `run_NNN` is created. The source TOML is copied next
to the outputs and a `run_info.toml` records the resolved paths so each run is
reproducible from its own directory.

---

## 6. Loading data and pretrained networks

- `load_state_tensor` (lines 589–604) reads `/trajectories/states` as a rank-3
  tensor `(time, K=40, ntraj)` and `/trajectories/time`, normalising the axis
  order whatever the file convention.
- `load_models` (lines 606–636) loads the BSON checkpoints, moves both networks
  to the chosen device, and reads:
  - `score.sigma`, `joint.sigma` (the DSM noise scales),
  - the per-component `mean` / `std` used during training,
  - `joint.metadata.tau_min` / `tau_max` (the conditional-score training range).
- `build_pair_sampler` (lines 638–657) reads `Q`, `F`, the empirical
  decorrelation time, and computes the lag step grid
  $\tau \in [\tau_{\min},\tau_{\max}]$ in saved-time units; it stores the
  trajectories in memory for fast random pair sampling.
- `sample_pair_batch!` / `sample_state_batch!` (lines 660–686) draw stationary
  $(x_0,x_t)$ pairs from random trajectories / time indices.

---

## 7. Score evaluation in raw $x$-coordinates

The networks are trained on the standardized variable
$u_i=(x_i-\mu)/\sigma_x$. To stay consistent with the AGENTS.md convention,
this script always:

1. standardizes the input to $u$,
2. asks the network for $\hat s_u$ (or $\hat s_{u,t|0}$),
3. divides by `score_std` (componentwise) on the way back to raw $x$.

Three thin wrappers do this:

- `evaluate_stationary_score_x` (lines 718–734) → $s(x)=\hat s_u/\sigma_x$.
- `evaluate_joint_score_x0` (lines 736–756) → $\nabla_{x_0}\log p(x_0,x_t,\tau)$
  by selecting channel 1 of the joint UNet (channel 2 is the $x_t$-score; it
  is unused in this validation).
- `evaluate_conditional_score_x0` (lines 758–767) → the conditional transition
  score
  $$s_{t|0}(x_t\mid x_0) = \nabla_{x_0}\log p_t(x_t\mid x_0)
  = \nabla_{x_0}\log p(x_0,x_t,\tau) - s(x_0).$$

This is the relation derived in *Estimating the conditional transition score
from joint scores* (Appendix B.2 of the paper).

---

## 8. The observable family and the generator target

L96 has translation symmetry, so the script uses six translation-equivariant
families with `K=40` channels each (one per lattice site):

| key | $\phi_i(x)$ in $u$-units (centered) |
|---|---|
| `coord` | $u_i$ |
| `var` | $u_i^2 - \langle u^2\rangle$ |
| `nn1` | $u_i u_{i+1} - \langle\cdot\rangle$ |
| `nn2` | $u_i u_{i+2} - \langle\cdot\rangle$ |
| `adv` | $u_{i-1}(u_{i+1}-u_{i-2}) - \langle\cdot\rangle$ |
| `flux` | $u_i u_{i-1}(u_{i+1}-u_{i-2}) - \langle\cdot\rangle$ |

`compute_standardization_and_observable_means` (lines 780–838) computes
$\mu$, $\sigma_x$ and the five observable means in a multithreaded loop over
post-burn-in samples; `compute_observable_batches` (lines 840–875) evaluates
the centered observables in standardized coordinates on a batch.

`compute_generator_observable_batches` (lines 1003–1056) is the **generator
target** of AGENTS.md *Correction 1*. For an observable $\phi(x)$ in
standardized $u$,
$$L\phi = F(x)\cdot\nabla\phi + Q:\nabla\nabla\phi,$$
where each $\partial_{x_j}=\sigma_x^{-1}\partial_{u_j}$ contributes one
$1/\sigma_x$. Since the family is built from products of $u_j$ at neighbouring
sites, $L\phi$ is fully analytic — the function writes out, per coordinate, the
exact drift- and Hessian-trace contributions for each of the six families.
This produces $(L\phi_m)(x_t)$ on every batch, which is then translation-averaged
against $\phi_n(x_0)$ to yield
$$\dot C_{m,n}^{\text{data}}(t)
= \bigl\langle (L\phi_m)(x_t)\,\phi_n(x_0)\bigr\rangle$$
— the unbiased target the paper actually wants.

---

## 9. Local antisymmetric reference mobility

`fit_local_antisymmetric_reference` (lines 1145–1256) solves a sample-mean
least-squares problem for an ansatz $R(x)$ that is:

- **translation-equivariant** on the ring,
- **local** (action depends only on a finite window of $u$ around each pair),
- **antisymmetric** ($R^\top=-R$ by construction),
- **circulant in expectation** (so its mean value is determined by a single
  profile vector).

For each pair offset $r\in$ `antisymmetric_offsets` and each feature in the
library

$$\bigl\{1\bigr\}\;\cup\;\bigl\{u_{i+s}\bigr\}_{s\in F}\;\cup\;
\bigl\{u_{i+s_a}u_{i+s_b}\bigr\}_{(s_a,s_b)},$$

the helper `local_antisymmetric_feature_action!` evaluates the antisymmetric
combination

$$\beta_{r,k}(x)\,s_{i+r}(x) - \beta_{r,k}(\sigma_{-r}x)\,s_{i-r}(x)
+\text{(divergence terms when requested)}$$

with the divergence computed from explicit derivatives of $u$-features w.r.t.
raw $x$ (every $\partial_u/\sigma_x$). The fit then enforces the score-form
drift identity

$$F(x) - Qs(x) \approx R(x)s(x) + \nabla\cdot R(x)$$

on Monte-Carlo samples via a ridge-regularized normal equation. The feature
means are accumulated in the same pass so the fitted $R$ can be reduced to a
single 40×40 antisymmetric matrix `R` and the active reference becomes
$M^{\text{ref}}=Q+R$. If the system is ill-conditioned the code falls back to
$M^{\text{ref}}=Q$ (the "Q-only" baseline) and reports `cond` and the residual
RMS in the metrics file.

---

## 10. The validation loop: `evaluate_c_and_generator_cdot_channels`

This is the core of the script (lines 1269–1429). For each lag $\tau$ in the
joint-score training range it:

1. Draws `pairs_per_tau` stationary pairs $(x_0,x_t)$ in mini-batches.
2. Standardizes them to $(u_0,u_t)$.
3. Computes the L96 drift $f(x_t)$ analytically and the centered observables
   $\phi(u_t)$ together with their generator counterparts $L\phi(x_t)$.
4. Translation-averages
   $C_{m,n}(\tau)=\langle\phi_m(u_t)\,\phi_n(u_0)\rangle$ and
   $\dot C^{\text{data}}_{m,n}(\tau)=\langle (L\phi_m)(x_t)\,\phi_n(u_0)\rangle$
   for all separations.
5. **Only for $\tau>0$**, computes the conditional score
   $s_{t|0}(x_t\mid x_0)$ and forms the projections per the paper's identity:
   - **Q-only** prediction
     $$\dot C^{\text{ref-Q}}_{m,n}(\tau) = -\langle\phi_m(u_t)\,(Q^\top s_{t|0}/\sigma_x)_{n}\rangle.$$
   - **Local-antisymmetric** prediction (active when
     `reference_mobility=local_antisymmetric_fit` and the fit is stable)
     $$\dot C^{\text{ref-fit}}_{m,n}(\tau) = -\bigl\langle\phi_m(u_t)\,\bigl(Qs_{t|0} - R(x_0)s_{t|0}\bigr)_n/\sigma_x\bigr\rangle.$$
     The $-R(x_0)s_{t|0}$ sign comes from $M^\top=(Q+R)^\top=Q-R$ in the
     conditional-score identity, exactly as AGENTS.md prescribes; **no
     $\nabla\cdot R$ enters this projection**.

These four numbers (one $C$, one $\dot C^{\text{data}}$, two reference
predictions $\dot C^{\text{ref-Q}}$ and $\dot C^{\text{ref-fit}}$) are stored
per family per separation per lag.

After the loop, `smooth_all_channels` produces an additional
spline-derivative version of $\dot C^{\text{data}}$ for diagnostic plots only;
the *target* used by the rest of the pipeline is the generator-based one.

`build_rmse_tables` (lines 1481–1502) returns:

- per-observable RMSE and relative RMSE,
- per-channel RMSE,
- a global RMSE / relative RMSE.

A separate **sign test** (lines 1690–1696) compares RMSE vs. RMSE-with-flipped
sign and warns if the wrong sign is preferred — a cheap guardrail against
score-convention mistakes.

---

## 11. The mean mobility $\Phi$ and circulant profiles

Per Section 2.3 of the paper,
$$\Phi := \langle M\rangle = -\dot C_{x,x}(0^+).$$

In the code (lines 1698–1706):

- `Phi_data_profile = -Cdot_data[:coord_full][1, :]` — the right-derivative at
  $\tau=0^+$ of the **coordinate** correlation, with all 40 ring offsets
  included. This is the data-driven estimate of the circulant profile of
  $\Phi$ in $u$-units.
- `Phi_q = Q/\sigma_x^2` — the Q-only baseline circulant profile.
- `Phi_ref = M^{\text{ref}}/\sigma_x^2` — the reference circulant profile.

`circulant_matrix_from_profile` and `circulant_profile` are the obvious
back-and-forth utilities. The metrics file records `Phi RMSE` and `Phi rel RMSE`
between the data profile and the active reference profile.

---

## 12. The diagnostics figure and outputs

`create_diagnostics_figure` (lines 1571–1578, impl. 1504–1569) lays out a
multi-panel figure: per-family $\dot C(\tau)$ panels overlaying
`Cdot_data` and `Cdot_ref_active`, the $\Phi$-profile comparison
(data / Q-only / fitted), and a heatmap of $Q$.

`save_outputs` (lines 1580–1592) BSONs every quantity (taus, separations,
$C/C_\text{smooth}$, $\dot C^\text{data}/\dot C^\text{data}_\text{spline}$,
$\dot C^{\text{ref-active}}/\dot C^{\text{ref-Q}}/\dot C^{\text{ref-fit}}$,
$\Phi$-matrices, $Q$, $\mu$, $\sigma_x$, RMSE tables, the parsed config and the
reference fit struct).

`write_metrics_report` (lines 1594–1645) writes a human-readable text
summary including the global / per-family / Phi RMSEs, the sign-test result,
the configured vs. active reference mode, the fitted antisymmetric
coefficients (per pair offset), the fit's condition number and residual RMS,
and the input paths.

---

## 13. Neural mobility correction (`learned_mobility_pipeline.jl`)

When `mobility_nn.enabled = true`, `run_learned_mobility_pipeline` (line 1801
of `src/learned_mobility_pipeline.jl`) trains a state-dependent correction
$\delta M_\theta$ to the constant baseline $\Phi := M^{\text{ref}}$. This is
the machine-learning analogue of the paper's
$\delta M$ residual constraint (eq. `deltaM_residual_constraint`).

### 13.1 Training target $A_\text{data}$

Given the constant-mobility prediction
$$\Gamma_\Phi(t) = -\bigl\langle\phi_m(x_t)\,(\Phi^\top s_{t|0}/\sigma_x)_n\bigr\rangle,$$
computed by `evaluate_phi_gamma_channels`, the residual that $\delta M_\theta$
has to explain is
$$A_\text{data}(t) := \Gamma_\Phi(t) - \dot C^\text{data}(t)
\;\;=\;\;\hat{\mathcal A}_{m,n,t}[\delta M_\theta]_\text{target}.$$

`channel_tensor_from_dict` packs all selected channel/separation entries into a
tensor of shape `(nchannels, 1, nlags)` for training.

### 13.2 Network and ansatz

`build_local_offset_model` builds a small MLP (default widths `[64, 64]` with
`tanh` nonlinearities and an identity output head) that maps a window of $u$
values around each lattice site,
$$w_i(x) = \bigl(u_{i+s}\bigr)_{s\in W},$$
to a vector of antisymmetric-offset coefficients $c_r(x)$ for each
$r\in$ `mobility_nn.offsets`. The action on a vector signal $v$ is

$$\bigl[\delta R(x)\,v\bigr]_i
= \sum_r \bigl[c_r(x)\,v_{i+r} - c_r(\sigma_{-r}x)\,v_{i-r}\bigr],$$

implemented in `local_offset_action`. Because the same network is applied at
every site, the resulting $R$ is automatically translation-equivariant on the
ring; because the action is built from shifted differences, it is automatically
antisymmetric in expectation.

The final layer's weights and bias are zeroed (`initialize_local_offset_model!`)
so training starts from $\delta M_\theta\equiv 0$.

### 13.3 Loss

For each lag chunk the predicted $\delta\dot C$ channels are computed by

```
coeffs = MLP(windows)                    # (noffsets, K, B)
delta_signal = -local_offset_action(coeffs, scond, offsets) ./ sigma_x
pred[m,n,t]  = mean(phi_m(x_t) .* delta_signal[shift_n, ...])
```

(`predict_training_chunk`, lines 346–371). The training loss is

$$\mathcal L = \underbrace{\bigl\langle w_{m,n,t}\,\bigl((\text{pred}-A_\text{data})/\text{scale}\bigr)^2\bigr\rangle}_\text{data}
+ \lambda_\text{zm}\,\bigl\|\langle c_r\rangle_\text{anchor}\bigr\|^2
+ \lambda_\text{rms}\,\bigl\langle c_r^2\bigr\rangle_\text{anchor}
+ \lambda_\text{cur}\,\text{(MSE on the reference }R\text{-action on anchor states)}
+ \lambda_\text{wd}\,\|\theta\|^2/N_\theta,$$

annealed via `annealed_penalty_weight`. The zero-mean penalty enforces
$\langle\delta M_\theta\rangle\approx 0$ (eq. `deltaM_mean_constraint_neural`),
so the network only learns the part that is not already encoded in $\Phi$.

### 13.4 Optimizer, validation, checkpointing

- Adam with `mobility_nn.learning_rate` (default $10^{-3}$),
- per-lag chunks of size `tau_batch_size`,
- multiple validation caches drawn with independent seeds
  (`mobility_nn.validation_pair_seeds`) — each evaluation reports normalized
  and physical RMSE, anchor-RMS penalty, mean offset profile, weight $L_2$,
  and current-action nRMSE,
- two checkpoint metrics are supported:
  - `normalized_rmse` — best-validation nRMSE wins,
  - `regularized_objective` — nRMSE plus all the regularizers (the option
    used by the default config).
- The best CPU copy of the model is kept in `best_model` and returned at the
  end of `train_mobility_model`.

### 13.5 Held-out current diagnostics

`evaluate_current_diagnostics` (lines 1012–1060) compares three
reconstructions of the stationary probability current
$g(x) := F(x) - Q\,s(x)$:

- **Phi-only**: $g_\Phi(x) = (\Phi-Q)\,s(x)$,
- **reference**: $g_\text{ref}(x) = R^\text{fit}(x)\,s(x) + \nabla\cdot R^\text{fit}$,
- **learned**: $g_\theta(x) = \Phi\,s(x) + \delta R_\theta(x)\,s(x)
  + \nabla\cdot\delta R_\theta - Q\,s(x)$.

The divergence $\nabla\cdot\delta R_\theta$ is computed by hand (function
`local_coefficients_and_divergence`, lines 970–994), via the analytic Jacobian
of the MLP w.r.t. its window inputs (`manual_chain_output_and_jacobian`,
restricted to `tanh`/`identity` activations), with all derivatives in raw $x$
($\sigma_x^{-1}$ factors included). RMSE-vs-$g$ is reported per coordinate and
globally for all three reconstructions.

### 13.6 Figures and outputs (fit stage)

- `family_comparison_figure` — A-channel and $\dot C$-channel multipanel plots
  showing data / reference / learned overlays, per family, with RMSE in the
  panel titles.
- `training_diagnostics_figure` — training loss, validation RMSE curves,
  mean learned offset profile per epoch, final per-channel nRMSE bar plot, and
  text summary panels.
- `mobility_summary_figure` — circulant profiles of $Q$, $\Phi^\text{data}$,
  $M^\text{ref}$ and the learned $M$, plus the held-out current RMSE per
  coordinate.
- `save_mobility_model` — BSONs the trained MLP together with $\Phi$, $\mu$,
  $\sigma_x$, the channel specs, and the full training history.

---

## 14. Forward validation

When `forward_validation.enabled = true`, `run_forward_validation`
(lines 1717–1798 of `learned_mobility_pipeline.jl`) integrates two reduced
SDEs and compares them to the observed statistics:

- **Phi-only model** (Eq. `constant_mobility_closure_method` of the paper):
  $$\mathrm dx = \Phi\,s(x)\,\mathrm dt + \sqrt 2\,\Phi^{1/2}\,\mathrm dW_t.$$
- **Full learned-$M$ model**:
  $$\mathrm dx = \bigl[\Phi\,s(x) + R_\theta(x)\,s(x) + \nabla\cdot R_\theta\bigr]\,\mathrm dt
  + \sqrt 2\,\Phi^{1/2}\,\mathrm dW_t.$$

Both share initial conditions and (optionally) common random numbers, both
use Euler–Maruyama with `forward_validation.dt`, and both run for
`total_time` with `burnin_time` discarded. Optional support clamping
(`clamp_eval_to_support`, `hard_clamp_state`) is available for stability.

After integration (`integrate_forward_validation`, lines 1401–1482) the script
computes:

- **Univariate KDE PDF** on the observed reference grid + KL and RMSE.
- **Bivariate pair PDFs** at the requested ring offsets, on the observed
  bivariate KDE grids.
- **Translation-averaged ACF and shifted lattice cross-correlations**
  (`compute_lattice_correlations_at_steps`), with `t_decorrelation` estimated
  from the ACF envelope.
- **Observable channel correlations** $C_\phi(\tau)$ on the same separations
  used during training, evaluated on rollouts via
  `compute_rollout_channel_correlations`. The mean $C_\phi$ RMSE is reported
  for both the learned-$M$ rollout and the $\Phi$-only rollout.

Outputs:

- `forward_validation_stats.png` — univariate PDFs, ACF, cross-correlations,
  pair PDFs, plus text panels with KL / RMSE / decorrelation / clamp-fraction
  diagnostics and `||sym(Phi)-Q||_F`, $\lambda_\min/\lambda_\max(\text{sym }\Phi)$.
- `forward_validation_channels.png` — per-family lagged correlation overlays.
- `forward_validation_metrics.txt` — flat text dump of every numeric metric.
- `forward_validation_artifacts.bson` — all reference/predicted PDF, ACF,
  cross-correlation, $C_\phi$, current-RMSE arrays.
- `forward_validation_trajectories.h5` — the saved learned-$M$ and Phi-only
  trajectories.

---

## 15. Acceptance criterion and final summary

`agreement_is_acceptable` (line 1647 of `fit_dM.jl`) calls a no-NN run
*acceptable* when the conditional-score sign test passes and both the global
and Phi relative RMSE are below 50 %.

When the NN is enabled, `LearnedPipelineSummary` (lines 35–41 of
`learned_mobility_pipeline.jl`) declares the run *acceptable* when

- the held-out current RMSE of the learned $M$ is below the Phi-only one,
  **and**
- (if forward validation ran) the forward-rollout $C_\phi$ RMSE of the
  learned $M$ is below the Phi-only one.

`run_pipeline` ends by printing the run directory, the global RMSE,
per-family RMSEs, the Phi RMSE and the learned-$M$ summary, and returns
`(run_dir, acceptable)`.

---

## 16. Quick map: paper equation ↔ code

| Paper | Code |
|---|---|
| `score_definition_method` $s(x)=\nabla\log p_\text{ss}$ | `evaluate_stationary_score_x` |
| Conditional score $s_{t|0}=\nabla_{x_0}\log p_t(x_t\mid x_0)$ | `evaluate_conditional_score_x0` (joint − stationary) |
| `central_conditional_score_identity` | `evaluate_c_and_generator_cdot_channels` |
| Generator $L\phi=F\cdot\nabla\phi+Q:\nabla\nabla\phi$ | `compute_generator_observable_batches` |
| Score-form drift identity $F=Ms+\nabla\cdot M$ | `fit_local_antisymmetric_reference` |
| Mean mobility $\Phi=-\dot C_{x,x}(0^+)$ (eq. `Phi_estimator_method`) | `Phi_data_profile = -Cdot_data[:coord_full][1,:]` |
| Residual constraint `deltaM_residual_constraint` | `a_channels_from_gamma_and_cdot` |
| Mean-zero constraint `deltaM_mean_constraint_neural` | zero-mean penalty in `train_mobility_model` |
| Reduced Langevin model `compact_sde_full` | `predicted_full_step!`, `predicted_phi_step!` |

---

## 17. End-to-end: `run_pipeline`

Function `run_pipeline(param_file)` (lines 1653–1760) glues everything
together:

1. parse `fit_dM.toml`, set up the run directory, copy config and write
   metadata,
2. detect / activate the device, load the trajectory HDF5 and the two score
   networks,
3. compute the standardization $(\mu,\sigma_x)$ and observable means,
4. fit the local antisymmetric reference mobility,
5. run the validation loop (`evaluate_c_and_generator_cdot_channels`),
   smooth $C(\tau)$ for diagnostics, build RMSE tables and run the
   conditional-score sign test,
6. compute the data, Q-only and reference $\Phi$-profiles,
7. produce the diagnostics figure and the BSON / text outputs,
8. if `mobility_nn.enabled`, hand off to `run_learned_mobility_pipeline`
   (training + diagnostics + figures + BSON model),
9. if `forward_validation.enabled`, run forward validation,
10. print the human-readable summary and return `(run_dir, acceptable)`.

Invoking the script is

```bash
julia --project=. --threads 36 L96/fit_dM.jl L96/fit_dM.toml
```

If `fit_dM.toml` lives inside an existing `runs/run_NNN/`, that run is reused;
otherwise the next `run_NNN/` is created.
