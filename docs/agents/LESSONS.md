# Agent Lessons

## Repository Agent Memory

- No task-specific reusable lessons have been recorded yet.
- Future agents should add only durable, compact lessons here after an attempt
  reveals a pattern that is likely to matter again.

## Repository Commit Hygiene

- Keep benchmark source, configs, reports, writeup `.tex` files, paper sources,
  BibTeX, and repository documentation visible to Git.
- Keep generated benchmark artifacts ignored by default: `data/`, `models/`,
  `figures/`, `logs/`, `outputs/`, `runs/`, compiled PDFs, HDF5/JLD/BSON
  checkpoints, LaTeX build products, and local prompt scratch notes.
- The local `ScoreUNet1D.jl/` checkout is ignored unless it is intentionally
  converted into a real dependency or submodule.
- Before committing, use `git add -n .` and `git check-ignore -v` spot checks
  to confirm the staged set contains source/documentation only.

## SoftSpinLLGChain Cleanup

- The living report in `SoftSpinLLGChain/reports/agent_report.md` contains the
  key failed-branch metrics and lessons. Do not keep bulky superseded forward
  HDF5 files, failed branch logs, or obsolete comparison figures just to
  document failed attempts.
- Preserve the accepted production data, score/Phi artifacts, final learned-M
  artifacts, compact nonlinear-observable diagnostics, and final comparison
  figures. These are the useful restart points for future work.
- After the 2026-05-17 deep cleanup, the current retained generated artifact
  list is explicitly at the top of `SoftSpinLLGChain/reports/agent_report.md`.
  Treat older file paths in chronological report sections as historical branch
  names unless they also appear in that current retained list.

## SoftSpinLLGChain Plotting

- The `mz` order parameter has slow branch mixing. Short contiguous observation
  windows can be visibly biased toward one spin-inversion branch even though the
  full stationary dataset is symmetric.
- `sim_summary.png` uses all saved states, while `phi_forward_stats.png` uses an
  aligned early post-burn observation window. Do not treat asymmetry in that
  panel as a physical failure without checking the sampled window.
- For stationary PDF/covariance panels, sample from the full post-burn
  observation pool and all saved forward-model states. Keep aligned windows only
  for finite-lag correlation panels.
- Forward-correlation plots should use the shared
  `DEFAULT_FORWARD_CORR_MAX_LAGS` constant from `SoftSpinLLGChain/code/fit_Phi.jl`.
  The current accepted value is `300` lags, about `3 t_D`.
- For learned-M forward validation, `forward_cmn_*` should plot the retained
  nonlinear observable/target-component channels from
  `configs/nonlinear_observable_retained_channels_compact.toml`, not only
  coordinate auto-correlations. The current compact visual diagnostic uses all
  36 retained channel pairs with site-averaged correlations.
- The `forward_stats_*` figure should include PDFs, covariance/error heatmaps
  with explicit colorbars, and exact global-component ACF/cross-correlation
  panels. A sampled-pair version of the global cross-correlations was visibly
  noisy; use the FFT-based all-time-origin estimator in
  `SoftSpinLLGChain/code/render_forward_with_dM.jl`.

## SoftSpinLLGChain Forward Bottlenecks

- Use the full post-burn observation pool for final retained-C metrics as well
  as stationary PDFs. Short aligned observation windows add avoidable sampling
  noise in this slow branch-mixing system.
- True mobility paired with the legacy U-Net learned stationary score was not
  an upper bound: it gave much worse forward statistics than the accepted
  learned M, so that learned M was compensating for score bias. This does not
  apply to the later physical-feature score branch.
- Global mobility scaling and reversible/skew scaling did not improve retained
  Cmn over accepted vX. Skew gains also created large state tails. Do not repeat
  these as the next fix.
- Wider/continued compact lag7 MLP checkpoints improved some training or
  covariance diagnostics but did not beat accepted vX on retained-C forward
  validation. More local MLP capacity is unlikely to be the main path forward.
- In this environment, `CUDA_VISIBLE_DEVICES=0` exposes the RTX 5070, while
  `CUDA_VISIBLE_DEVICES=1` and `CUDA_VISIBLE_DEVICES=2` expose the two 2080 Ti
  cards. The code's printed `nvidia-smi index` can be misleading under CUDA
  masking; verify with `nvidia-smi --query-compute-apps` when parallelizing.

## SoftSpinLLGChain Score/Mobility Attempts

- For strict data-only Phi at the original `t_D/save_dt=100` resolution,
  direct symmetry-reduced onsite covariance fitting reduces dense off-profile
  noise and improved ex-post `<M_true>` rel.RMSE from `0.239` to `0.174`, but
  it did not improve retained forward correlations. Exact all-saved-pair
  polynomial, matrix-log, and increment-covariance fits were worse; do not
  repeat those as the next Phi fix unless new near-zero-lag data are allowed.
- Better DSM or posterior losses did not imply a better conditional-score
  operator. Targeted lag-24 conditional scores had lower posterior DSM MSE but
  worse true-M operator rel.RMSE than accepted vA; do not promote them to
  mobility training without passing the operator diagnostic.
- Continuing the physical-score U-Net conditional residual from vA to
  `vA_cont2` improved the true-M operator only modestly (`0.3949 -> 0.3786`
  rel.RMSE). Warm-started M retraining from vJ with that conditional (`vN`) and
  an earlier-lag variant (`vO`) did not improve A validation or forward metrics.
  Best fully learned-M forward remains `vJ x1.10`.
- Do not repeat the physical-feature conditional MLP route as-is; prior pC
  feature MLP attempts had true-M operator rel.RMSE around `1.7-2.0`. If using a
  feature library for the conditional score, try a stationary-score-style
  ridge/normal-equation residual fit rather than another unconstrained MLP.
- The score vC/data-only-Phi branch worsened retained forward dynamics when
  paired with old or retrained M. The accepted vX model appears tuned to the
  accepted legacy score/Phi/conditional-score stack.
- Under the current stricter no-cheating rule, the old short-lag Phi
  resimulation should be treated as a legacy estimator. A fully trajectory-only
  Phi was tried and was less accurate, and retraining M against it did not
  improve forward validation.
- `fit_dM.jl` now supports an optional `data.cond_score_config` TOML entry.
  This avoids silently evaluating a new conditional-score checkpoint with the
  old hard-coded `cond_score_gpu0_vA.toml` architecture/lag normalization.
- The first structured joint-score wave did not beat the accepted direct
  conditional residual score. Best branch was physical-augmented residualized
  joint score with true-M operator rel.RMSE `0.594637`, versus old conditional
  `0.378631`. Raw joint score and initial-only active-lag DSM were much worse;
  do not promote those checkpoints to mobility training.
- For joint-score gate decisions, compute the retained nonlinear active-lag
  operator metric used by M training. A model can fail the global all-channel
  diagnostic but still be useful on the active retained channels: `vE` had
  global rel.RMSE `0.486413` but retained active rel.RMSE `0.289740` and corr
  `0.957278`.
- The second-wave physical-full joint score `vF` was much worse than the
  physical-augmented GroupNorm `vE` branch, despite a plausible feature set.
  Do not assume `physical_full` is better without checking the operator metric.
- The targeted `vH` physical-full initial-only joint score is a split result:
  globally it is unusable (`4.99` true-M operator rel.RMSE), but on retained
  active nonlinear M-training channels it improved rel.RMSE to `0.257` with
  corr `0.966`. Treat this as an M-training candidate only for the retained
  active target; do not cite it as a good all-channel joint score.
- Better retained active score diagnostics do not guarantee better M fits:
  `vH` warm-start M finished at A rel.RMSE `0.290345`, worse than `vE` wide
  `0.289004`, despite `vH` having the better retained active score diagnostic.
  However, the wider `vH` M branch later reached `0.27556` at epoch 25, so
  capacity/optimization can determine whether the conditional-score advantage
  transfers.
- The first `vE` joint-score forward validation at scale `1.0` only marginally
  improved over old `vJ x1.10` on retained C (`0.134558` vs `0.134815`) while
  improving covariance. Treat this as promising but not solved; run scale
  sweeps and forward-validate the better `vH` wide M branch.
- The `vH` wide M branch showed a concrete A-validation/forward mismatch:
  data-only A rel.RMSE improved to `0.268108`, but retained nonlinear forward
  `C(t)` was worse than `vE x1.05` at scales `0.95`, `1.0`, and `1.05`.
  Future joint-score M branches must still be forward-validated; A-validation
  alone is not enough.
- `vH` wide scaling improved covariance but degraded retained nonlinear
  dynamics when pushed upward. Do not keep increasing global correction scale
  to chase covariance if retained `C(t)` is worsening and state tails grow.
- For `joint vE wide`, the useful forward scale is narrow: `1.055` improved
  retained nonlinear `C(t)` to about `0.1216`, while `1.0575`/`1.05875`
  worsened dynamics and `1.06` was nonfinite. Do not repeat broader high-scale
  sweeps above this boundary.
- The late structured joint-score campaign ended with a forward-transfer
  mismatch. `vL` had the best retained-active joint-score operator metric
  (`0.245659` rel.RMSE, `0.969966` corr) and best data-only M A-validation
  (`0.259819` rel.RMSE, `0.967913` corr), but its forward retained nonlinear
  `C(t)` error stayed around `0.146` or worse. The best forward model remained
  `joint vE wide x1.055` at retained-C rel.RMSE `0.121584`.
- Do not treat global M scale sweeps around the `vE` optimum as a reliable
  improvement route. Later `vE x1.0525` and `x1.054` probes were nonfinite even
  though an earlier `x1.055` trajectory was finite and best, so the stability
  boundary is stochastic and narrow.
- `vK` isolated physical-augmented GroupNorm with initial-only active-lag DSM.
  It was eligible on retained active channels but forward validation was poor
  (`0.172489` retained-C rel.RMSE). Do not repeat this exact hybrid route as
  the next joint-score fix.
- The lower-noise uniform physical-augmented joint-score branch `vI` failed the
  retained-active gate (`0.631758` rel.RMSE, `0.777902` corr). Do not repeat
  this exact lower-noise uniform route as the next conditional-score fix.
- When adding conditional-source dispatch to M training, do not add fields to
  serialized `DMConfig`; read new TOML-only options such as
  `data.cond_score_kind` with helper functions to preserve old checkpoint
  loading.
- An oracle true-M residual-target diagnostic using learned `vL` transition
  scores, true local mobility, and \(\Phi=\langle M_{\rm true}\rangle\) showed
  that cleaner targets can improve retained forward dynamics but do not solve
  M recovery. The local-r2 oracle branch reached retained-C rel.RMSE `0.099442`
  but still had true-block rel.RMSE `0.428071`, covariance rel.RMSE `0.096924`,
  and imperfect global cross-correlations. This branch is explicitly not
  data-only; do not cite it as benchmark success.
- Oracle observable selection by true-M block-entry identifiability helped only
  when paired with a physically equivariant \(r^2\) M parameterization. The
  all-family identifiable library plus strong-mean equivariant branch improved
  true-block rel.RMSE from the old compact oracle `0.428071` to `0.381906`
  with corr `0.989175`, but local-r2 on the same library worsened to `0.508442`
  and the smoother neighbor-only identifiable library worsened to `0.563124`.
  Do not repeat "more identifiable channels" without also constraining the M
  parameterization and mean-mobility behavior. This remains an oracle,
  non-data-only diagnostic.

## SoftSpinLLGChain Stationary Score Recovery

- When true-M plus a learned score fails badly, do not keep iterating only on
  generic U-Net DSM scores. A data-only physical-feature DSM normal-equation
  score with features `x`, `r2*x`, `lap`, and `r2^2*x` recovered the stationary
  score much better and made `M_true + learned score` match the retained
  nonlinear dynamics very well.
- The physical-feature score is still data-only: coefficients are fitted from
  trajectory samples plus Gaussian DSM noise. Analytic score coefficients are
  allowed only for ex-post diagnostics, not for fitting, weighting, or model
  selection.
- `PhysicalFeatureScore` must use the direct-score dispatch
  `score_from_dsm_model(model::PhysicalFeatureScore, batch, sigma)`. The generic
  score model path assumes noise-prediction semantics and gives the wrong sign
  and scale for this model family.
- Select score variants by observed forward diagnostics such as covariance/PDF
  and retained nonlinear `C_mn(t)`, not by analytic-score RMSE. In the accepted
  2026-05-13 branch, `score_phys_pC_sigma02.bson` was the best all-around
  observed-forward choice: covariance rel.RMSE about `0.0516`, retained-C
  rel.RMSE about `0.0302`, retained-C corr about `0.99955` when paired with
  true mobility.

## SoftSpinLLGChain Mobility Observables

- Generalizing the right observable `phi_n` away from coordinates did not
  improve M recovery in the 2026-05-16 campaign.  Full and structured nonlinear
  right-observable libraries gave oracle true-M block rel.RMSE around
  `0.46-0.53`, worse than coordinate `phi_n=x`.
- The successful route was not "more channels" but a cleaner retained left
  observable set.  Starting from the 60-channel oracle-identifiable library,
  retain only channels whose active-lag data-only A target agrees with the
  oracle true-M A target.  The accepted `clean37` subset used rel.RMSE `<0.35`
  and corr `>0.95`, giving active data-vs-oracle target rel.RMSE `0.1818` and
  corr `0.9833`.
- For the data-only M loss, per-entry target scaling overweights tiny/noisy
  matrix entries.  Use data-only per-retained-channel RMS scaling for these
  residual targets.
- The clean37 data-only equivariant M branch is the current best SoftSpinLLG M
  result.  The strongest-forward branch is clean37 mean15: true-M block
  rel.RMSE `0.253830`, retained nonlinear C rel.RMSE `0.056724`, covariance
  rel.RMSE `0.084851`.  The best covariance branch remains clean37 mean10:
  covariance rel.RMSE `0.062004`, retained nonlinear C rel.RMSE `0.060607`.
  Keep the distinction clear: true M was used only to design/filter the
  observable set and for ex-post diagnostics, not in the final target values or
  M losses.
- Further pruning below clean37 was tried and failed.  Clean16, clean21, and
  clean32 gave A-validation rel.RMSE `0.93`, `0.91`, and `0.86`, respectively,
  so the residual operator became underdetermined.  Do not keep shrinking the
  library as the next default direction.
- Close clean37 optimizer/lag refinements did not beat the final tradeoff:
  mean06, lag5:20, lag8:24, and warm-started mean-penalty runs were all worse
  on A validation.  New improvements probably require a new data-only selection
  proxy or a genuinely new referee-acceptable M parameterization.
- A declared physics-informed four-coefficient onsite ansatz can substantially
  improve the final SoftSpinLLG forward validation when its coefficients are
  fitted from the clean37 data-only residual target.  The best branch used the
  old direct conditional residual score, a strong data-only Phi mean penalty,
  and a small PSD coefficient floor; it reached true-block rel.RMSE `0.0317`
  ex post, retained-C rel.RMSE `0.0472`, and covariance rel.RMSE `0.0492`.
  This is not a pure nonparametric NN result: the true functional form is a
  declared structural prior, while the true coefficients remain ex-post only.
- For the physics-informed ansatz, the late joint-score `vL` transition source
  was worse than the older direct conditional residual.  It pushed the symmetric
  coefficients toward an effective isotropic term and gave worse forward
  retained-C errors (`0.095-0.110`) than the direct-conditional ansatz branch.
- Do not forward-integrate a physics-informed ansatz whose fitted parallel
  symmetric coefficient is effectively zero.  The no-floor direct-conditional
  ansatz had excellent ex-post true-M error but produced nonfinite forward
  states.  A small PSD floor (`0.001`) stabilized the SDE and gave the best
  paper-facing forward metrics.

## Agent Persistence

- If the user instructs the agent to keep working until a quantitative result is
  reached, a status question or an intermediate failed branch is not a stopping
  condition. Reply briefly if needed, then continue the next concrete
  experiment until the target is met or a real blocker is documented.

## LaTeX Writeup PDFs

- Some IDE PDF viewers can fail on large image-heavy PDFs even when the file is
  valid. If `pdfinfo`, `pdftotext`, and `pdftoppm` can parse/render the file,
  rewrite the PDF through Ghostscript as a conservative PDF 1.4 compatibility
  copy before editing the LaTeX source.

## Human Writeups

- Do not leave non-obvious experimental choices implicit in
  `writeup/system_report.tex`. For SoftSpinLLGChain, the observable-library
  design, filtering thresholds, retained-channel structure, and the distinction
  between simulator-assisted legacy Phi and strict trajectory-only Phi must be
  stated directly in the human report, not only in `reports/agent_report.md`.
- A human writeup for this repository must include estimator-level equations,
  not only results. At minimum, include the data normalization, empirical
  correlation estimators, DSM targets, GFDT formulas, conditional-score target,
  mobility residual target, M parameterization, loss terms, divergence
  computation, and forward SDE used for validation.

## FHDChainCapillaryN32 Phi Resolution

- The original `save_dt = 0.5` FHDChainCapillaryN32 trajectory, even though it
  has 100 saved snapshots per pilot decorrelation time, is too coarse to
  recover the derivative-at-zero mean mobility `Phi`.
- Data-only improvements tried on the original cadence included block-circulant
  covariance projection, local polynomial extrapolation, weighted log-covariance
  fits, VAR-log fits, Euler-generator fits, Fourier branch unwrapping, and
  coarse quadratic-variation symmetric estimates. The best ex-post candidate
  still had `Phi` rel.RMSE about `0.97` against `<M_true>`.
- Do not repeat coarse-cadence Phi tuning as the next fix for this system. The
  successful `4096` short-burst estimate worked because it supplied necessary
  near-zero-lag information, not because it used a slightly better fitting
  routine.
- In a Gaussian-score forward sweep, the `dt=0.0005` Phi estimate gave combined
  rho/m ACF RMSE about `0.210`, while data-only selected `dt=0.005` and
  `dt=0.05` estimates were around `0.455` and `0.443`. The stationary covariance
  RMSE stayed near `0.11` for all cases, so judge Phi cadence by dynamical
  correlations, not only stationary covariance.
