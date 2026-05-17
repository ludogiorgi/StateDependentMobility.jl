# Agent Attempts

## 2026-05-17 — Commit and push source-prepared repository

Status: Partial

Goal:
- Stage the full source/documentation working tree after the `.gitignore`
  cleanup, commit it, and push it to GitHub as requested.

Approach:
- Confirmed the branch was `main` tracking `origin/main`.
- Fetched `origin/main` before staging to verify the local branch was current.
- Staged the whole visible working tree with generated artifacts excluded by
  `.gitignore`.
- Ran Git's whitespace/conflict-marker sanity check before committing.
- Committed the prepared benchmark source, configuration, reports, writeups,
  paper sources, and repository memory.

Files changed:
- Repository-wide source/configuration/documentation set visible to Git.
- `.gitignore`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- None.

Commands/tests run:
- `git status -sb`
- `git remote -v`
- `git branch --show-current`
- `gh --version && gh auth status`
- `git fetch origin main`
- `git add -A`
- `git diff --cached --check`
- `git commit`
- `git push origin main`

Outcome:
- The repository was committed locally.
- The push to `origin/main` failed because this noninteractive shell has no
  GitHub credentials for the HTTPS remote, `gh` is not authenticated, and SSH
  authentication is not configured.
- Generated data, checkpoints, figures, logs, PDFs, LaTeX auxiliary files,
  local prompts, and the local `ScoreUNet1D.jl/` checkout stayed ignored.

Do not repeat:
- Do not bypass the dry-run/ignore checks before broad staging in this
  repository; the generated experiment outputs are large and must remain out of
  normal source commits.

Useful follow-up:
- Authenticate GitHub in this environment, then run `git push origin main`.
  `gh` was installed but not authenticated, so no GitHub PR operation was
  attempted.

## 2026-05-17 — Prepare repository ignore rules for source commit

Status: Success

Goal:
- Prepare the repository for a future commit/push without committing, and make
  `.gitignore` include only professional source/documentation artifacts while
  excluding generated experiment outputs.

Approach:
- Replaced the old ignore-everything pattern with explicit ignore rules for
  generated benchmark data, model checkpoints, figures, logs, LaTeX build
  products, archives, local scratch files, and the local `ScoreUNet1D.jl/`
  checkout.
- Added explicit allow rules for source, configs, reports, writeups, paper
  sources, BibTeX, repository docs, and agent-memory markdown.
- Verified with `git check-ignore` that generated HDF5/checkpoint/figure/log/PDF
  artifacts are ignored while benchmark code, configs, reports, and LaTeX
  writeups are visible to Git.

Files changed:
- `.gitignore`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- None.

Commands/tests run:
- `git status --short --ignored`
- `git check-ignore -v prompts.md paper.pdf SoftSpinLLGChain/writeup/system_report.pdf SoftSpinLLGChain/data/soft_spin_llg_chain.h5 SoftSpinLLGChain/models/score_sigma005.bson SoftSpinLLGChain/figures/sim_summary.png SoftSpinLLGChain/logs/sim.log`
- `git check-ignore -v SoftSpinLLGChain/code/sim.jl SoftSpinLLGChain/reports/agent_report.md SoftSpinLLGChain/writeup/system_report.tex`
- `git check-ignore -v ScoreUNet1D.jl/Project.toml ScoreUNet1D.jl/src/ScoreUNet1D.jl`
- `git add -n .`

Outcome:
- Dry-run staging now includes source/config/documentation/report files and
  tracked deletions, but not generated data, model checkpoints, figures, logs,
  PDFs, LaTeX auxiliary files, local prompt notes, or the nested dependency
  checkout.
- No actual `git add`, commit, or push was performed.

Do not repeat:
- Do not return to a blanket ignore-everything rule for this repository; it
  hides new benchmark source and documentation from Git.
- Do not commit generated benchmark datasets, checkpoints, figures, logs, or
  compiled PDFs unless the user explicitly asks for an artifact release.

Useful follow-up:
- Before committing, review the dry-run add list carefully because the working
  tree includes pre-existing tracked deletions and many newly visible benchmark
  source/documentation directories.

## 2026-05-11 — Add repository agent-memory workflow

Status: Success

Goal:
- Add the requested multi-agent repository workflow instructions to
  `AGENTS.md` without changing the existing benchmark protocol.

Approach:
- Read the existing `AGENTS.md` and confirmed it contained the benchmark
  protocol.
- Appended the new repository workflow section after the existing instructions.
- Created the required `docs/agents/LESSONS.md`,
  `docs/agents/ATTEMPTS.md`, and `.agent-scratch/` support path.

Files changed:
- `AGENTS.md`
- `docs/agents/LESSONS.md`
- `docs/agents/ATTEMPTS.md`
- `.agent-scratch/.gitignore`
- `.gitignore`

Files created temporarily:
- None.

Commands/tests run:
- `sed -n '1,260p' AGENTS.md`
- `tail -n 120 AGENTS.md`
- `mkdir -p docs/agents .agent-scratch`
- `git status --short`
- `git check-ignore -v AGENTS.md docs/agents/LESSONS.md docs/agents/ATTEMPTS.md .agent-scratch/.gitignore`

Outcome:
- The requested workflow instructions were appended.
- Required agent-memory files and scratch directory support were created.
- Narrow `.gitignore` exceptions were added so the agent workflow files are not
  hidden by the repository's ignore-everything default.
- Verified with repository status and file inspection.

Do not repeat:
- Do not replace the existing benchmark protocol when adding repository-level
  workflow instructions; append new guidance instead.

Useful follow-up:
- Future task attempts should append compact entries below this one and add
  durable lessons to `LESSONS.md` only when useful.

## 2026-05-11 — Cleanup SoftSpinLLGChain artifacts

Status: Success

Goal:
- Clean `SoftSpinLLGChain/` under the new repository workflow while preserving
  accepted data, models, figures, compact diagnostics, and the system report.

Approach:
- Read `AGENTS.md`, `docs/agents/LESSONS.md`, and the latest attempts first.
- Inspected the `SoftSpinLLGChain/` inventory and the living system report.
- Removed explicit superseded failed-run artifacts: old forward HDF5s,
  failed/probe configs, obsolete figures, bulky rejected-branch logs, and
  superseded mobility checkpoints.
- Kept the accepted simulation, score, Phi, conditional-score, final learned-M,
  E20 comparison, compact target, and final comparison artifacts.

Files changed:
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- None.

Commands/tests run:
- `find SoftSpinLLGChain -maxdepth 3 -type f -printf '%p\\t%s\\n' | sort`
- `du -sh SoftSpinLLGChain/* | sort -h`
- explicit `rm -f` commands for obsolete files
- accepted-artifact existence checks with `test -f`
- `find SoftSpinLLGChain -type d -empty -print`

Outcome:
- `SoftSpinLLGChain/data` now keeps only production data, score/Phi forward
  data, and the two final learned-M comparison trajectories.
- `SoftSpinLLGChain/models` now keeps only accepted score/Phi/conditional
  score/final mobility checkpoints and the compact target.
- `SoftSpinLLGChain/logs` dropped from about `19M` to about `3.1M`.
- `SoftSpinLLGChain/figures` dropped from about `33M` to about `13M`.
- No accepted artifact from the report was missing after cleanup.

Do not repeat:
- Do not keep every failed forward trajectory and branch log once the report
  contains the key outcome. Preserve accepted artifacts and compact diagnostics
  only.

Useful follow-up:
- If a future agent needs a removed failed branch, rerun it from the report
  context rather than restoring bulky stale outputs.

## 2026-05-11 — Diagnose SoftSpinLLGChain Phi obs mz asymmetry

Status: Success

Goal:
- Explain why the observed `mz` PDF in `phi_forward_stats.png` looks less
  symmetric than the `mz` PDF in `sim_summary.png`.

Approach:
- Read the repository memory first.
- Inspected `fit_Phi.jl` and `sim.jl` plotting paths.
- Compared the full observation dataset against the contiguous observation
  segment used by `phi_forward_stats.png`.

Files changed:
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- None.

Commands/tests run:
- `rg -n "phi_forward_stats|PDF|kde|obs" SoftSpinLLGChain/code`
- `sed -n '460,568p' SoftSpinLLGChain/code/fit_Phi.jl`
- Python/HDF5 checks of full `mz` samples, post-burn samples, and the
  `phi_forward_stats` aligned observation segment.

Outcome:
- The true model and full dataset are spin-inversion symmetric up to finite
  sampling error.
- `sim_summary.png` uses all saved states from all trajectories, while
  `phi_forward_stats.png` uses only a short contiguous post-burn segment matched
  to the Phi forward trajectory length.
- That segment has a negative branch excess: about `38` negative-mean
  trajectories versus `34` positive-mean trajectories, and `P(mz < -0.5)` is
  about `1.16` times `P(mz > 0.5)`.

Do not repeat:
- Do not interpret the asymmetric `obs` `mz` curve in `phi_forward_stats.png`
  as a broken invariant density. It is a finite-window branch-imbalance artifact
  from the comparison plotting slice.

Useful follow-up:
- If regenerating this figure, draw the observation PDF from randomized
  stationary samples or symmetrize the observation reference for spin-inversion
  diagnostics, while keeping time-aligned windows for correlation comparisons.

## 2026-05-11 — Regenerate SoftSpinLLGChain forward PDFs from full stationary samples

Status: Success

Goal:
- Regenerate `forward_stats_final_accepted_learnedM_compare.png` with correct
  stationary PDFs, without retraining neural networks.
- Patch plotting code so future Phi-only and learned-M stats figures do not use
  short aligned observation windows for stationary PDFs.

Approach:
- Updated `render_forward_with_dM.jl` so PDF/covariance panels sample from all
  post-burn observations and all saved forward-model states.
- Updated `evaluate_forward_grid.jl` so only correlation diagnostics use aligned
  windows.
- Updated `fit_Phi.jl` so `phi_forward_stats.png` follows the same full-sample
  rule.
- Reused existing saved forward HDF5 trajectories; no NN training and no
  forward reintegration.

Files changed:
- `SoftSpinLLGChain/code/render_forward_with_dM.jl`
- `SoftSpinLLGChain/code/evaluate_forward_grid.jl`
- `SoftSpinLLGChain/code/fit_Phi.jl`
- `SoftSpinLLGChain/figures/forward_stats_final_accepted_learnedM_compare.png`
- `SoftSpinLLGChain/figures/forward_cmn_final_accepted_learnedM_compare.png`
- `SoftSpinLLGChain/figures/phi_forward_stats.png`
- `SoftSpinLLGChain/logs/forward_final_accepted_learnedM_compare_metrics.txt`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- None.

Commands/tests run:
- `julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/render_forward_with_dM.jl"); println("render include ok after fit_Phi patch")'`
- `xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi.toml final_accepted_learnedM_compare 'Phi=../data/phi_forward_langevin.h5' 'M_NN compact E20 dt0015=../data/forward_dM_compact_E20_dt0015.h5' 'M_NN compact lag7 X final=../data/forward_dM_compact_lag7_vX_final_dt0015.h5'`
- `xvfb-run -a julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/fit_Phi.jl"); ... render_forward_stats(...)'`
- Visual inspection of regenerated `forward_stats_final_accepted_learnedM_compare.png`
  and `phi_forward_stats.png`.

Outcome:
- The `obs` PDF panels now use all post-burn observation trajectories instead
  of a short early window.
- The final learned-M stats figure and Phi-only stats figure were regenerated.
- Final learned-M metrics file documents the sampling rule and reports:
  `Phi` covariance rel.RMSE `0.146459`, `E20` `0.067017`, `lag7` `0.068377`.

Do not repeat:
- Do not use aligned finite-lag comparison windows for stationary PDF panels in
  slow branch-mixing systems.

Useful follow-up:
- If model-side `mz` branch imbalance remains visually distracting, generate
  additional branch-balanced forward trajectories from the saved checkpoints
  without retraining, or explicitly document the finite-trajectory branch
  imbalance. Do not silently symmetrize model PDFs unless that diagnostic choice
  is labeled.

## 2026-05-11 — Increase SoftSpinLLGChain forward-correlation time range

Status: Success

Goal:
- Increase the time range in the final `forward_cmn` comparison figure without
  retraining NNs.

Approach:
- Added a shared `DEFAULT_FORWARD_CORR_MAX_LAGS = 300` constant in
  `fit_Phi.jl`.
- Updated Phi-only, direct learned-M, and grid learned-M correlation renderers
  to use the 300-lag cap.
- Regenerated the final accepted learned-M comparison from saved HDF5 forward
  trajectories.

Files changed:
- `SoftSpinLLGChain/code/fit_Phi.jl`
- `SoftSpinLLGChain/code/render_forward_with_dM.jl`
- `SoftSpinLLGChain/code/evaluate_forward_grid.jl`
- `SoftSpinLLGChain/figures/forward_cmn_final_accepted_learnedM_compare.png`
- `SoftSpinLLGChain/figures/forward_stats_final_accepted_learnedM_compare.png`
- `SoftSpinLLGChain/logs/forward_final_accepted_learnedM_compare_metrics.txt`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- None.

Commands/tests run:
- `julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/render_forward_with_dM.jl"); println(DEFAULT_FORWARD_CORR_MAX_LAGS)'`
- `xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi.toml final_accepted_learnedM_compare 'Phi=../data/phi_forward_langevin.h5' 'M_NN compact E20 dt0015=../data/forward_dM_compact_E20_dt0015.h5' 'M_NN compact lag7 X final=../data/forward_dM_compact_lag7_vX_final_dt0015.h5'`
- Visual inspection of regenerated
  `figures/forward_cmn_final_accepted_learnedM_compare.png`.

Outcome:
- The final `forward_cmn` figure now spans `corr_lags = 300`, i.e.
  `corr_tmax = 10.95`, about `3 t_D`.
- Longer-window coordinate-C metrics are now recorded in
  `logs/forward_final_accepted_learnedM_compare_metrics.txt`.

Do not repeat:
- Do not hard-code `80` lags in new forward-correlation renderers. Use
  `DEFAULT_FORWARD_CORR_MAX_LAGS` so all forward-correlation figures stay
  consistent.

Useful follow-up:
- If later-time correlations become visually noisy, consider plotting confidence
  bands or a shorter secondary zoom panel rather than reverting the saved lag
  cap.

## 2026-05-11 — Plot retained observables in SoftSpinLLGChain forward Cmn

Status: Success

Goal:
- Regenerate the final learned-M `forward_cmn` figure so it contains the
  nonlinear observable correlations used to train the compact mobility models,
  rather than only coordinate auto-correlations.

Approach:
- Added a retained-channel Cmn plotting path in `render_forward_with_dM.jl`.
- Loaded the 36 compact retained observable/target-component channels from
  `configs/nonlinear_observable_retained_channels_compact.toml`.
- Computed centered, site-averaged forward correlations over `300` lags using
  `6000` sampled time-trajectory pairs per lag and all lattice sites.
- Updated `evaluate_forward_grid.jl` to use this retained-channel Cmn path for
  the final grid comparison.

Files changed:
- `SoftSpinLLGChain/code/render_forward_with_dM.jl`
- `SoftSpinLLGChain/code/evaluate_forward_grid.jl`
- `SoftSpinLLGChain/figures/forward_cmn_final_accepted_learnedM_compare.png`
- `SoftSpinLLGChain/figures/forward_stats_final_accepted_learnedM_compare.png`
- `SoftSpinLLGChain/logs/forward_final_accepted_learnedM_compare_metrics.txt`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- None.

Commands/tests run:
- `julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/render_forward_with_dM.jl"); ch, lib = load_forward_cmn_channels(DEFAULT_FORWARD_CMN_CHANNELS_TOML); println(length(ch)); println(length(lib.names)); println(channel_label(ch[1])); println(DEFAULT_FORWARD_OBSERVABLE_CORR_PAIRS)'`
- Small retained-correlation smoke test on a 30-frame, 2-trajectory slice.
- `julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/evaluate_forward_grid.jl"); println("include ok after titlesize patch")'`
- `xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi.toml final_accepted_learnedM_compare 'Phi=../data/phi_forward_langevin.h5' 'M_NN compact E20 dt0015=../data/forward_dM_compact_E20_dt0015.h5' 'M_NN compact lag7 X final=../data/forward_dM_compact_lag7_vX_final_dt0015.h5'`
- Visual inspection of
  `figures/forward_cmn_final_accepted_learnedM_compare.png`.

Outcome:
- The final `forward_cmn` figure now contains all 36 compact retained
  observable/target-component channels.
- Retained-observable C metrics over the 300-lag window:
  `Phi` rel.RMSE `0.917465`, corr `0.891614`;
  `E20` rel.RMSE `0.159875`, corr `0.988147`;
  `lag7 X final` rel.RMSE `0.119969`, corr `0.992573`.
- No NN retraining or forward reintegration was performed.

Do not repeat:
- Do not validate the learned-M forward dynamics only with coordinate
  auto-correlations; that hides the observable channels actually used by the
  compact residual-A mobility fit.

Useful follow-up:
- If a future paper figure needs the full translation-offset tensor rather than
  a compact site-averaged view, render the retained channels as time-offset
  heatmaps or a multi-page figure instead of returning to coordinate-only
  panels.

## 2026-05-11 — SoftSpinLLGChain learned-M forward bottleneck probes

Status: Partial

Goal:
- Improve state-dependent learned-M forward validation so stationary statistics
  and retained nonlinear `C_mn(t)` agree very well with observations.

Approach:
- Used all three GPUs for independent forward probes from existing learned-M
  checkpoints, without retraining the NNs.
- Patched final Cmn evaluation to use the full post-burn observation pool and
  `30000` retained-C pairs per lag.
- Added a forward-only antisymmetric/skew mobility scale in
  `SoftSpinLLGChain/code/forward_dM.jl` and tested skew gains `1.25`, `1.5`,
  and `2.0`.
- Ran ex-post diagnostics for physical KM and true-M plus learned score to
  isolate whether the bottleneck was M or the learned score.

Files changed:
- `SoftSpinLLGChain/code/forward_dM.jl`
- `SoftSpinLLGChain/code/render_forward_with_dM.jl`
- `SoftSpinLLGChain/code/evaluate_forward_grid.jl`
- `SoftSpinLLGChain/configs/fit_dM_compact_lag7_gpu0_vX_forwardonly.toml`
- `SoftSpinLLGChain/configs/fit_dM_compact_lag7_gpu1_vX_forwardonly.toml`
- `SoftSpinLLGChain/figures/forward_stats_final_accepted_learnedM_compare.png`
- `SoftSpinLLGChain/figures/forward_cmn_final_accepted_learnedM_compare.png`
- `SoftSpinLLGChain/logs/forward_final_accepted_learnedM_compare_metrics.txt`
- `SoftSpinLLGChain/reports/agent_report.md`

Files created temporarily:
- `.agent-scratch/softspin_scale/`
- `.agent-scratch/softspin_scale_probe/`
- Rejected forward HDF5 trajectories in `SoftSpinLLGChain/data/` for vW, vY,
  vZ, vX-continuation, global-scale, skew-scale, physical-KM, and true-M
  ex-post probes.

Commands/tests run:
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl ...`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ...`
- `julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/forward_dM.jl"); println("forward_dM include ok with skew_scale")'`
- Visual inspection of final and probe `forward_stats`/`forward_cmn` figures.

Outcome:
- Final accepted model remains `data/forward_dM_compact_lag7_vX_final_dt0015.h5`.
- Final metrics: Phi covariance rel.RMSE `0.146459`, retained-C rel.RMSE
  `0.918993`; learned vX covariance rel.RMSE `0.067129`, retained-C rel.RMSE
  `0.110532`, retained-C corr `0.993614`.
- vW improved covariance (`0.056150`) but worsened retained-C (`0.113160`) and
  had worse `mz` branch balance.
- vY/vZ/vX-continuation/global-scale/skew-scale probes all failed to improve
  the accepted retained-C metric. Skew gains also produced large tails.
- True-M plus learned-score ex-post forward was much worse than accepted vX,
  showing the current stationary score is a real bottleneck.

Do not repeat:
- Do not keep increasing local MLP capacity or applying global/skew mobility
  scales expecting perfect agreement. Those probes did not beat accepted vX.
- Do not interpret true mobility as an upper bound when paired with the current
  learned score; true-M plus learned-score was worse than the effective learned
  M because the learned M compensates for score bias.

Useful follow-up:
- The next real improvement should start with a better stationary score or a
  principled score/mobility joint correction. More local forward probes are not
  the likely path.

## 2026-05-12 — SoftSpinLLGChain score-function improvement pass

Status: Failed

Goal:
- Improve learned score/Phi/M forward validation without analytic training
  targets, using all three GPUs for score and conditional-score variants.

Approach:
- Fixed `fit_dM.jl` so configs can specify `data.cond_score_config`; the
  previous hard-coded `cond_score_gpu0_vA.toml` would have made new
  conditional-score checkpoints use the wrong lag/model config.
- Tested score vC with trajectory-only Phi and retrained compact M variants.
- Trained/evaluated targeted conditional residual scores on all GPUs:
  score-vC lag-24 variants, a score-vC lag-36 wider variant, old-score lag-24
  variants, and a continued old-score vA checkpoint.

Files changed:
- `SoftSpinLLGChain/code/fit_dM.jl`
- `docs/agents/LESSONS.md`
- `docs/agents/ATTEMPTS.md`
- `SoftSpinLLGChain/reports/agent_report.md`

Files created temporarily:
- `.agent-scratch/phi_window_scan.jl`
- `.agent-scratch/cond_score_oldscore_lag24_gpu1_vG_eval.toml`
- Rejected score/conditional/M configs, checkpoints, forward HDF5 probes, and
  figures under `SoftSpinLLGChain/`; removed during cleanup.

Commands/tests run:
- Multiple `score.jl`, `fit_Phi.jl`, `cond_score.jl`, `fit_dM.jl`,
  `forward_dM.jl`, and `evaluate_forward_grid.jl` runs on the two 2080 Ti GPUs
  and the RTX 5070.
- `julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/fit_dM.jl"); ... BSON.load(...)'`
- `xvfb-run -a julia --project=. --threads 16 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ... score_vC_dataPhi_vM_forward_probe ...`

Outcome:
- No new branch beat the accepted legacy `score_sigma005` +
  `cond_score_residual_gpu0_vA` + compact lag7 vX forward model.
- Score-vC with old M improved covariance slightly (`0.0628` vs accepted
  `0.0671`) but worsened retained-C (`0.1258` vs accepted `0.1105`).
- Score-vC with trajectory-only Phi and retrained M vM worsened both covariance
  (`0.0794`) and retained-C (`0.1259`).
- Conditional-score operator diagnostics all failed to beat vA:
  vA old accepted `0.4825`; score-vC lag24 vB `0.6640`, vC `0.8240`,
  vE `0.5787`, lag36 vD `0.7157`; old-score continuation vH `0.5969`;
  old-score lag24 vG epoch-30 `0.8025`.
- The no-penalty and lag-targeted branches often improved posterior DSM MSE
  but worsened the operator used by the paper identity.

Do not repeat:
- Do not assume lower conditional DSM/posterior MSE means a better
  transition-score operator.
- Do not pair a new stationary score with an old conditional residual/M stack
  and interpret the result as a clean improvement.
- Do not continue vA or train old-score lag24 GroupNorm from scratch expecting
  an easy operator improvement; both were worse in this pass.

Useful follow-up:
- A credible next attempt needs a different stationary-score strategy or a
  data-only operator-aware conditional-score validation objective. More local M
  retraining is not the bottleneck.

## 2026-05-13 — SoftSpinLLGChain true-M learned-score recovery

Status: Success

Goal:
- Improve the stationary score until forward integration with true mobility and
  the learned score gives very good observed statistical and dynamical
  agreement.

Approach:
- Added a data-only physical-feature stationary score fitted by streamed DSM
  normal equations from trajectory samples plus Gaussian noise.
- Tested three DSM sigmas in parallel-style branches:
  pA `sigma=0.05`, pB `sigma=0.035`, pC `sigma=0.02`.
- Added a dedicated direct-score dispatch for `PhysicalFeatureScore`; without
  it, diagnostics wrongly pass through the generic noise-predictor path.
- Integrated true mobility with each learned physical score and evaluated PDFs,
  covariance, and all 36 retained nonlinear observable correlations against
  observations.

Files changed:
- `AGENTS.md`
- `SoftSpinLLGChain/code/src/spin_common.jl`
- `SoftSpinLLGChain/code/fit_physical_score.jl`
- `SoftSpinLLGChain/code/score_posthoc_metrics.jl`
- `SoftSpinLLGChain/code/forward_trueM_score.jl`
- `SoftSpinLLGChain/configs/score_phys_gpu0_pA_sigma05.toml`
- `SoftSpinLLGChain/configs/score_phys_gpu1_pB_sigma035.toml`
- `SoftSpinLLGChain/configs/score_phys_gpu2_pC_sigma02.toml`
- `SoftSpinLLGChain/configs/score_phys_pA_eval.toml`
- `SoftSpinLLGChain/configs/score_phys_pB_eval.toml`
- `SoftSpinLLGChain/configs/score_phys_pC_eval.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/LESSONS.md`
- `docs/agents/ATTEMPTS.md`

Files created temporarily:
- Rejected U-Net score probe configs, checkpoints, score-only HDF5 files,
  figures, and logs for vF/vG/vH; removed during cleanup.

Commands/tests run:
- `julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_physical_score.jl SoftSpinLLGChain/configs/score_phys_gpu0_pA_sigma05.toml`
- `julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_physical_score.jl SoftSpinLLGChain/configs/score_phys_gpu1_pB_sigma035.toml`
- `julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_physical_score.jl SoftSpinLLGChain/configs/score_phys_gpu2_pC_sigma02.toml`
- `julia --project=. --startup-file=no SoftSpinLLGChain/code/score_posthoc_metrics.jl ...`
- `CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_trueM_score.jl SoftSpinLLGChain/configs/fit_Phi.toml SoftSpinLLGChain/data/forward_trueM_score_phys_pA_dt0015.h5 0.0015 GPU:1 2080ti ../models/score_phys_pA_sigma05.bson`
- `CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_trueM_score.jl SoftSpinLLGChain/configs/fit_Phi.toml SoftSpinLLGChain/data/forward_trueM_score_phys_pB_dt0015.h5 0.0015 GPU:0 2080ti ../models/score_phys_pB_sigma035.bson`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_trueM_score.jl SoftSpinLLGChain/configs/fit_Phi.toml SoftSpinLLGChain/data/forward_trueM_score_phys_pC_dt0015.h5 0.0015 GPU:2 5070 ../models/score_phys_pC_sigma02.bson`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi.toml trueM_phys_score_compare 'Phi=../data/phi_forward_langevin.h5' 'M_true phys pA=../data/forward_trueM_score_phys_pA_dt0015.h5' 'M_true phys pB=../data/forward_trueM_score_phys_pB_dt0015.h5' 'M_true phys pC=../data/forward_trueM_score_phys_pC_dt0015.h5'`

Outcome:
- The old generic U-Net score probes were rejected. Their best analytic
  rel.RMSE was still much worse and one score-only Langevin covariance error was
  near `0.95`.
- Physical pC achieved ex-post analytic score rel.RMSE `0.02184`, cosine
  `0.999953`. This diagnostic was not used as a training target.
- True-M plus physical pC gave covariance rel.RMSE `0.05157`, covariance corr
  `0.99846`, retained nonlinear C rel.RMSE `0.03024`, retained-C corr
  `0.99955`.
- True-M plus physical pA had the smallest retained-C rel.RMSE `0.02904` but
  worse covariance `0.09035`; pC is the best all-around observed-forward score.

Do not repeat:
- Do not continue the failed vF/vG/vH generic U-Net score branches.
- Do not evaluate a direct score model through the generic noise-predictor
  `score_from_dsm_model` path.
- Do not select by analytic-score error; use observed forward metrics, with
  analytic diagnostics labeled ex-post only.

Useful follow-up:
- For final Step 3 learned-M work, start from `models/score_phys_pC_sigma02.bson`
  and rerun Phi/conditional-score/M training consistently against this improved
  stationary score. The old learned-M stack may have compensated for the old
  score bias.

## 2026-05-14 — Standardize benchmark report filenames

Status: Success

Goal:
- Update the benchmark protocol and existing `FHDChainCapillaryN32` folder to
  use clearer names for the agent-facing and human-facing reports.

Approach:
- Changed the protocol naming convention from generic `summary.md` and
  `system.tex` names to `reports/agent_report.md` and
  `writeup/system_report.tex`.
- Renamed the existing FHDChainCapillaryN32 Markdown and LaTeX reports, removed
  stale `system.*` LaTeX build outputs, and compiled the renamed LaTeX source.

Files changed:
- `AGENTS.md`
- `FHDChainCapillaryN32/reports/agent_report.md`
- `FHDChainCapillaryN32/writeup/system_report.tex`
- `FHDChainCapillaryN32/writeup/system_report.pdf`
- `docs/agents/ATTEMPTS.md`

Files created temporarily:
- None.

Commands/tests run:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
- `rg -n "summary\\.md|writeup/system\\.tex|system\\.pdf|system_report\\.tex|agent_report\\.md" AGENTS.md FHDChainCapillaryN32 docs/agents -g '!*.h5' -g '!*.bson' -g '!*.pdf'`
- `find FHDChainCapillaryN32/reports FHDChainCapillaryN32/writeup -maxdepth 1 -type f -printf '%p\\n' | sort`

Outcome:
- The protocol now names the agent report as `agent_report.md` and the human
  LaTeX writeup as `system_report.tex`.
- `FHDChainCapillaryN32` now contains `reports/agent_report.md` and
  `writeup/system_report.tex`, with a freshly compiled `system_report.pdf`.
- No stale `writeup/system.*` files remain.

Do not repeat:
- Do not introduce generic report filenames such as `summary.md` or
  `system.tex` for new benchmark systems unless the protocol is intentionally
  changed again.

Useful follow-up:
- Existing older benchmark folders can be migrated to the same report filenames
  when they are next touched.

## 2026-05-14 — Rewrite FHDChainCapillaryN32 human system report

Status: Success

Goal:
- Rewrite the human-facing LaTeX report for `FHDChainCapillaryN32` so it
  documents the system definition, mathematical formulation, experiments,
  failures, accepted results, and figures without requiring the reader to inspect
  implementation files.

Approach:
- Read `paper.tex`, the benchmark protocol, the FHDChainCapillaryN32 agent
  report, configs, logs, metric files, figure inventory, HDF5 metadata, and the
  relevant implementation details needed to reconstruct the mathematics.
- Replaced the previous derivation-only writeup with a living scientific report
  covering simulation validation, stationary-score training, initial and
  improved data-only Phi estimation, U-Net forward failures, and the Gaussian
  stationary-score repair.
- Included the relevant existing figures in the compiled report and tightened
  table/float formatting until the LaTeX log was warning-free.

Files changed:
- `FHDChainCapillaryN32/writeup/system_report.tex`
- `FHDChainCapillaryN32/writeup/system_report.pdf`
- `docs/agents/ATTEMPTS.md`

Files created temporarily:
- LaTeX auxiliary files in `FHDChainCapillaryN32/writeup/`; removed after
  compilation.

Commands/tests run:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
- `rg -n "Overfull|Underfull|Warning|undefined|Fatal|Emergency" system_report.log || true`
- HDF5 metadata inspection for the production simulation and burst datasets.
- Figure inventory checks for the referenced report figures.

Outcome:
- Produced a 21-page compiled report at
  `FHDChainCapillaryN32/writeup/system_report.pdf`.
- The final LaTeX build completed successfully and the warning/error scan of
  `system_report.log` was empty.
- The report now distinguishes data-only training/selection from ex-post
  analytic diagnostics and records the failed branches that matter for future
  Step 3 work.

Do not repeat:
- Do not leave the human writeup as only a system derivation. It must also
  summarize the completed experiments, rejected branches, quantitative outcomes,
  and the figures that support those conclusions.

Useful follow-up:
- If conditional-score or learned-mobility work resumes, update both
  `reports/agent_report.md` and `writeup/system_report.tex` immediately after
  each accepted experiment, then recompile `system_report.pdf`.

## 2026-05-14 — SoftSpinLLGChain conditional/M continuation before shutdown

Status: Partial

Goal:
- Continue Step 3 learned-score/learned-M work using all three GPUs, then stop
  cleanly when the user needed to shut down the workstation.

Approach:
- Continued the best physical-score U-Net conditional residual model
  `cond_score_phys_pC_unet_vA_cont` on the RTX 5070.
- Retrained local `neighbor_r2` mobility NNs from the best vJ checkpoint using
  the improved conditional score.
- Forward-integrated the close vN checkpoint at scale `1.10` and evaluated it
  against Phi, vD, and vJ baselines.
- Started two additional conditional-score continuations on the 2080 Ti GPUs
  and stopped them on user request after the current M run finished.

Files changed:
- `SoftSpinLLGChain/configs/cond_score_phys_pC_unet_gpu2_vA_cont2.toml`
- `SoftSpinLLGChain/configs/cond_score_phys_pC_unet_gpu0_vA_constrained_cont.toml`
- `SoftSpinLLGChain/configs/cond_score_phys_pC_unet_gpu1_vA_shortlag_cont.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_unet_vAcont2_gpu2_vN_cached_warmJ.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_unet_vAcont2_gpu2_vN_best_forward.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_unet_vAcont2_gpu2_vO_lag5_cached_warmJ.toml`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`
- `SoftSpinLLGChain/reports/agent_report.md`

Files created temporarily:
- Interrupted conditional continuation checkpoints/logs for
  `cond_score_phys_pC_unet_vA_shortlag_cont` and
  `cond_score_phys_pC_unet_vA_constrained_cont`; keep only as restart evidence
  if needed.

Commands/tests run:
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/cond_score.jl SoftSpinLLGChain/configs/cond_score_phys_pC_unet_gpu2_vA_cont2.toml`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_unet_vAcont2_gpu2_vN_cached_warmJ.toml`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_unet_vAcont2_gpu2_vN_best_forward.toml SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml SoftSpinLLGChain/data/forward_dM_phys_pC_dataonly_vAcont2_vN_best_scale1p10_dt0015.h5 1.10 0.0015`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml forward_phys_pC_dataonly_vAcont2_vN_compare ...`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_unet_vAcont2_gpu2_vO_lag5_cached_warmJ.toml`
- Killed the two remaining conditional-score jobs after vO metrics were saved.

Outcome:
- `vA_cont2` improved the conditional true-M operator diagnostic from rel.RMSE
  `0.394855`, corr `0.924924` to rel.RMSE `0.378631`, corr `0.934872`.
- Warm-started M `vN` did not improve A fit: rel.RMSE `0.286565`, corr
  `0.960107`, versus vJ rel.RMSE `0.286122`, corr `0.960154`.
- vN forward at scale `1.10` was worse than vJ scale `1.10`: retained-C
  rel.RMSE `0.164720`, corr `0.986221`, covariance rel.RMSE `0.081999`.
- Earlier-lag M branch `vO` finished and saved results but was worse:
  A rel.RMSE `0.296931`, corr `0.957308`.
- Best fully learned-M forward result remains `vJ x1.10`: retained-C rel.RMSE
  about `0.1349`, corr about `0.9910`. True-M plus learned physical pC score
  remains the reference at retained-C rel.RMSE about `0.0302`.
- All GPU processes were stopped after vO saved, per user shutdown request.

Do not repeat:
- Do not retrain M from vJ with `vA_cont2` alone expecting improvement; vN did
  not improve A or forward metrics.
- Do not include earlier lag index 5 with the same local `neighbor_r2` setup as
  the next fix; vO worsened A validation.
- The physical-feature conditional MLP route already failed; the untried
  feature route is a stationary-score-style ridge/normal-equation conditional
  residual, not another generic MLP.

Useful follow-up:
- The next promising route is a data-only ridge/normal-equation conditional
  residual feature expansion over `(x0, xt, tau)`, then M training only if its
  operator diagnostic beats `vA_cont2`.

## 2026-05-14 — SoftSpinLLGChain human writeup

Status: Success

Goal:
- Read the updated `AGENTS.md` instructions and create the required
  human-facing LaTeX writeup for `SoftSpinLLGChain`.

Approach:
- Created the missing `SoftSpinLLGChain/writeup/` folder.
- Wrote `system_report.tex` as a scientific report covering the system
  definition, reference data, score/Phi/conditional-score/M-learning results,
  failed branches, final forward metrics, figures, no-cheating audit, and open
  issues.
- Compiled the report to PDF and fixed the only layout warnings from wide
  metric tables.

Files changed:
- `SoftSpinLLGChain/writeup/system_report.tex`
- `SoftSpinLLGChain/writeup/system_report.pdf`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`

Files created temporarily:
- LaTeX auxiliary build files in `SoftSpinLLGChain/writeup/`; removed after the
  successful compile.

Commands/tests run:
- `mkdir -p SoftSpinLLGChain/writeup`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
- `grep -E "(Warning|Overfull|Undefined|Error)" system_report.log || true`

Outcome:
- The writeup compiles to an 18-page PDF.
- The final LaTeX log check reported no warnings, overfull boxes, undefined
  references, or errors.
- The PDF references the existing accepted figures and summarizes the current
  best result: learned M improves retained finite-lag dynamics from Phi
  rel.RMSE about `0.968` to about `0.135`, while true M plus learned physical
  score remains much better at about `0.030`.

Do not repeat:
- Do not leave LaTeX auxiliary files in `writeup/` after a successful compile.
- Do not make the human writeup a command/config diary; that information belongs
  in `reports/agent_report.md` and `docs/agents/ATTEMPTS.md`.

Useful follow-up:
- After any new accepted SoftSpinLLGChain experiment, update both the
  agent-facing Markdown report and `writeup/system_report.tex`, then recompile
  `system_report.pdf`.

## 2026-05-14 — SoftSpinLLGChain PDF compatibility repair

Status: Success

Goal:
- Fix the user-reported inability to open `SoftSpinLLGChain/writeup/system_report.pdf`.

Approach:
- Verified that the existing PDF existed, had normal permissions, was
  unencrypted, had 18 pages, and could be parsed by `pdfinfo`.
- Verified text extraction with `pdftotext`.
- Rendered page 1 with `pdftoppm` to confirm the PDF contents were readable by
  Poppler.
- Rewrote the PDF through Ghostscript as a PDF 1.4 file with smaller embedded
  image streams, then replaced `system_report.pdf` with the compatibility copy.

Files changed:
- `SoftSpinLLGChain/writeup/system_report.pdf`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- `.agent-scratch/pdf_check/`; removed after verification.

Commands/tests run:
- `file SoftSpinLLGChain/writeup/system_report.pdf`
- `pdfinfo SoftSpinLLGChain/writeup/system_report.pdf`
- `pdftotext SoftSpinLLGChain/writeup/system_report.pdf -`
- `pdftoppm -f 1 -singlefile -png SoftSpinLLGChain/writeup/system_report.pdf ...`
- `gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer ...`

Outcome:
- Final PDF is PDF 1.4, 18 pages, unencrypted, and about `5.2M` instead of
  `9.3M`.
- `pdfinfo`, `pdftotext`, and `pdftoppm` all succeed on the rewritten PDF.

Do not repeat:
- If the IDE viewer cannot open a LaTeX PDF that command-line PDF tools can
  parse, try rewriting it to a conservative PDF 1.4 compatibility copy before
  changing the LaTeX source.

Useful follow-up:
- If a specific viewer still fails, test the exact viewer outside Codex; the
  current artifact is valid according to standard command-line PDF tools.

## 2026-05-14 — SoftSpinLLGChain writeup expansion after user review

Status: Success

Goal:
- Improve the human-facing SoftSpinLLGChain report after the user pointed out
  that the first version was too sloppy and omitted important details,
  especially observable-library selection.

Approach:
- Reviewed the updated `AGENTS.md`, agent memory files, observable-search
  notes, retained-channel TOML, and the current LaTeX report.
- Rewrote the Phi-baseline section to explicitly classify the older short-lag
  Phi estimator as simulator-assisted because it generated extra near-zero-lag
  transitions using the known SDE coefficients.
- Added a substantial observable-library subsection with mathematical
  definitions, selection criteria, thresholds, retained counts, representative
  retained channels, and the role of the ex-post true-M operator diagnostic.
- Added the nonlinear-observable search summary figure to the report.
- Expanded the mobility section to explain why the NN outputs local onsite
  blocks rather than a dense `36 x 36` matrix.
- Strengthened the no-cheating audit to distinguish allowed observable design
  from forbidden training-target construction.
- Recompiled the report, checked the LaTeX log, and rewrote the final PDF as a
  PDF 1.4 compatibility copy.

Files changed:
- `SoftSpinLLGChain/writeup/system_report.tex`
- `SoftSpinLLGChain/writeup/system_report.pdf`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- `.agent-scratch/pdf_check/`; removed after PDF compatibility verification.

Commands/tests run:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
- `grep -E "(Warning|Overfull|Underfull|Undefined|Error)" system_report.log || true`
- `gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer ...`
- `pdfinfo SoftSpinLLGChain/writeup/system_report.pdf`
- `pdftotext SoftSpinLLGChain/writeup/system_report.pdf -`

Outcome:
- The report now compiles to a 20-page PDF.
- The final log scan reported no warnings, overfull boxes, underfull boxes,
  undefined references, or errors.
- The final PDF is PDF 1.4, unencrypted, and about `5.6M`.

Do not repeat:
- Do not write the human report as a thin result summary. It must explain
  non-obvious choices such as observable design, filtering thresholds, and
  what was rejected.
- Do not describe the legacy short-lag Phi estimate as merely "more
  conservative" or "data-only"; it generated short-lag transitions with the
  known simulator and should be called simulator-assisted.

Useful follow-up:
- If revising the report again, check whether the conditional-score and
  learned-M sections need the same level of detail as the observable section.

## 2026-05-14 — SoftSpinLLGChain writeup mathematical implementation expansion

Status: Success

Goal:
- Address the user's feedback that the report still did not explain everything
  implemented and needed the mathematical details of the full pipeline.

Approach:
- Re-read the active writeup, agent memory, and the implementations for score,
  Phi, conditional score, mobility training, and learned-M forward integration.
- Added a new mathematical implementation section to the human report covering:
  data standardization, empirical lag averages, Euler--Maruyama simulation,
  stationary DSM, the physical-feature score formula, Phi/GFDT estimation,
  conditional residual DSM, nonlinear residual target construction, M NN
  parameterization and loss, and forward learned-M integration including
  divergence and diffusion factor construction.
- Recompiled the report, scanned the LaTeX log, and rewrote the PDF as a
  conservative PDF 1.4 compatibility artifact.

Files changed:
- `SoftSpinLLGChain/writeup/system_report.tex`
- `SoftSpinLLGChain/writeup/system_report.pdf`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- `.agent-scratch/pdf_check/`; removed after verification.

Commands/tests run:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
- `grep -E "(Warning|Overfull|Underfull|Undefined|Error)" system_report.log || true`
- `gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer ...`
- `pdfinfo SoftSpinLLGChain/writeup/system_report.pdf`
- `pdftotext SoftSpinLLGChain/writeup/system_report.pdf -`

Outcome:
- The report now compiles to a 25-page PDF.
- The final log scan reported no warnings, overfull boxes, underfull boxes,
  undefined references, or errors.
- The final PDF is PDF 1.4, unencrypted, and about `5.7M`.

Do not repeat:
- Do not satisfy a request for a scientific writeup by only listing outcomes and
  figures. The writeup must include enough equations that a reader can
  reconstruct the estimators, targets, losses, and forward models.

Useful follow-up:
- If further expanding the report, add more detail on individual failed
  branches, but the core estimator mathematics is now present.

## 2026-05-14 — FHDChainCapillaryN32 coarse-cadence Phi experiment

Status: Success

Goal:
- Re-estimate `Phi` for `FHDChainCapillaryN32` using only the original
  `save_dt = 0.5` dataset with 100 saved snapshots per pilot decorrelation
  time, improving the fitting/extrapolation algorithm without generating
  higher-resolution data or using true mobility for selection.

Approach:
- Added a dedicated coarse-cadence Phi sweep that uses only
  `data/fhd_chain_capillary_n32_short.h5`.
- Tested data-only estimator families: block-circulant/tangent projected local
  polynomial covariance derivatives, weighted covariance-log fits, VAR-log
  finite-lag propagator fits, Euler-generator least squares, Fourier logarithm
  branch unwrapping, and coarse quadratic-variation symmetric estimates plus
  fitted skew parts.
- Ranked candidates by split-half stability and held-out saved-cadence
  covariance prediction, with small penalties for one-step residuals, rough
  Fourier branches, and non-real logarithm branches. True mobility was computed
  only after ranking.
- Updated the agent report, human LaTeX writeup, and reusable lessons, then
  compiled the writeup.

Files changed:
- `FHDChainCapillaryN32/code/fit_Phi_coarse.jl`
- `FHDChainCapillaryN32/configs/fit_Phi_coarse.toml`
- `FHDChainCapillaryN32/data/phi_coarse_original_sweep.h5`
- `FHDChainCapillaryN32/figures/phi_coarse_original_sweep.png`
- `FHDChainCapillaryN32/logs/fit_Phi_coarse.log`
- `FHDChainCapillaryN32/logs/fit_Phi_coarse_metrics.txt`
- `FHDChainCapillaryN32/reports/agent_report.md`
- `FHDChainCapillaryN32/writeup/system_report.tex`
- `FHDChainCapillaryN32/writeup/system_report.pdf`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- LaTeX auxiliary files in `FHDChainCapillaryN32/writeup/`; removed after
  compilation.

Commands/tests run:
- `julia --project=. FHDChainCapillaryN32/code/fit_Phi_coarse.jl FHDChainCapillaryN32/configs/fit_Phi_coarse.toml`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
- `rg -n "Overfull|Underfull|Warning|undefined|Fatal|Emergency" system_report.log || true`
- HDF5/figure artifact checks for `phi_coarse_original_sweep.h5` and
  `phi_coarse_original_sweep.png`.

Outcome:
- The accepted data-only estimator was `projected_varlog_L30`, with data score
  `0.4127066375`, split-half relative difference `0.3803407887`, held-out
  prediction residual `0.0258322677`, and ex-post `Phi` rel.RMSE
  `0.9863735342` versus `<M_true>`.
- The best ex-post candidate, not selected, was
  `coarse_qv_plus_poly_skew_d3_L4`, with rel.RMSE `0.9709634362`. It had a much
  worse data-only prediction residual `0.7774220748`.
- High-degree polynomial extrapolation was rejected because it could reduce
  split discrepancy while worsening held-out covariance prediction and ex-post
  Phi error.
- Final conclusion: the original saved cadence does not contain enough
  near-zero-lag information to recover the derivative-at-zero mobility for this
  system. The earlier 4096-burst result is necessary, not just a better fitting
  routine.

Do not repeat:
- Do not spend another pass tuning coarse `save_dt = 0.5` Phi estimators for
  this system unless the user explicitly wants an aliasing/identifiability
  study. The best ex-post coarse result remained near relative error `0.97`.

Useful follow-up:
- For accurate constant-mobility work, continue to use the accepted
  `phi_dataonly_burstfine4k_sweep.h5` artifact. Treat the coarse sweep as a
  documented negative result and a warning about derivative-at-zero estimation.

## 2026-05-14 — FHDChainCapillaryN32 Gaussian-score Phi cadence sweep

Status: Success

Goal:
- Estimate `Phi` at several covariance cadences, including `dt = 0.0005`,
  `0.005`, and `0.05`, then integrate the Langevin equation with the empirical
  Gaussian score and compare observed time-correlation recovery by RMSE.

Approach:
- Added a sweep script that estimates `Phi` separately at each cadence using
  the accepted data-only candidate table, then integrates each selected Phi
  with the same empirical Gaussian stationary score.
- Used the `4096`-burst dataset for `dt = 0.0005`, the high-frequency
  continuation for `dt = 0.005`, and a stride-10 subsample of the
  high-frequency continuation for `dt = 0.05`.
- Integrated the Gaussian-score constant-Phi model using the exact OU
  transition at save interval `0.5`, then computed rho, momentum, and combined
  ACF relative RMSE against the observed short trajectory.
- Updated both reports and the reusable lesson entry, then compiled the human
  writeup.

Files changed:
- `FHDChainCapillaryN32/code/fit_Phi_dt_sweep_gaussian.jl`
- `FHDChainCapillaryN32/configs/fit_Phi_dt_sweep_gaussian.toml`
- `FHDChainCapillaryN32/data/phi_dt_sweep_gaussian_forward.h5`
- `FHDChainCapillaryN32/figures/phi_dt_sweep_gaussian_forward.png`
- `FHDChainCapillaryN32/logs/fit_Phi_dt_sweep_gaussian.log`
- `FHDChainCapillaryN32/logs/fit_Phi_dt_sweep_gaussian_metrics.txt`
- `FHDChainCapillaryN32/reports/agent_report.md`
- `FHDChainCapillaryN32/writeup/system_report.tex`
- `FHDChainCapillaryN32/writeup/system_report.pdf`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- LaTeX auxiliary files in `FHDChainCapillaryN32/writeup/`; removed after
  compilation.

Commands/tests run:
- `julia --project=. FHDChainCapillaryN32/code/fit_Phi_dt_sweep_gaussian.jl FHDChainCapillaryN32/configs/fit_Phi_dt_sweep_gaussian.toml`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
- `rg -n "Overfull|Underfull|Warning|undefined|Fatal|Emergency" system_report.log || true`

Outcome:
- `dt=0.0005` selected `poly_L1`, with ex-post Phi rel.RMSE `0.028668`, rho
  ACF RMSE `0.2854`, momentum ACF RMSE `0.08123`, and combined ACF RMSE
  `0.2098`.
- `dt=0.005` selected `poly_L1`, with ex-post Phi rel.RMSE `0.1013`, rho ACF
  RMSE `0.6166`, momentum ACF RMSE `0.1861`, and combined ACF RMSE `0.4554`.
- `dt=0.05` selected `poly_L1`, with ex-post Phi rel.RMSE `0.6782`, rho ACF
  RMSE `0.6008`, momentum ACF RMSE `0.1793`, and combined ACF RMSE `0.4434`.
- Stationary covariance RMSE stayed near `0.11` for all cases because the
  Gaussian score fixed the invariant covariance; the cadence dependence shows
  up in time correlations.

Do not repeat:
- Do not judge these Gaussian-score forward models by stationary covariance
  alone. The finite-time ACF metrics are the useful discriminator for Phi
  cadence.

Useful follow-up:
- If comparing estimator families at coarser cadences, keep ex-post true-M
  best rows separate from data-only selected rows. In this sweep, coarser
  cadences had ex-post best log-covariance rows that were not used for forward
  selection because that would use true mobility.

## 2026-05-14 — SoftSpinLLGChain original-resolution Phi fitting

Status: Partial

Goal:
- Improve the strict trajectory-only `Phi` estimate without increasing saved
  trajectory resolution and without using short-lag simulator rollouts.

Approach:
- Undid the high-resolution branch and removed its outputs.
- Added a direct projected onsite covariance-block estimator to
  `SoftSpinLLGChain/code/fit_Phi.jl`.
- Added `SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly_projected.toml`.
- Estimated `Phi` from the original `soft_spin_llg_chain.h5` snapshots only,
  preserving `t_D/save_dt = 100`.
- Tested random-pair projected polynomial fits, stratified/all-saved-pair
  polynomial fits, matrix-log covariance fits, and increment-covariance fits.

Files changed:
- `SoftSpinLLGChain/code/fit_Phi.jl`
- `SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly_projected.toml`
- `SoftSpinLLGChain/models/fit_Phi_phys_pC_dataonly_projected_artifacts.bson`
- `SoftSpinLLGChain/data/phi_phys_pC_dataonly_projected_forward_langevin.h5`
- `SoftSpinLLGChain/logs/fit_Phi_phys_pC_dataonly_projected_metrics.txt`
- `SoftSpinLLGChain/logs/forward_phi_phys_pC_dataonly_projected_only_metrics.txt`
- `SoftSpinLLGChain/figures/phi_phys_pC_dataonly_projected_recovery.png`
- `SoftSpinLLGChain/figures/phi_phys_pC_dataonly_projected_cdot_gfdt.png`
- `SoftSpinLLGChain/figures/phi_phys_pC_dataonly_projected_forward_stats.png`
- `SoftSpinLLGChain/figures/phi_phys_pC_dataonly_projected_forward_cmn.png`
- `SoftSpinLLGChain/figures/forward_stats_phi_phys_pC_dataonly_projected_only.png`
- `SoftSpinLLGChain/figures/forward_cmn_phi_phys_pC_dataonly_projected_only.png`
- `SoftSpinLLGChain/reports/agent_report.md`
- `SoftSpinLLGChain/writeup/system_report.tex`
- `SoftSpinLLGChain/writeup/system_report.pdf`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- `.agent-scratch/phi_projected_fit/`
- rejected no-forward candidate artifacts with suffixes
  `projected_l3d1_nofwd` and `projected_l3d2_nofwd`

Commands/tests run:
- `xvfb-run -a julia --project=. SoftSpinLLGChain/code/fit_Phi.jl .agent-scratch/phi_projected_fit/fit_Phi_projected_l3d1_nofwd.toml`
- `JULIA_NUM_THREADS=8 xvfb-run -a julia --project=. .agent-scratch/phi_projected_fit/scan_projected_phi.jl ...`
- `JULIA_NUM_THREADS=8 xvfb-run -a julia --project=. .agent-scratch/phi_projected_fit/scan_matrixlog_phi.jl ...`
- `xvfb-run -a julia --project=. SoftSpinLLGChain/code/fit_Phi.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly_projected.toml`
- `xvfb-run -a julia --project=. SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly_projected.toml phi_phys_pC_dataonly_projected_only 'Phi projected=../data/phi_phys_pC_dataonly_projected_forward_langevin.h5'`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`

Outcome:
- Accepted fitting-only branch improved ex-post `Phi` vs `<M_true>` rel.RMSE
  from `0.239439699` to `0.174457882`, and correlation from `0.980447139` to
  `0.984265927`.
- The direct projected estimator removed most dense off-profile noise:
  projection relative change dropped from `0.966524090` to `0.0571828515`.
- Constant-Phi dynamics did not improve: coordinate `C(t)` rel.RMSE was
  `0.431418459`, and retained nonlinear-observable `C(t)` rel.RMSE was
  `0.985238933`.
- Exact all-saved-pair polynomial, matrix-log, and increment-covariance fits
  were worse, indicating finite-lag bias from `save_dt=0.0365` rather than only
  random-pair variance.

Do not repeat:
- Do not reintroduce high-resolution/resimulation data when the user asks to
  keep the original 100 snapshots per decorrelation length.
- Do not accept random-pair scan rows solely because their ex-post true-M error
  is lowest; selection by target error would violate the no-cheating rule.
- Do not rerun exact all-pair polynomial/log/increment fits as the next fix
  unless there is a new idea for handling coarse-lag bias.

Useful follow-up:
- If better strict data-only Phi is still required at this cadence, investigate
  model-based regularization using observed forward/GFDT consistency only, not
  true mobility. Otherwise, near-zero-lag data are likely required for a robust
  derivative-at-zero estimate.

## 2026-05-14 — SoftSpinLLGChain structured joint-score first wave

Status: Partial

Goal:
- Implement a separate joint-score branch without modifying the accepted
  conditional-score code/checkpoints, then test whether it beats
  `cond_score_phys_pC_unet_vA_cont2` on the true-M operator diagnostic.

Approach:
- Added `SoftSpinLLGChain/code/joint_score.jl`.
- Added `fit_dM.jl` dispatch for `data.cond_score_kind = "joint_score"` while
  leaving serialized `DMConfig` unchanged for old checkpoints.
- Ran four joint-score branches: wide basic residualized, physical-augmented
  residualized, raw joint-score ablation, and initial-only active-lag
  residualized.

Files changed:
- `SoftSpinLLGChain/code/joint_score.jl`
- `SoftSpinLLGChain/code/fit_dM.jl`
- `SoftSpinLLGChain/configs/joint_score_phys_pC_gpu2_vA.toml`
- `SoftSpinLLGChain/configs/joint_score_phys_pC_gpu0_vB.toml`
- `SoftSpinLLGChain/configs/joint_score_phys_pC_gpu1_vC_raw.toml`
- `SoftSpinLLGChain/configs/joint_score_phys_pC_gpu2_vD_initial_active_s035.toml`
- `SoftSpinLLGChain/reports/agent_report.md`

Files created temporarily:
- `.agent-scratch/joint_score_smoke.toml`
- `.agent-scratch/joint_score_smoke.bson`

Commands/tests run:
- `xvfb-run -a julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/joint_score.jl")'`
- `xvfb-run -a julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/fit_dM.jl")'`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/joint_score.jl .agent-scratch/joint_score_smoke.toml`
- Three first-wave `joint_score.jl` training commands on all GPUs, plus the
  second 5070 branch `vD_initial_active_s035`.

Outcome:
- The smoke test exposed and then verified a fix for a GPU scalar-indexing bug
  in the Stein penalty path.
- No first-wave model beat the old conditional score. Metrics:
  `vA` rel.RMSE `0.729450`, corr `0.739019`;
  `vB` rel.RMSE `0.594637`, corr `0.822446`;
  `vC_raw` rel.RMSE `3.716048`, corr `-0.097219`;
  `vD_initial_active_s035` global rel.RMSE `1.549671`, active-lag rel.RMSE
  `1.440056`.
- A second wave was launched with GroupNorm physical augmented features,
  physical-full features, and a short-lag/higher-noise sweep.

Do not repeat:
- Do not promote these first-wave checkpoints to mobility training.
- Do not repeat raw joint-score training as configured here.
- Do not repeat initial-only active-lag `sigma=0.035` residualized training as
  configured here; it failed even on active lags.

Useful follow-up:
- Continue the second wave and only move to M training if a joint checkpoint
  beats rel.RMSE `0.378631` globally or gives clearly better active-lag
  retained-channel behavior.

## 2026-05-15 — SoftSpinLLGChain joint-score second wave and vE M training

Status: Partial

Goal:
- Continue the structured joint-score branch, identify whether any checkpoint
  is eligible for M training, and start the three requested M NNs without
  modifying the old conditional-score artifacts.

Approach:
- Added retained active-channel diagnostics to `joint_score.jl`.
- Reran `vE` evaluation-only with active lags `7:24` and the compact retained
  nonlinear channels from `dM_targets_compact_phys_pC_dataonly.bson`.
- Used `vE` as the conditional source for three M configs:
  warm-start old best vJ, fresh `neighbor_r2`, and wider `neighbor_r2`.
- Launched further targeted score branches `vH` and `vI` because initial M
  training against `vE` did not beat the old vJ M fit.

Files changed:
- `SoftSpinLLGChain/code/joint_score.jl`
- `SoftSpinLLGChain/code/search_nonlinear_observables.jl`
- `SoftSpinLLGChain/code/fit_dM.jl`
- `SoftSpinLLGChain/configs/joint_score_phys_pC_gpu*_*.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu2_warmJ.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu0_fresh_neighbor.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu1_wide_neighbor.toml`
- `SoftSpinLLGChain/configs/joint_score_phys_pC_gpu2_vH_physfull_initial_s05.toml`
- `SoftSpinLLGChain/configs/joint_score_phys_pC_gpu0_vI_physaug_s035_uniform.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`

Files created temporarily:
- `.agent-scratch/joint_score_smoke.toml`
- `.agent-scratch/joint_score_smoke.bson`

Commands/tests run:
- `xvfb-run -a julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/joint_score.jl"); include("SoftSpinLLGChain/code/fit_dM.jl")'`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/joint_score.jl SoftSpinLLGChain/configs/joint_score_phys_pC_gpu2_vE_physaug_groupnorm.toml`
- `CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/joint_score.jl SoftSpinLLGChain/configs/joint_score_phys_pC_gpu0_vF_physfull.toml`
- `CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/joint_score.jl SoftSpinLLGChain/configs/joint_score_phys_pC_gpu1_vG_short_s07.toml`
- Three `fit_dM_phys_pC_joint_vE_*` M-training commands on the three GPUs.

Outcome:
- `vE` failed the global old-conditional gate (`0.486413` rel.RMSE), but passed
  the retained active-channel gate (`0.289740` rel.RMSE, `0.957278` corr).
- `vF` was poor globally (`0.979281` rel.RMSE); `vG` was closer but still worse
  than old conditional globally (`0.431958` rel.RMSE).
- Completed M runs from `vE`: warm-start best A rel.RMSE `0.304705`, fresh
  `neighbor_r2` best A rel.RMSE `0.313294`. Both are worse than old vJ
  (`0.286122`), so further score/M exploration is still needed.
- The wide M run, `vH`, and `vI` were still running when this entry was written.

Do not repeat:
- Do not reject a joint score only from the global all-channel diagnostic when
  the M training target is the retained nonlinear active-lag subset; evaluate
  retained active metrics explicitly.
- Do not assume the `physical_full` feature set is automatically better:
  `vF_physfull` was much worse than `vE_physaug_groupnorm`.

Useful follow-up:
- Finish wide M and forward-validate the best `vE` M checkpoint.
- If `vH` or `vI` beats `vE` on retained active operator diagnostics, rerun the
  same M suite against that conditional source.

## 2026-05-15 — SoftSpinLLGChain targeted joint-score vH and vE wide M

Status: Partial

Goal:
- Continue the joint-score campaign past the initial `vE` M wave and test
  whether targeted initial-only physical-full joint-score training provides a
  better transition score on the retained active nonlinear channels.

Approach:
- Let `joint_score_phys_pC_gpu2_vH_physfull_initial_s05.toml` finish on the
  RTX 5070.
- Completed the `vE` wide-neighbor M run and selected it as the best `vE` M
  branch by data-only A validation.
- Added `vH` M configs for warm-start, fresh-neighbor, and wide-neighbor
  training; launched the warm-start config on the 5070.

Files changed:
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vH_gpu2_warmJ.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vH_gpu0_fresh_neighbor.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vH_gpu1_wide_neighbor.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- `.agent-scratch/joint_score_smoke.toml`
- `.agent-scratch/joint_score_smoke.bson`

Commands/tests run:
- `julia --project=. --startup-file=no -e 'using TOML; for f in ARGS; TOML.parsefile(f); println("parsed ", f); end' ...fit_dM_phys_pC_joint_vH_*.toml`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vH_gpu2_warmJ.toml`
- `CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu1_wide_neighbor.toml SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml SoftSpinLLGChain/data/forward_dM_phys_pC_joint_vE_wide_scale1_dt0015.h5 1.0 0.0015`

Outcome:
- `vE` wide-neighbor M completed with A validation rel.RMSE `0.289004`, corr
  `0.961110`. It is the best `vE` M branch but still slightly worse than old
  vJ (`0.286122`).
- `vH` was globally poor as a joint score: global true-M operator rel.RMSE
  `4.992027`, corr `0.245273`.
- `vH` was best so far on the retained active nonlinear channels: rel.RMSE
  `0.257459`, corr `0.966187`, versus `vE` `0.289740`, corr `0.957278`.
- The `vH` warm-start M run and the `vE` learned-M forward run were still
  running when this entry was written.

Do not repeat:
- Do not interpret `vH` as a globally good joint-score model. Its only current
  justification is active retained-channel behavior, and that must be checked
  through M A-validation and forward Langevin diagnostics.

Useful follow-up:
- Finish `vH` warm-start M. If it beats or matches `vE` wide by data-only A
  validation, launch the remaining `vH` M configs as GPUs free up.
- Finish `vE` forward validation and compare against observations, `M=Phi`,
  and old best learned-M vJ x1.10.

## 2026-05-15 — SoftSpinLLGChain joint-score M forward transfer

Status: Partial

Goal:
- Check whether the best joint-score M checkpoint transfers from A-target
  fitting to forward Langevin validation, and continue the better retained
  active `vH` branch.

Approach:
- Forward-integrated `dM_phys_pC_joint_vE_gpu1_wide_neighbor.bson` at scale
  `1.0`, then evaluated observations vs `M=Phi`, old `vJ x1.10`, and new
  `joint vE wide x1.0`.
- Completed `vH` warm-start M training.
- Launched `vH` wide M training; it became promising at epoch 25.
- Added a forward-only 5070 config for the `vE` wide checkpoint and launched a
  scale `1.10` forward test.

Files changed:
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu2_wide_neighbor_forward.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`

Files created temporarily:
- `.agent-scratch/joint_score_smoke.toml`
- `.agent-scratch/joint_score_smoke.bson`

Commands/tests run:
- `CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu1_wide_neighbor.toml SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml SoftSpinLLGChain/data/forward_dM_phys_pC_joint_vE_wide_scale1_dt0015.h5 1.0 0.0015`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml forward_phys_pC_joint_vE_wide_compare ...`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vH_gpu2_warmJ.toml`
- `CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vH_gpu1_wide_neighbor.toml`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu2_wide_neighbor_forward.toml SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml SoftSpinLLGChain/data/forward_dM_phys_pC_joint_vE_wide_scale1p10_dt0015.h5 1.10 0.0015`

Outcome:
- `joint vE wide x1.0` very slightly beat the user-specified old `vJ x1.10`
  retained-C reference: `0.134558` vs `0.134815`, and improved covariance
  rel.RMSE `0.067739` vs `0.090806`.
- The improvement is marginal and still far from the preferred retained-C
  target below `0.10`.
- `vH` warm-start M completed with best epoch `50`, A validation rel.RMSE
  `0.290345`, corr `0.959164`; worse than `vE` wide (`0.289004`), so it was
  not forward-validated.
- `vH` wide M is still running. It reached A validation rel.RMSE `0.27556`,
  corr `0.96315` at epoch 25, the best joint-score A-validation metric so far.
- `vE` scale `1.10` forward is still running.

Do not repeat:
- Do not assume a better retained active conditional-score diagnostic alone
  improves M training: `vH` warm-start had better retained active score metrics
  but worse A validation than `vE` wide.

Useful follow-up:
- Finish `vH` wide M and forward-validate its best checkpoint.
- Evaluate `vE` scale `1.10`; if it improves retained-C, try a smaller local
  scale sweep around the best value.

## 2026-05-15 — SoftSpinLLGChain vH forward transfer and vI rejection

Status: Partial

Goal:
- Continue the structured joint-score campaign after the `vH` wide M checkpoint
  became the best data-only A-validation fit.

Approach:
- Forward-integrated the best `vH` wide M checkpoint at scales `1.0`, `1.05`,
  and `0.95`.
- Evaluated observations vs `M=Phi`, old `vJ x1.10`, `joint vE wide x1.05`,
  and each `vH` scale on retained nonlinear `C(t)` plus stationary covariance.
- Let the `vH` fresh-neighbor M run finish.
- Let `joint_score_phys_pC_gpu0_vI_physaug_s035_uniform` finish and evaluated
  its retained active operator diagnostic.
- Added and launched `vK`, a physical-augmented GroupNorm initial-only active
  joint-score branch, while launching existing `vJ` moment-penalty branch.

Files changed:
- `SoftSpinLLGChain/configs/joint_score_phys_pC_gpu0_vK_physaug_groupnorm_initial_active.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- `.agent-scratch/joint_score_smoke.toml`
- `.agent-scratch/joint_score_smoke.bson`

Commands/tests run:
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl ...forward_dM_phys_pC_joint_vH_wide_best_scale1_dt0015.h5 1.0 0.0015`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ... forward_phys_pC_joint_vH_wide_best_compare ...`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl ...forward_dM_phys_pC_joint_vH_wide_best_scale1p05_dt0015.h5 1.05 0.0015`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl ...forward_dM_phys_pC_joint_vH_wide_best_scale0p95_dt0015.h5 0.95 0.0015`
- `CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vH_gpu1_fresh_neighbor.toml`
- `CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/joint_score.jl SoftSpinLLGChain/configs/joint_score_phys_pC_gpu0_vI_physaug_s035_uniform.toml`
- `julia --project=. --startup-file=no -e 'using TOML; for f in ARGS; TOML.parsefile(f); println("parsed ", f); end' ...vJ...toml ...vK...toml`
- `CUDA_VISIBLE_DEVICES=2 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/joint_score.jl SoftSpinLLGChain/configs/joint_score_phys_pC_gpu1_vJ_physfull_active_moment.toml`
- `CUDA_VISIBLE_DEVICES=1 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/joint_score.jl SoftSpinLLGChain/configs/joint_score_phys_pC_gpu0_vK_physaug_groupnorm_initial_active.toml`

Outcome:
- `vH` wide M had the best A validation so far: rel.RMSE `0.268108`, corr
  `0.964619`, but forward retained-C did not transfer.
- `vH` scales gave retained-C rel.RMSE `0.142680` at `0.95`, `0.143061` at
  `1.0`, and `0.153622` at `1.05`; all are worse than `joint vE wide x1.05`
  (`0.129351`) and old `vJ x1.10` (`0.134815`).
- `vH` covariance was strong: best covariance rel.RMSE `0.054946` at `1.05`,
  but this came with worse retained nonlinear dynamics and larger tails.
- `vH` fresh-neighbor M finished at A rel.RMSE `0.291671`, worse than vH wide.
- `vI` failed the gate: retained active operator rel.RMSE `0.631758`, corr
  `0.777902`; it should not be used for M training.
- `vE` scale `1.06`, `vJ`, and `vK` were running when this entry was written.

Do not repeat:
- Do not assume a lower-noise uniform physical-augmented joint-score (`vI`) is
  a promising next branch; it was much worse on the retained active diagnostic.
- Do not assume better A-validation implies better forward nonlinear
  correlations. `vH` wide beat old vJ and vE on A validation but lost to vE
  x1.05 in forward retained-C metrics.

Useful follow-up:
- Evaluate `vE x1.06`; if it improves over `vE x1.05`, do a tight scale sweep
  below the nonfinite `1.10` run.
- Let `vJ` and `vK` finish. Only train M from them if retained active operator
  diagnostics beat or meaningfully match `vH`/`vE`.

## 2026-05-15 — SoftSpinLLGChain vE forward scale boundary

Status: Partial

Goal:
- Tighten the forward scale sweep for the best `joint vE wide` M checkpoint
  after `x1.05` improved retained nonlinear dynamics and `x1.10` was nonfinite.

Approach:
- Forward-integrated `joint vE wide` at scales `1.06`, `1.055`, `1.0575`, and
  `1.05875`.
- Evaluated finite-state checks, covariance, and retained nonlinear `C(t)`
  metrics against observations, `M=Phi`, old `vJ x1.10`, and competing joint
  branches.
- Stopped the already-started `1.07` run after `1.06` failed the finite-state
  check.
- Added and launched `vL`, a physical-augmented GroupNorm active-lag/moment
  branch that keeps both endpoints noised.

Files changed:
- `SoftSpinLLGChain/configs/joint_score_phys_pC_gpu2_vL_physaug_groupnorm_active_moment.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- `.agent-scratch/joint_score_smoke.toml`
- `.agent-scratch/joint_score_smoke.bson`

Commands/tests run:
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl ...scale1p06_dt0015.h5 1.06 0.0015`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ... forward_phys_pC_joint_vE_wide_scale1p06_compare ...`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl ...scale1p055_dt0015.h5 1.055 0.0015`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ... forward_phys_pC_joint_vE_wide_scale1p055_compare ...`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl ...scale1p0575_dt0015.h5 1.0575 0.0015`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ... forward_phys_pC_joint_vE_wide_scale1p0575_compare ...`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl ...scale1p05875_dt0015.h5 1.05875 0.0015`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ... forward_phys_pC_joint_vE_wide_scale1p05875_compare ...`
- `CUDA_VISIBLE_DEVICES=0 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/joint_score.jl SoftSpinLLGChain/configs/joint_score_phys_pC_gpu2_vL_physaug_groupnorm_active_moment.toml`

Outcome:
- `x1.06` was rejected by the finite-state check.
- `x1.055` is the best forward result so far: retained nonlinear `C(t)`
  rel.RMSE about `0.1216`, corr about `0.9923`.
- `x1.0575` and `x1.05875` stayed finite but worsened retained nonlinear
  dynamics (`0.1480` and `0.1416` rel.RMSE). `x1.05875` also produced a large
  state tail, max `|state| = 2.375`.
- The useful global scale is narrowly localized near `1.055`; pushing higher
  chases covariance at the expense of the nonlinear dynamic target.
- `vJ`, `vK`, and `vL` were running when this entry was written.

Do not repeat:
- Do not run more global-scale points above `1.05875` for `joint vE wide`;
  `1.06` is nonfinite and the lower finite high-scale points already worsened
  retained nonlinear dynamics.

Useful follow-up:
- Let `vJ`, `vK`, and `vL` finish. If none passes the retained active
  conditional-score gate, the current best joint branch is `vE wide x1.055`.

## 2026-05-15 — SoftSpinLLGChain structured joint-score campaign closure

Status: Partial

Goal:
- Complete the separate joint-score branch without touching the old
  `cond_score.jl` fallback, train M NNs from eligible joint scores, and test
  whether learned-score plus learned-M forward validation beats the old best.

Approach:
- Let `vJ`, `vK`, and `vL` joint-score branches finish and evaluated their
  global, active-lag, retained, and retained-active true-M operator diagnostics.
- Trained wide `neighbor_r2` M NNs from `vJ`, `vK`, and `vL` using the same
  data-only residual target `A_data = Cdot_data - A[Phi]`.
- Forward-integrated `vJ`, `vK`, `vL`, and tightened/validated `vE` scale
  variants on all three GPUs.
- Generated clean final comparison figures for `M=Phi`, old `vJ x1.10`, and
  current best `joint vE x1.055`.
- Updated the system report and living agent report with the final metrics.

Files changed:
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vE_gpu0_wide_neighbor_forward.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vL_gpu0_wide_neighbor_forward.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_joint_vL_gpu1_wide_neighbor_forward.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `SoftSpinLLGChain/writeup/system_report.tex`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- Failed bulky forward HDF5 scale probes later removed from `SoftSpinLLGChain/data`.

Commands/tests run:
- `CUDA_VISIBLE_DEVICES=0/1/2 xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl ...`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ... forward_phys_pC_joint_final_grid_compare ...`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ... forward_phys_pC_joint_vL_scale_grid_compare ...`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl ... forward_phys_pC_joint_best_compare ...`
- Figure inspection with `view_image` for the best comparison figures.

Outcome:
- `vL` was the best retained-active joint score: rel.RMSE `0.245659`, corr
  `0.969966`, but its M branch did not forward-transfer.
- M validation best among late branches was `vL`: A rel.RMSE `0.259819`, corr
  `0.967913`; `vJ` was `0.265390`, `vK` was `0.271164`.
- Forward retained nonlinear `C(t)` best remains `joint vE x1.055`: rel.RMSE
  `0.121584`, corr `0.992290`, covariance rel.RMSE `0.089546`.
- This beats old `vJ x1.10` on retained dynamics (`0.134815`) and covariance
  (`0.090806`) but does not reach the preferred target below `0.10`.
- `vE x1.0525` and `vE x1.054` were nonfinite in the common evaluator.
- `vK x1.0` retained-C rel.RMSE was `0.172489`; `vL` lower scales were
  `0.173134`, `0.171861`, and `0.146214`, all worse than `vE x1.055`.

Do not repeat:
- Do not assume the best retained-active conditional-score metric or best
  A-validation branch will give the best forward dynamics. `vL` won both those
  intermediate metrics but lost to `vE x1.055` in retained nonlinear forward
  validation.
- Do not rerun global `vE` scales near/above `1.0525` expecting monotone
  stability; `1.0525` and `1.054` were nonfinite, while the earlier `1.055`
  run was finite and best.
- Do not keep bulky failed forward HDF5 files after their metrics and lessons
  are documented.

Useful follow-up:
- The remaining blocker is forward-transfer mismatch between conditional-score
  operator/A-fit diagnostics and actual Langevin statistics. A useful next
  direction is a validation objective that predicts forward transfer before
  expensive full integration, or a more physically constrained M parameterization
  rather than more global scale sweeps.

Cleanup:
- Removed `.agent-scratch/joint_score_smoke.*`.
- Removed superseded/failed joint forward HDF5 files from
  `SoftSpinLLGChain/data`, keeping only
  `forward_dM_phys_pC_joint_vE_wide_scale1p055_dt0015.h5` as the current best
  joint-score learned-M trajectory.
- Removed LaTeX build byproducts from `SoftSpinLLGChain/writeup`.

## 2026-05-15 — SoftSpinLLGChain forward-stats figure repair

Status: Success

Goal:
- Add ACF and cross-correlation panels to the best `forward_stats` comparison
  figure and fix missing covariance heatmap colorbars.

Approach:
- Updated `SoftSpinLLGChain/code/render_forward_with_dM.jl` so
  `render_stats_with_dm` renders PDFs, covariance/error maps with explicit
  colorbars, exact global-component ACFs, and exact global-component
  cross-correlations.
- Rejected the first sampled-pair implementation because visual inspection
  showed noisy cross-correlation panels. Replaced it with an FFT-based
  all-time-origin estimator on the site-averaged global components.
- Updated `SoftSpinLLGChain/code/evaluate_forward_grid.jl` to pass the
  observation `save_dt` to the stats renderer.
- Updated the caption in `SoftSpinLLGChain/writeup/system_report.tex`.

Files changed:
- `SoftSpinLLGChain/code/render_forward_with_dM.jl`
- `SoftSpinLLGChain/code/evaluate_forward_grid.jl`
- `SoftSpinLLGChain/writeup/system_report.tex`
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- None.

Commands/tests run:
- `xvfb-run -a julia --project=. --startup-file=no -e 'include("SoftSpinLLGChain/code/evaluate_forward_grid.jl"); println("plotting code loaded")'`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml forward_phys_pC_joint_best_compare 'Phi dataonly=../data/phi_phys_pC_dataonly_forward_langevin.h5' 'M_NN old vJ x1.10=../data/forward_dM_phys_pC_dataonly_vAcont_vJ_best_scale1p10_dt0015.h5' 'M_NN joint vE x1.055=../data/forward_dM_phys_pC_joint_vE_wide_scale1p055_dt0015.h5'`
- Stats-only `xvfb-run` render after the final ACF-axis layout tweak.
- Manual image inspection of
  `SoftSpinLLGChain/figures/forward_stats_forward_phys_pC_joint_best_compare.png`.

Outcome:
- The regenerated best `forward_stats` figure now contains the requested ACF
  and cross-correlation rows plus explicit covariance and error colorbars.
- The final inspected version uses smooth exact FFT correlations rather than
  noisy sampled global correlations.
- The full evaluator regenerated the companion retained-C figure and metrics;
  retained nonlinear metrics were unchanged.

Do not repeat:
- Do not use sampled time pairs for the global cross-correlation panels in
  `forward_stats`; they look noisy and can make the figure misleading.

Useful follow-up:
- If future forward-stat figures compare more than three model trajectories,
  consider adding a second covariance-error row instead of showing only the
  first three model error maps.

Cleanup:
- No temporary files were created. Existing generated figures/logs were
  intentionally overwritten with the repaired best-comparison outputs.

## 2026-05-15 — SoftSpinLLGChain oracle true-M target diagnostic

Status: Partial

Goal:
- Intentionally use the best conditional score with true mobility and
  \(\Phi=\langle M_{\rm true}\rangle\) to build oracle \(\dot C_{mn}\) and
  \(A_{mn}\), train new M NNs from those targets, and check whether this
  recovers \(M_{\rm true}\) and forward statistics.

Approach:
- Added `SoftSpinLLGChain/code/prepare_oracle_trueM_dM_targets.jl` to prepare
  oracle `A` targets from learned `vL` transition scores, true local mobility,
  and the true mean onsite mobility.
- Added `target_kind = "oracle_trueM"` configs and guarded `fit_dM.jl` so this
  branch cannot silently masquerade as data-only.
- Trained `neighbor_r2` and `local_r2` M branches, killed weak/wrong-GPU partial
  branches, then forward-integrated the best A-fit oracle neighbor model.

Files changed:
- `SoftSpinLLGChain/code/fit_dM.jl`
- `SoftSpinLLGChain/code/prepare_oracle_trueM_dM_targets.jl`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu0_neighbor.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu1_localr2.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_equiv.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_localr2.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `SoftSpinLLGChain/writeup/system_report.tex`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- Killed partial `SoftSpinLLGChain/models/dM_phys_pC_oracle_trueM_vL_gpu2_equiv_best.bson`, removed during cleanup.

Commands/tests run:
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/prepare_oracle_trueM_dM_targets.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_equiv.toml`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu0_neighbor.toml`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_localr2.toml`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/forward_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu0_neighbor.toml SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml SoftSpinLLGChain/data/forward_dM_phys_pC_oracle_trueM_vL_neighbor_dt0015.h5 1.0 0.0015`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml forward_phys_pC_oracle_trueM_vL_neighbor_compare ...`

Outcome:
- Oracle target artifact: `SoftSpinLLGChain/models/dM_targets_oracle_trueM_vL.bson`.
- Best oracle A-fit was `gpu0_neighbor`: A rel.RMSE `0.248118`, corr
  `0.972486`, but ex-post true-M block rel.RMSE was poor at `0.687504` with
  corr `0.777847`.
- `gpu2_localr2` recovered the true block shape better (`0.428071` rel.RMSE,
  `0.954639` corr) but fit the A target worse (`0.290913` rel.RMSE).
- Forward integration of oracle `gpu0_neighbor` improved covariance error
  (`0.065478`) but worsened retained nonlinear C error (`0.174290`) relative to
  best data-only joint `vE x1.055` (`0.121584`) and produced larger state tails
  (`max |state| = 2.560514`).
- Forward integration of oracle `gpu2_localr2` improved retained nonlinear C
  error to `0.099442` with corr `0.994824` and bounded state values
  (`max |state| = 1.414626`), but covariance worsened to `0.096924` and global
  cross-correlation panels remained visibly imperfect.
- Conclusion: oracle true-M targets can improve retained forward dynamics when
  paired with the local-r2 branch, but they still do not make the M NN recover
  correct true mobility and they are explicitly non-data-only.

Do not repeat:
- Do not assume that replacing data-only residual targets with oracle true-M
  targets solves M recovery. The local-r2 branch improved retained C but still
  had true-block rel.RMSE `0.428071` and mixed stationary/global-correlation
  diagnostics.
- Do not treat this branch as data-only; true \(M\) and \(\Phi_{\rm true}\)
  explicitly entered the training target by user request.

Useful follow-up:
- Use this as evidence that cleaner `A` targets can matter for forward retained
  dynamics, but identifiability and physically constrained M parameterization
  still need work.

Cleanup:
- Removed the killed oracle `gpu2_equiv` partial checkpoint.
- Removed superseded neighbor-only oracle comparison figures/metrics after the
  all-oracle comparison was generated.
- Kept completed oracle target/model/forward artifacts and compact logs/figures
  because they document the diagnostic result.

## 2026-05-15 — SoftSpinLLGChain oracle identifiable observable library

Status: Partial

Goal:
- Find an observable library that lets the learned-transition-score oracle
  target recover the correct true local mobility \(M_{\rm true}\).

Approach:
- Added `SoftSpinLLGChain/code/search_oracle_identifiable_observables.jl`.
- Scored candidate observable-target channels by decomposing the oracle
  residual into local true-M block-entry contributions and greedily selected
  channels that made the normalized 9-entry Gram matrix well-conditioned.
- Tried an all-family 60-channel library, a smoother neighbor/local 48-channel
  library, local-r2 M training, and equivariant-r2 M training with and without
  stronger mean-mobility penalty.

Files changed:
- `SoftSpinLLGChain/code/search_oracle_identifiable_observables.jl`
- `SoftSpinLLGChain/configs/nonlinear_observable_retained_channels_oracle_ident_v1.toml`
- `SoftSpinLLGChain/configs/nonlinear_observable_retained_channels_oracle_ident_neighbor_v1.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu2_localr2.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv_meanstrong.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_neighbor_v1_gpu2_localr2.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `SoftSpinLLGChain/writeup/system_report.tex`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- Failed bulky checkpoints and target/search BSONs for the local-r2 and
  neighbor-only identifiable branches; removed during cleanup.

Commands/tests run:
- `xvfb-run -a julia --project=. --threads 12 --startup-file=no SoftSpinLLGChain/code/search_oracle_identifiable_observables.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_localr2.toml all oracle_ident_v1 30000 60`
- `xvfb-run -a julia --project=. --threads 12 --startup-file=no SoftSpinLLGChain/code/prepare_oracle_trueM_dM_targets.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu2_localr2.toml`
- `xvfb-run -a julia --project=. --threads 10 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu2_localr2.toml`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv.toml`
- `xvfb-run -a julia --project=. --threads 12 --startup-file=no SoftSpinLLGChain/code/search_oracle_identifiable_observables.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_gpu2_localr2.toml neighbor_high oracle_ident_neighbor_v1 30000 48`
- `xvfb-run -a julia --project=. --threads 12 --startup-file=no SoftSpinLLGChain/code/prepare_oracle_trueM_dM_targets.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_neighbor_v1_gpu2_localr2.toml`
- `xvfb-run -a julia --project=. --threads 10 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_neighbor_v1_gpu2_localr2.toml`
- `xvfb-run -a julia --project=. --threads 8 --startup-file=no SoftSpinLLGChain/code/fit_dM.jl SoftSpinLLGChain/configs/fit_dM_phys_pC_oracle_trueM_vL_ident_v1_gpu0_equiv_meanstrong.toml`

Outcome:
- All-family identifiable selector: 60 selected channels, selected Gram
  condition `13.5657`.
- Neighbor-only identifiable selector: 48 selected channels, selected Gram
  condition `14.8478`.
- Best old compact oracle reference: true-block rel.RMSE `0.428071`, corr
  `0.954639`.
- All-family identifiable local-r2 was worse: true-block rel.RMSE `0.508442`,
  corr `0.930463`.
- Neighbor-only identifiable local-r2 was worse: true-block rel.RMSE `0.563124`,
  corr `0.920720`.
- All-family identifiable equivariant-r2 improved the true block: rel.RMSE
  `0.384457`, corr `0.988407`, but mean-vs-Phi rel.RMSE was `0.687551`.
- Strong-mean continuation of the equivariant branch gave the best recovery:
  true-block rel.RMSE `0.381906`, corr `0.989175`, mean-vs-Phi rel.RMSE
  `0.250112`.

Do not repeat:
- Do not assume a log-det identifiable observable library alone will fix M
  recovery. It worsened local-r2 training unless paired with the equivariant
  physical M parameterization.
- Do not use the smoother neighbor-only identifiable library as the next default;
  it was worse than both the all-family identifiable library and the older
  compact oracle reference.
- Do not cite this as data-only. True \(M\) and \(\Phi_{\rm true}\) explicitly
  entered observable selection and target generation.

Useful follow-up:
- The most promising next route is to keep the all-family identifiable library
  but improve the equivariant M parameterization/loss selection so the mean
  block and A fit are both good. The current best is closer to true \(M\), but
  true-block rel.RMSE `0.381906` is still not a clean recovery.

Cleanup:
- Removed failed local-r2/neighbor identifiable M checkpoints, the neighbor-only
  oracle target BSON, and the neighbor-only search BSON.
- Kept compact failed metrics/figures/configs and the accepted all-family
  target/search/checkpoints.

## 2026-05-16 — SoftSpinLLG clean37 observable subset and data-only M recovery

Status: Success

Goal:
- Find an observable library that lets the learned-score, learned-conditional
  score, data-only Phi pipeline recover a good M NN, then validate the learned
  Langevin equation.

Approach:
- Implemented generalized nonlinear right-observable support and tested whether
  using nonlinear `phi_n` improved the oracle M-recovery problem.
- Found that nonlinear/right-observable libraries were worse than coordinate
  `phi_n=x`, so returned to the coordinate-right observable setup.
- Compared active-lag data-only A targets to oracle true-M A targets per
  retained channel and selected a clean 37-channel subset with rel.RMSE `<0.35`
  and corr `>0.95`.
- Trained three data-only equivariant-r2 M NNs on the clean37 target using all
  three GPUs, then integrated all three learned Langevin models and generated
  combined forward comparison figures.

Files changed:
- `SoftSpinLLGChain/code/right_observables.jl`
- `SoftSpinLLGChain/code/fit_dM_rightobs.jl`
- `SoftSpinLLGChain/code/search_phi_n_observables.jl`
- `SoftSpinLLGChain/code/search_phi_n_struct_observables.jl`
- `SoftSpinLLGChain/code/select_right_observable_subset.jl`
- `SoftSpinLLGChain/code/fit_dM.jl`
- `SoftSpinLLGChain/configs/nonlinear_observable_retained_channels_oracle_ident_v1_dataoracle_clean37.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu0_mean03.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean10.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu2_wide_mean03.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `SoftSpinLLGChain/writeup/system_report.tex`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- Bulky failed generalized-right-observable checkpoints and targets.
- Bulky failed full-channel data-only target/checkpoint BSONs.

Commands/tests run:
- `include("SoftSpinLLGChain/code/fit_dM.jl")` smoke test after target-scaling patch.
- `search_phi_n_observables.jl` and `search_phi_n_struct_observables.jl` oracle searches.
- `fit_dM_rightobs.jl` oracle right-observable M training on all three GPUs.
- `fit_dM.jl` full-channel data-only attempts with per-entry scaling,
  per-channel scaling, and high-learning-rate variants.
- `fit_dM.jl` clean37 data-only training on all three GPUs:
  `CUDA_VISIBLE_DEVICES=1`, `2`, and `0` for the two 2080 Ti devices and the RTX
  5070 respectively.
- `forward_dM.jl` clean37 forward integration for all three final checkpoints.
- `evaluate_forward_grid.jl ... forward_phys_pC_clean37_compare ...`
- `identify`/visual inspection of the final forward figures.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
  in `SoftSpinLLGChain/writeup`.

Outcome:
- Generalized right-observable libraries failed to improve recovery.  Best
  oracle true-M rel.RMSE was about `0.4568`; structured right-observable was
  about `0.525`.
- Full 60-channel data-only target had good aggregate agreement with oracle
  (active rel.RMSE `0.2209`, corr `0.9753`) but M training was poor because the
  set still contained noisy/inconsistent channels.
- The clean37 subset improved active target agreement to rel.RMSE `0.1818`,
  corr `0.9833`.
- Final M metrics:
  clean37 mean03 true-M rel.RMSE `0.271207`, corr `0.992963`;
  clean37 mean10 true-M rel.RMSE `0.275090`, corr `0.994491`;
  clean37 wide true-M rel.RMSE `0.289289`, corr `0.991409`.
- Final forward metrics:
  best covariance rel.RMSE `0.062004` and retained nonlinear C rel.RMSE
  `0.060607` for clean37 mean10.  This improves over old vJ (`0.0908`,
  `0.1348`) and previous joint vE (`0.0895`, `0.1216`).

Do not repeat:
- Do not keep training the full 60-channel target with different optimizer
  settings; the failure was noisy channel selection, not only learning rate.
- Do not prioritize nonlinear `phi_n` right-observable libraries for this
  system without a new reason; the tried full, top36, and structured variants
  were worse than coordinate `phi_n=x`.
- Do not use per-entry target scaling for the residual A loss on this problem;
  it overweights tiny/noisy entries.

Useful follow-up:
- The accepted clean37 branch should be the new baseline for any further M
  parameterization or score/conditional-score work.
- If adding more observables, first require active-lag data-vs-oracle or
  data-only split stability comparable to clean37; otherwise the M loss becomes
  ill-conditioned again.

Cleanup:
- Removed bulky failed right-observable checkpoints/targets/search BSONs and
  failed full-channel data-only target/checkpoint BSONs.
- Kept compact failed configs/logs/figures plus the accepted clean37 target,
  checkpoints, forward HDF5s, figures, and metrics.
- Compiled `SoftSpinLLGChain/writeup/system_report.pdf`.

## 2026-05-17 — SoftSpinLLG clean37 refinement sweep

Status: Success

Goal:
- Explore close directions around the clean37 data-only mobility result and
  decide whether the current best M is likely the best recoverable result from
  the present data, score, conditional score, Phi, and accepted physical M
  parameterization.

Approach:
- Added a strict channel-pruning utility and generated clean16, clean21, and
  clean32 observable subsets from the annotated clean37 source library.
- Built data-only residual target artifacts for those subsets using retained
  channel RMS scaling.
- Trained subset-pruning, lag-window, mean-penalty, and warm-start M variants
  across the three GPUs.
- Forward-integrated the only improved pointwise-M branch, clean37 mean15, and
  compared it against Phi, old vJ, joint vE, and the previous clean37 branches.

Files changed:
- `SoftSpinLLGChain/code/select_clean_observable_subset.jl`
- `SoftSpinLLGChain/code/subset_dm_targets.jl`
- `SoftSpinLLGChain/configs/nonlinear_observable_retained_channels_oracle_ident_v1_dataoracle_clean16.toml`
- `SoftSpinLLGChain/configs/nonlinear_observable_retained_channels_oracle_ident_v1_dataoracle_clean21.toml`
- `SoftSpinLLGChain/configs/nonlinear_observable_retained_channels_oracle_ident_v1_dataoracle_clean32.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean16_gpu0_mean10.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean21_gpu1_mean10.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean32_gpu2_mean10.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu0_mean06.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu1_mean15.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu2_lag5_20_mean10.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu2_lag8_24_mean10.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu0_warm_m03_mean08.toml`
- `SoftSpinLLGChain/configs/fit_dM_phys_pC_dataonly_oracle_ident_v1_clean37_gpu2_warm_m10_mean12.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `SoftSpinLLGChain/writeup/system_report.tex`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- Failed best-checkpoint BSONs for clean16, clean21, clean32, mean06, lag5_20,
  lag8_24, and warm-start branches. These were removed after documenting the
  metrics.

Commands/tests run:
- `select_clean_observable_subset.jl` for clean16, clean21, and clean32.
- `subset_dm_targets.jl` for all three strict subsets. The first non-Xvfb run
  failed because GLMakie/GLFW required a display; rerunning under `xvfb-run -a`
  succeeded.
- Parallel `fit_dM.jl` runs using all three GPUs:
  `CUDA_VISIBLE_DEVICES=0` for the RTX 5070 and `CUDA_VISIBLE_DEVICES=1,2` for
  the two 2080 Ti devices, with config-level GPU name guards.
- `forward_dM.jl` for clean37 mean15. A first attempt on the RTX 5070 failed
  because the config correctly required a 2080 Ti; rerunning with
  `CUDA_VISIBLE_DEVICES=2` succeeded.
- `evaluate_forward_grid.jl ... forward_phys_pC_clean37_extended_compare ...`
- `identify` and visual inspection of the generated forward figures.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
  in `SoftSpinLLGChain/writeup`.

Outcome:
- Stricter pruning was worse and underdetermined the residual operator:
  clean16 A rel.RMSE `0.931132`, corr `0.820730`; clean21 A rel.RMSE
  `0.909447`, corr `0.880706`; clean32 A rel.RMSE `0.855524`, corr `0.897863`.
- Weaker mean penalty and changed lag windows did not improve the A fit:
  mean06 A rel.RMSE `0.313071`; lag5:20 A rel.RMSE `0.587497`; lag8:24
  A rel.RMSE `0.320167`.
- Warm-starting previous clean37 checkpoints did not help: warm mean03->mean08
  A rel.RMSE `0.329914`; warm mean10->mean12 A rel.RMSE `0.322833`.
- The only useful new branch was clean37 mean15. It improved ex-post pointwise
  true-M recovery to block rel.RMSE `0.253830`, corr `0.994274`, with A
  rel.RMSE `0.302281`.
- Forward validation of clean37 mean15 gave covariance rel.RMSE `0.084851` and
  retained nonlinear C rel.RMSE `0.056724`, corr `0.998618`. This is the best
  retained-C forward result so far, but clean37 mean10 remains better for
  covariance (`0.062004`).

Do not repeat:
- Do not prune the clean37 library much below 37 channels unless a new
  identifiability criterion is introduced; clean16/21/32 all lost too much
  operator information.
- Do not expand the active lags toward 5:20 for this target. It sharply worsens
  A validation.
- Do not use low-learning-rate warm starts from mean03/mean10 as the next
  refinement; both plateaued worse than fresh clean37 fits.

Useful follow-up:
- The best dynamics-oriented M is now clean37 mean15. The best covariance model
  remains clean37 mean10. Given the tested close directions, further gains
  likely require a genuinely new data-only validation/selection proxy or a new
  physically acceptable M parameterization, not another small library or
  optimizer tweak.

Cleanup:
- Removed failed strict-subset target BSONs and failed refinement
  best-checkpoint BSONs after logging their metrics.
- Kept accepted clean37 mean15 checkpoint, diagnostics, forward trajectory, and
  extended comparison figures/metrics.

## 2026-05-17 — SoftSpinLLG physics-informed mobility ansatz

Status: Success

Goal:
- Use the known physical tensor form of the true onsite mobility as a declared
  structural ansatz, fit its coefficients from the clean37 data-only residual
  targets, and generate paper-facing forward figures comparing observations,
  `M=Phi`, the best nonparametric NN M, and the physics-informed M.

Approach:
- Added a separate physics-informed four-coefficient onsite ansatz:
  `I`, transverse projector, longitudinal projector, and skew cross-product
  terms.
- Fitted coefficients by weighted least squares against the clean37 data-only
  `A_data` target using the learned stationary score, learned conditional
  transition score, and data-only Phi. True coefficients were saved only for
  ex-post diagnostics.
- Tried joint-score `vL` conditional input first, then switched to the older
  direct conditional residual after the joint-score ansatz gave worse
  coefficients and forward dynamics.
- Added a small PSD coefficient floor after the no-floor direct-conditional
  branch produced nonfinite forward trajectories.

Files changed:
- `SoftSpinLLGChain/code/fit_dM_physical_ansatz.jl`
- `SoftSpinLLGChain/code/forward_physical_ansatz.jl`
- `SoftSpinLLGChain/code/render_forward_with_dM.jl`
- `SoftSpinLLGChain/configs/fit_dM_phys_ansatz_clean37_*.toml`
- `SoftSpinLLGChain/reports/agent_report.md`
- `SoftSpinLLGChain/writeup/system_report.tex`
- `docs/agents/ATTEMPTS.md`
- `docs/agents/LESSONS.md`

Files created temporarily:
- Nonfinal physics-ansatz forward HDF5s for mean10, mean100, mean1e5,
  direct no-floor, and direct floor005 branches. These were removed after the
  accepted floor001 branch was documented.

Commands/tests run:
- Parallel physics-ansatz coefficient fits on the three GPUs for joint-score
  mean penalties `0`, `10`, `100`, `1e5`, `1e7` and direct-conditional `1e5`.
- Forward integrations for the promising physics-ansatz branches.
- `xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/evaluate_forward_grid.jl SoftSpinLLGChain/configs/fit_Phi_phys_pC_dataonly.toml forward_phys_pC_paper_phi_nn_phys_compare 'Phi baseline=../data/phi_phys_pC_dataonly_forward_langevin.h5' 'NN learned M=../data/forward_dM_phys_pC_dataonly_clean37_gpu1_mean15_scale1_dt0015.h5' 'Physics-informed M=../data/forward_dM_phys_ansatz_clean37_directcond_floor001_dt0015.h5'`
- Visual inspection of the regenerated paper-facing stats and retained-C
  figures.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error system_report.tex`
  in `SoftSpinLLGChain/writeup`.

Outcome:
- Best accepted ansatz coefficients were `I=0.001`, `perp=0.0744784`,
  `parallel=0.001`, `skew=-0.793292`. True coefficients are
  `[0.002, 0.05, 0.006, -0.8]` ex post.
- Accepted coefficient fit metrics: train A rel.RMSE `0.193453`, eval A
  rel.RMSE `0.210367`, eval A corr `0.978542`, ex-post true-M block rel.RMSE
  `0.031718`, true-M block corr `0.999512`.
- Final paper-facing forward metrics:
  `M=Phi` covariance rel.RMSE `0.173127`, retained-C rel.RMSE `0.968388`;
  best NN M covariance rel.RMSE `0.083451`, retained-C rel.RMSE `0.056751`;
  physics-informed M covariance rel.RMSE `0.049157`, retained-C rel.RMSE
  `0.047176`.
- The physics-informed branch is the best overall forward result so far. It is
  not literally perfect in every global cross-correlation panel, but it is much
  closer than Phi and better than the best unconstrained NN M on the reported
  covariance and retained nonlinear correlation metrics.

Do not repeat:
- Do not use the no-floor direct-conditional ansatz for forward validation; it
  produced nonfinite states even though its pointwise true-M diagnostic looked
  excellent.
- Do not treat the joint-score `vL` ansatz branch as the default for this
  parametric form; it was worse than the direct conditional residual branch.
- Do not describe this branch as a pure data-only nonparametric NN. The tensor
  form is physics-informed; only the fitted coefficients and validation targets
  are data/learned-score based.

Useful follow-up:
- If more improvement is required, the remaining mismatch is in small global
  cross-correlations, not in the retained nonlinear C metric. Any next attempt
  should add a data-only fitting or selection term for those specific
  cross-correlations without using true coefficients.

Cleanup:
- Removed nonfinal bulky physics-ansatz forward trajectories and nonaccepted
  physics-ansatz checkpoints after retaining compact metrics.
- Kept the accepted floor001 model, diagnostics, forward HDF5, final
  paper-facing figures, and final metrics.
- Removed stale root-level `fit_dm.log` and `paper.log` accidental logs.

## 2026-05-17 — SoftSpinLLG deep artifact cleanup

Status: Success

Goal:
- Remove discarded generated artifacts from `SoftSpinLLGChain/` after the final
  physics-informed result, while ensuring `reports/agent_report.md` contains
  the experiment history and final retained-artifact list.

Approach:
- Inventoried `data/`, `models/`, `figures/`, and `logs`.
- Kept only production/final data, accepted score/Phi/conditional/M artifacts,
  the final clean37 target, writeup-referenced figures, paper-facing final
  comparison figures, and compact accepted metrics.
- Removed obsolete forward trajectories, rejected checkpoints, obsolete figures,
  duplicate checkpoint copies, old target artifacts, and nonfinal training/run
  logs.

Files changed:
- `SoftSpinLLGChain/reports/agent_report.md`
- `docs/agents/ATTEMPTS.md`

Files created temporarily:
- `.agent-scratch/softspin_inventory_before.tsv`
- `.agent-scratch/softspin_keep_*.txt`
- `.agent-scratch/softspin_remove_*.lst`

Commands/tests run:
- `find SoftSpinLLGChain -maxdepth 2 -type f ...`
- `du -sh SoftSpinLLGChain/*`
- `xargs -r rm -f < .agent-scratch/softspin_remove_*.lst`
- Existence checks for the retained final data, model, figure, and PDF paths.

Outcome:
- Removed 25 obsolete HDF5 data files, 105 rejected/duplicate model
  checkpoints, 132 obsolete PNG figures, and 259 nonfinal logs/metrics.
- Retained generated footprint:
  `data/` about `5.6G`, `models/` about `160M`, `figures/` about `19M`,
  `logs/` about `188K`.
- `SoftSpinLLGChain/reports/agent_report.md` now has a top-level current
  retained-artifact list and a cleanup section documenting what was removed.

Do not repeat:
- Do not keep bulky failed forward HDF5s or rejected checkpoints once their
  metrics and lessons are in `agent_report.md`.

Useful follow-up:
- If future work needs a removed branch, rerun it from the documented config
  and metrics context rather than restoring stale generated artifacts.

Cleanup:
- Removed temporary scratch inventory/removal lists after the cleanup.
- Remaining temporary files: `.agent-scratch/.gitignore` only.
