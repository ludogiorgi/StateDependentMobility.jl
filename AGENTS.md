# Protocol for Adding a New Benchmark System

These instructions are for agents implementing the full data-only stochastic
mobility pipeline for a new system described in a user-provided file such as
`system.txt`. Follow them exactly. The work is intentionally step-gated: finish
only the current requested step, verify it carefully, and then stop for user
feedback before moving to the next step.

## Global Rules

- Do not use analytic score, analytic mobility, simulator coefficients, generator
  formulas, or true model internals inside any minimized training loss or
  data-driven training target. In particular, analytic information must not enter
  DSM score labels, conditional-score losses, Phi estimation, `Cdot_data`,
  `A_data`, residual targets, loss weights derived from targets, `M_theta`
  training tensors, or model/hyperparameter selection by target error.
- Analytic or simulator information may be used to design and select meaningful
  observables, to understand the system, and to produce clearly labeled ex-post
  validation diagnostics. The numerical quantities used for training those
  observables must still be computed from trajectory data and learned score
  models only.
- Before saving final artifacts or reporting that a step is complete, explicitly
  audit the implementation and make very sure there has been no cheating: no
  analytic-model information may have entered any minimized loss, data-driven
  estimator, training target, residual target, or target-derived weight. It is
  acceptable to document analytic guidance used for observable selection and to
  include labeled analytic validation plots.
- Read `paper.tex`, the system description, and the best existing examples before
  coding. Use the paper conventions:
  `dx = [M(x)s(x) + div_x M(x)]dt + sqrt(2) Sigma(x)dW`,
  `Sigma Sigma' = sym(M)`, and `s = grad log p_ss`.
- Use the GPU specified by the user. If a GPU name/index is specified, add a
  startup check and abort if the selected device does not match. Do not default
  silently to GPU 0.
- Optimize for long runs: parallelize independent trajectories on CPU threads,
  batch score evaluations on GPU, cache expensive pair/score tensors, and avoid
  rerunning simulations when saved data already exists.
- Do not shortcut the requested work: do precisely what the user asked, run the
  required scripts and acceptance checks to completion, and take all the time
  needed rather than substituting smoke tests or partial verification.
- When the user has instructed the agent to keep working until a quantitative
  acceptance target is met, do not stop merely because the user asks for status
  or because an intermediate attempt failed. Give the status briefly, then
  continue executing the next concrete experiment unless the user explicitly
  says to stop, pause, or only reply.
- Figures must be publication-quality GLMakie figures: readable labels, equal
  panel widths, consistent line styles, legends that do not cover data, and no
  visibly undersampled trajectory traces. Inspect figures before finishing.
- Use this folder layout for every new system:

```text
SystemShortName/
  code/       # sim.jl, score.jl, fit_Phi.jl, cond_score.jl, fit_dM.jl
  configs/    # matching TOML files
  data/       # HDF5 datasets and forward-validation trajectories
  models/     # BSON/JLD2/Flux checkpoints
  figures/    # PNG/PDF figures
  logs/       # training logs and metric text files
  reports/    # agent_report.md: one living agent-friendly report
  writeup/    # system_report.tex: one living human-facing LaTeX report
```

If an existing benchmark still uses a flat layout, do not copy that layout for
new systems.

## Persistent Reports, Writeup, and Cleanup

For each benchmark system, maintain exactly one agent-friendly Markdown report
at `SystemShortName/reports/agent_report.md`. Create it at the end of Step 1
and update the same file after every later completed step. Do not create
separate per-step reports unless the user explicitly asks for them. A later
agent must be able to read this single report and understand precisely what was
tried, what worked, what failed, why decisions were made, and what should not be
repeated on similar systems.

The report must be practical and implementation-oriented, not a polished paper
summary. At minimum, after each completed step update it with:

- The exact scripts/configs run, key command lines, GPU/device actually used,
  wall-clock scale when useful, and final artifact paths.
- The final accepted numerical checks and acceptance metrics.
- Every important failed attempt or diagnostic branch: what was changed, what
  output/metric/figure showed it was wrong, and the concrete lesson for a future
  agent.
- Any estimator, architecture, observable, projection, lag-window, sampling, or
  validation choice that was non-obvious, including whether analytic information
  was used only for ex-post diagnostics.
- A clear no-cheating audit for the step.
- The current next action, blockers, and warnings for Step 2 or Step 3 agents.

For each benchmark system, also maintain exactly one human-facing LaTeX writeup
at `SystemShortName/writeup/system_report.tex`. Create it at the end of Step 1
and update the same file immediately after every completed experiment,
diagnostic branch, failed attempt that affects later decisions, accepted result,
or figure generation. Compile the LaTeX file after every update and fix
compilation errors before reporting progress or stopping. Keep the compiled PDF
beside `system_report.tex` unless the user asks otherwise.

The LaTeX writeup is for the human reader and should be professional,
mathematically precise, clear, and coherent enough to be copied into a scientific
paper after final editing. It must let the reader understand the system and the
complete experimental history without opening any code. Do not refer to source
files, scripts, config files, function names, line numbers, command lines, or
implementation details that require reading code. Describe what was implemented
in mathematical, algorithmic, statistical, and experimental terms instead.

At minimum, `writeup/system_report.tex` must contain and keep current:

- A precise definition of the considered stochastic system: state variables,
  domain, constraints, invariants, boundary conditions, parameter values,
  state ordering when relevant, stationary distribution when known, mobility
  structure when known, and the exact SDE using the paper conventions.
- A clear description of the numerical experiment design: integration scheme at
  the level of numerical method, time step, burn-in, saved resolution, sampling
  budget, decorrelation-time estimate, validation observables, and figure
  interpretation. This description must not require consulting code.
- A chronological, human-readable account of all important experiments tried by
  the agent for that system, including failed attempts, what changed
  scientifically or statistically, what results showed, and what conclusion was
  drawn.
- Precise mathematical descriptions of all estimators and learned objects used
  in the current step: DSM score learning, tangent-space noise projections,
  Phi estimation, GFDT correlation estimators, conditional-score diagnostics,
  mobility residual targets, forward validation, and any projections or symmetry
  reductions.
- Tables or compact numerical summaries of accepted metrics and important
  rejected metrics, with enough context to understand why a result was accepted
  or rejected.
- References to generated figures by figure number and scientific content, with
  captions explaining what is shown and what conclusion should be drawn. Include
  final accepted figures and any compact diagnostic figures needed to understand
  important failures.
- A clearly labeled no-cheating statement for each step: analytic information
  used only for system understanding, observable design, or ex-post diagnostics
  must be distinguished from data-only training targets and minimized losses.

The Markdown report and the LaTeX writeup have different audiences. The
Markdown report is an agent memory snapshot: it may mention exact files,
commands, devices, configs, cleanup decisions, and restart warnings. The LaTeX
writeup is a human scientific report: it must avoid code references and present
the same work as a coherent mathematical and experimental narrative.

After a step is completed, verified, the Markdown report is updated, and the
LaTeX writeup is updated and compiled, clean the system folder of useless
outputs from failed experiments. Remove failed-checkpoint copies, obsolete data
files, obsolete figures, scratch outputs, and misleading intermediate artifacts
that are not needed to reproduce or understand the final accepted result. Keep
only the final accepted artifacts plus any compact logs or diagnostic files that
are explicitly referenced by the Markdown report and useful for future debugging,
and any figures or compiled writeup outputs referenced by
`writeup/system_report.tex`.
Never delete the accepted production dataset, accepted model checkpoints,
accepted figures, accepted metrics, the single Markdown report, or the single
LaTeX writeup.

## Step 1: Simulation Only

Create only `code/sim.jl` and `configs/sim.toml`, plus generated data/figures
and the required `reports/` and `writeup/` documents. Then stop.

Required behavior:

- Implement the SDE exactly as described in the system file. Preserve invariants,
  constraints, periodicity, positivity, and conservation laws by construction;
  do not clip physical variables unless the system description explicitly allows it.
- Estimate the decorrelation time `t_D` from a pilot run using relevant
  observables and ACF envelopes. Confirm the chosen ACF window does not cap the
  estimate.
- Match the previous accepted benchmark sampling budget:
  `N_uncorr = T / t_D` equal to the reference dataset, and
  `t_D / save_dt = 100`. If no reference is specified, use the latest accepted
  benchmark dataset; historical default is about `2777.78` decorrelation times.
- Save all metadata needed by later steps: state ordering, channel names, `dt`,
  `save_dt`, `t_D`, `T`, number of saved snapshots per `t_D`, burn-in, seed,
  parameters, conservation diagnostics, and any constraints/projections.
- Save HDF5 data with at least `/trajectories/time`, `/trajectories/states`, and
  `/trajectories/states_flat`. Document flat ordering explicitly.

Required figures:

- Summary figure: PDFs, covariance/correlation structure, ACFs, cross-correlations,
  conservation/constraint diagnostics, and sampling-budget text.
- Dynamics figure: high-resolution time-space/Hovmoller views or equivalent
  system-specific visualization. Use the full saved resolution, not a heavily
  downsampled trace.
- Trajectory figure: representative site/component traces, phase portraits or
  reduced projections, and invariant/constraint traces.

Acceptance checks before stopping:

- `t_D / save_dt` is approximately `100`.
- `T / t_D` matches the reference uncorrelated-sample count.
- No hidden instability, positivity failure, conservation drift, or figure-layout
  issue remains.
- The saved dataset resolution is explicitly reported.
- Create or update the single system report at `reports/agent_report.md` with
  the successful simulation setup, failed attempts, final checks, no-cheating
  audit, and lessons for future agents.
- Create or update `writeup/system_report.tex` with a precise system definition,
  all parameter values, the simulation design, generated figures, accepted
  checks, failed simulation attempts, and a no-cheating statement; compile it
  successfully.
- Remove useless failed simulation outputs.

## Step 2: Stationary Score and Phi Baseline

This step has two scripts: `code/score.jl` and `code/fit_Phi.jl`, each with a
matching TOML file. Finish both, verify them, then stop.

### Stationary Score

Train a stationary score U-Net using DSM only.

- Use `sigma = 0.05` unless the user explicitly asks otherwise.
- Add DSM noise only in the physically valid tangent subspace. For conserved
  zero modes, subtract noise means per conserved channel and project network
  outputs to the same subspace.
- Use a symmetry-aware architecture. For lattice systems, use a periodic 1D
  U-Net with one channel per physical component. Use no BatchNorm; use no
  normalization or a sample-independent option such as GroupNorm/LayerNorm.
- Use analytic stationary score only for ex-post diagnostics, never for training
  or model selection.
- Save the score checkpoint, normalization/statistics, training history, and
  diagnostics.

Mandatory score validation:

- Plot DSM train/validation loss.
- If analytic score is available, report cosine similarity and relative RMSE on
  both full and safe-subset samples.
- Integrate the score-only Langevin equation and compare its stationary
  statistics to observations. In normalized constrained coordinates use
  `dz = s_theta(z)dt + sqrt(2)P dW`, where `P` projects out conserved modes.
- The score is not acceptable until score-only Langevin reproduces observed PDFs,
  covariance/spectra, and conserved modes. A good analytic-score plot alone is
  not enough.

### Phi and Constant-Mobility Baseline

Create `fit_Phi.jl` after the score is validated.

- Use a meaningful observable library `phi_m(x)` with good signal-to-noise for
  later mobility learning. Always use `phi_n(x)=x`.
- Estimate `Cdot_mn(t)` from data over `t in [0,t_D]`, using robust finite
  differences or smooth interpolation of empirical correlations. Extrapolate
  `Cdot_xx(0+)` to estimate `Phi = -Cdot_xx(0+)`.
- If symmetries are known, estimate the full data-only Phi first, then project
  to the configured symmetry class. Save the raw and projected versions and
  report off-profile norm, projection change, and tangent-space eigenvalues.
- If true `M(x)` is available, compare `Phi` with `<M_true>` only after `Phi` is
  constructed. Work on the estimator until this diagnostic is good, or clearly
  isolate why it cannot be.

Important formula learned from the FHDChain failure:

- For `M=Phi`, do not use the conditional score to estimate the Phi Cdot curve.
  Use the stationary-score GFDT channel:

```text
B_{Phi,n,j}(x) =
    s(x)' Phi grad(phi_{n,j})(x)
    + tr(Phi Hessian(phi_{n,j})(x))

Cdot_mn^Phi(t) =
    < phi_m(x_t) B_{Phi,n}(x_0) >
```

- For the common choice `phi_n(x)=x`, this reduces to
  `Cdot_mx^Phi(t) = <phi_m(x_t) s(x_0)'> Phi`.
- This Phi+plain-score diagnostic is often much more accurate than a
  conditional-score Phi diagnostic and should be the default baseline check.

Mandatory outputs:

- Save `Cdot_data`, `Phi_raw`, `Phi_projected`, score-based Phi Cdot curves,
  observable definitions, and a `M=Phi` forward Langevin trajectory.
- Produce comprehensive figures for score validation, Phi recovery, data vs
  Phi-GFDT `Cdot_mn(t)`, and `M=Phi` forward statistics/correlations.

Acceptance checks before stopping:

- Stationary score passes score-only Langevin validation.
- `Phi` is stable under short-lag fit-window changes and PSD/tangent checks.
- If true mobility is available, `Phi` agrees with `<M_true>` to the expected
  tolerance; target relative error is below `0.10`, preferred below `0.05`.
- The Phi forward model preserves the observed invariant density and reproduces
  the coordinate/selected baseline correlations as well as expected.
- Update the single system report with the accepted score/Phi implementation,
  all important failed score/Phi attempts, final diagnostics, no-cheating audit,
  and guidance for Step 3.
- Update and compile `writeup/system_report.tex` with the score-learning method,
  Phi estimator, baseline validation, figures, accepted and rejected
  diagnostics, and a no-cheating statement, all written without code references.
- Remove useless failed score/Phi data, figures, checkpoints, and scratch
  outputs.

## Step 3: Conditional Score, Mobility NN, and Full Validation

Start this step only after explicit user approval.

### Conditional Score

Train a direct reverse/posterior conditional score with DSM.

- Prefer a direct model for `q_tau(x0,xt)=grad_x0 log p(x0 | xt,tau)` over the old
  joint-score route. The transition score required by the paper identity is
  `r_tau = q_tau - s_ss(x0)`. Never use the posterior score directly as the
  transition score.
- Best-performing architecture from prior attempts: stationary-score-plus-residual
  parameterization,
  `q_tau(x0,xt) = s_theta(x0) + residual_theta(x0,xt,tau)`.
  The operational transition score is the learned residual.
- Add DSM noise to `x0` in the valid tangent subspace. Do not unnecessarily
  smooth the conditioning variable `xt`; smoothing `xt` biased previous joint
  score diagnostics.
- Use Fourier time features or FiLM conditioning for lag. A single scalar time
  channel is not sufficient. Oversample the lag range used by mobility training.
- Use no BatchNorm.

Mandatory conditional-score diagnostics:

- Lagwise `||E[r_tau]||/sqrt(D)` and `||E[r_tau x0^T]||/||I||`.
- Posterior reconstruction diagnostics for `s_theta + residual`.
- Conditional-score operator test with true `M` if available:
  compare `Cdot_data` with `-<phi_m(x_t)(M_true(x0)' r_tau)_n>`.
- Do not move to mobility training if the conditional-score operator diagnostic
  is poor, unless the user explicitly tells you to proceed.

### Mobility NN

Train `M_theta` using the paper residual loss involving `Phi`.

- The training target is
  `A_data = Cdot_data - A[Phi]`, where `A[Phi]` must come from the stationary-score
  GFDT formula above, not from the conditional score.
- For a mobility correction,
  `A[delta M_theta] = -<phi_m(x_t) (delta M_theta(x0)' r_tau)_n>`.
- Enforce `<M_theta> ~= Phi` with the mean-mobility penalty from the paper.
- Parameterize `M_theta` to respect known structure: PSD symmetric part,
  skew/antisymmetric part, conservation laws, locality, equivariance, and
  support constraints. Do not add unconstrained local residual blocks that break
  PSD or physical structure.
- Choose observables by data-only signal-to-noise, smoothness, and Phi-baseline
  residual signal. Exclude noisy channels even if their raw magnitude is large.

Mandatory mobility figures:

- Training diagnostics: loss vs epoch, validation A rel.RMSE/correlation, mean
  penalty, PSD/constraint diagnostics, and representative A target panels.
- Final A comparison: `A_data`, ex-post `A_trueM` if available, and
  `A[M_theta]` on all selected channels.

Do not claim success if the NN only learns Phi. In previous failures, a mobility
NN was useless when `A_train` vs `A_data` had rel.RMSE near `1` and low
correlation, even though `<M_NN>` looked close to `<M_true>` because Phi was
already correct.

### Forward Validation

After the A target is fit well, integrate:

- Constant baseline:
  `dx = Phi s_theta(x)dt + sqrt(2) Sigma_Phi dW`,
  `Sigma_Phi Sigma_Phi' = sym(Phi)` on the tangent subspace.
- Learned model:
  `dx = [M_theta(x)s_theta(x) + div_x M_theta(x)]dt + sqrt(2)Sigma_theta(x)dW`,
  `Sigma_theta Sigma_theta' = sym(M_theta)`.

Use Cholesky/eigendecomposition/factor construction appropriate to the PSD
operator; do not write an ambiguous scalar `sqrt(sym(Phi))`. Compute
`div_x M_theta` by exact AD or validated finite differences.

Required final figures:

- Forward statistics: observations vs `M=Phi` vs `M_theta` for PDFs, covariance,
  spectra, ACFs, cross-correlations, and invariants.
- Forward correlations: observed `C_mn(t)` vs Phi Langevin vs learned-M Langevin.

Acceptance checks before stopping:

- `A[M_theta]` reproduces the selected data targets with high correlation and
  low normalized error.
- Forward `M_theta` improves finite-lag dynamical statistics over `M=Phi`
  without degrading stationary PDFs/covariance.
- If `M=Phi` already matches the selected constraints as well as true `M`, report
  that the system may be a poor benchmark for state-dependent mobility recovery
  and do not hide the failure.
- Update the single system report with the conditional-score, mobility, and
  forward-validation outcomes; include failed attempts and lessons for future
  agents.
- Update and compile `writeup/system_report.tex` with the conditional-score
  method, mobility-learning target, final forward validation, all important
  rejected branches, generated figures, accepted metrics, and a no-cheating
  statement, all written without code references.
- Remove useless failed Step 3 outputs while preserving the final accepted
  artifacts, Markdown report, LaTeX writeup, and compiled writeup PDF.

## Known Pitfalls to Avoid

- Do not train with analytic scores, true mobility, generator formulas, or any
  analytic-model tensor in the minimized loss. That is cheating even when the
  same information is useful for observable design or ex-post diagnostics.
- Do not evaluate the Phi-only Cdot baseline with a conditional score. Use the
  stationary-score GFDT formula.
- Do not trust good-looking generated PDFs from a joint score as evidence that
  the transition-score projections are accurate.
- Do not subtract the stationary score from a residual conditional model whose
  output is already the transition-score residual.
- Do subtract the stationary score from a posterior model that outputs
  `grad_x0 log p(x0 | xt,tau)`.
- Do not let BatchNorm mix samples in score networks.
- Do not use a low-resolution downsample for trajectory panels while claiming to
  visualize the saved dataset.
- Do not make leftmost panels wider/narrower than the others unless the layout
  intentionally uses a spanning panel and this is visually clear.
- Do not proceed to mobility NN interpretation when Phi or the conditional-score
  operator diagnostic is failing.
- Do not call a mobility NN successful because `<M_NN>` matches `<M_true>` if the
  residual A target is not fitted; that usually means the model learned only the
  constant baseline.

---

# Repository Workflow for ChatGPT Agents

This repository may be edited by multiple ChatGPT agents over time. The goal is
to avoid repeating failed attempts, keep the repository clean, and maintain a
compact memory of what has already been tried.

Before starting any task, read this file and then read:

```text
docs/agents/LESSONS.md
docs/agents/ATTEMPTS.md
```

If these files do not exist, create them.

## Required Files

The repository should contain:

```text
docs/agents/LESSONS.md
docs/agents/ATTEMPTS.md
.agent-scratch/
```

Use:

```text
docs/agents/LESSONS.md
```

for compact, reusable lessons that future agents should know.

Use:

```text
docs/agents/ATTEMPTS.md
```

for the chronological record of attempted solutions, including failed ones.

Use:

```text
.agent-scratch/
```

for temporary files, experiments, logs, intermediate outputs, debugging scripts,
copied data, and anything that should not remain part of the repository.

Do not create temporary files in the repository root.

## Before Making Changes

Before editing code, do the following:

1. Read `docs/agents/LESSONS.md`.
2. Read the most recent entries in `docs/agents/ATTEMPTS.md`.
3. Identify whether the current task resembles a previous failed attempt.
4. If reusing or revisiting a previously failed approach, explicitly explain
   what is different this time.

Do not repeat an approach listed as failed unless there is a clear reason why
the previous failure no longer applies.

## During an Attempt

Keep the repository organized.

Temporary files must go in:

```text
.agent-scratch/
```

Examples of temporary files include:

```text
.agent-scratch/debug_script.py
.agent-scratch/test_output.txt
.agent-scratch/failed_plot.png
.agent-scratch/tmp_data/
.agent-scratch/experiment_notes.md
```

Do not leave files such as these scattered in the repository:

```text
debug.py
tmp.py
test123.py
notes.txt
output.png
junk.csv
old_version.py
copy_of_file.py
```

If a temporary file becomes useful, move it to an appropriate tracked location
and give it a meaningful name.

## After Every Attempt

After every attempt, successful or failed, update:

```text
docs/agents/ATTEMPTS.md
```

Use this format:

```markdown
## YYYY-MM-DD — Short task title

Status: Success / Failed / Partial

Goal:
- Briefly state what the agent tried to accomplish.

Approach:
- Summarize the attempted solution.
- Mention the main files or modules involved.

Files changed:
- `path/to/file1`
- `path/to/file2`

Files created temporarily:
- `.agent-scratch/example_file`
- `.agent-scratch/example_directory/`

Commands/tests run:
- `command here`

Outcome:
- State what happened.
- If successful, explain how success was verified.
- If failed, explain the precise failure mode.

Do not repeat:
- For failed or partial attempts, list the specific mistake, dead end, or
  fragile assumption that future agents should avoid.

Useful follow-up:
- Mention any promising direction that was not completed.
```

Keep entries compact. The goal is to help the next agent quickly understand what
was tried, not to write a long diary.

## Updating Reusable Lessons

If an attempt reveals a lesson that is likely to matter again, also update:

```text
docs/agents/LESSONS.md
```

Use this format:

```markdown
## Topic or subsystem name

- Compact lesson learned.
- What should future agents avoid?
- What approach appears more promising?
- Mention relevant files if useful.
```

Only put durable information in `LESSONS.md`.

Good examples:

```markdown
## Residual NODE training

- Do not blindly whiten residuals with `Sigma^{-1}`. Previous attempts showed
  that this amplified noisy directions and destabilized training.
- Start with the unwhitened residual first, then test regularized whitening only
  after checking the spectrum of `Sigma`.
```

Bad examples:

```markdown
## Random note

- I tried something and it did not work.
```

## Cleanup After Every Attempt

Before finishing, clean the repository.

Delete temporary files that are no longer needed, especially files in:

```text
.agent-scratch/
```

Temporary files may remain in `.agent-scratch/` only if they are useful for the
next agent and are explicitly listed in `docs/agents/ATTEMPTS.md`.

Remove accidental files from the repository root, including:

```text
debug scripts
temporary notebooks
temporary plots
temporary logs
copied source files
backup files
partial outputs
```

Do not delete source code, data, configuration files, or documentation unless
the task explicitly requires it.

After cleanup, report the final state:

```markdown
Cleanup:
- Removed temporary files: yes/no
- Remaining temporary files: list, with reason
- Repository root checked for accidental files: yes/no
```

## Required Final Response From an Agent

At the end of each task, the agent must report:

```markdown
Summary:
- What changed.

Verification:
- What tests or checks were run.

Memory updated:
- `docs/agents/ATTEMPTS.md`: yes/no
- `docs/agents/LESSONS.md`: yes/no, with reason

Cleanup:
- Temporary files removed: yes/no
- Remaining temporary files: list or none

Caveats:
- Anything unresolved or uncertain.
```

Do not claim that the task is complete unless the attempt log has been updated
and cleanup has been performed.

## Important Rule

A failed attempt is not finished until:

1. The repository has been cleaned.
2. `docs/agents/ATTEMPTS.md` has been updated.
3. `docs/agents/LESSONS.md` has been updated if the failure produced a reusable
   lesson.

Future agents rely on this memory. Keep it accurate, compact, and useful.
