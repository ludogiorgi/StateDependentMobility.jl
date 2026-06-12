# Soft-Spin Manuscript Figure Set: Physics-Informed 11-Observable Branch

Generated in:

```text
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/
```

## Selected Objects

```text
stationary score:
SoftSpinLLGChain/models/score_phys_pC_sigma02.bson

conditional score:
SoftSpinLLGChain/models/cond_score_phys_pC_unet_vA_cont3.bson

M-NN model:
SoftSpinLLGChain/models/mobility11/analytic/M_best.bson

M-NN forward trajectory:
SoftSpinLLGChain/data/mobility11/analytic/forward_best.h5

matched Phi forward trajectory:
SoftSpinLLGChain/data/mobility11/analytic/forward_phi.h5

M training / residual-render config:
SoftSpinLLGChain/configs/mobility11_analytic_fullcache_nosignal_gpu0.toml

A target:
SoftSpinLLGChain/models/mobility11/analytic/A_target.bson
```

This is the requested 11-observable physics-informed branch.  It is not the old
clean37 physics-informed ansatz, which is used only as a reference curve.

## Generated Figures

```text
spin_correlations.png
spin_residuals.png
spin_pdfs.png
spin_hovmoller.png
spin_mobility.png
spin_phitilde_structure.png
spin_trueM_condscore_compare.png
```

`spin_trueM_condscore_compare.png` was copied from:

```text
SoftSpinLLGChain/figures/manuscript_softspin_final/spin_trueM_condscore_compare.png
```

SHA256:

```text
05f41ffa875a9236d625ee5e5f118d4cdd311b3b1bc24d3f8dafbc15b61ff2f5
```

## Commands

Coordinate correlations:

```bash
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_softspin_coordinate_summary.jl --force --phi=SoftSpinLLGChain/data/mobility11/analytic/forward_phi.h5 --nn=SoftSpinLLGChain/data/mobility11/analytic/forward_best.h5 --phys=SoftSpinLLGChain/data/forward_dM_phys_ansatz_clean37_directcond_floor001_dt0015.h5 --out=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/spin_correlations.png --debug=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/logs/spin_correlations_debug.png --cache=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/logs/spin_coordinate_correlations.h5 --metrics=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/logs/spin_coordinate_metrics.txt --heatmap-t=1
```

PDFs, Hovmoller, and mobility field:

```bash
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_pdf_nn_compare.jl SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/configs/render_pdf_nn_compare.toml
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_hovmoller_forward_compare.jl SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/configs/render_hovmoller_forward_compare.toml
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_mobility_field_compare.jl SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/configs/render_mobility_field_compare.toml
```

Residual/A figure:

```bash
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_mobility11_A_fourway.jl SoftSpinLLGChain/configs/mobility11_analytic_fullcache_nosignal_gpu0.toml SoftSpinLLGChain/models/mobility11/analytic/M_best.bson SoftSpinLLGChain/models/mobility11/analytic/A_target.bson SoftSpinLLGChain/models/dM_phys_ansatz_clean37_directcond_mean1e5_floor001_gpu2.bson SoftSpinLLGChain/configs/cond_score_dataonly_seed602_scaled_gpu0.toml SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/spin_residuals.png SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/logs/spin_residual_fourway.bson 60000 GPU:2 5070
```

Integrated constant-mobility structure diagnostic:

```bash
xvfb-run -a julia --project=. --threads 4 --startup-file=no SoftSpinLLGChain/code/render_phitilde_structure_diagnostic.jl SoftSpinLLGChain/models/score_phys_pC_sigma02.bson SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/spin_phitilde_structure.png SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/logs/spin_phitilde_structure.bson SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/phys_score_11obs_cond/logs/spin_phitilde_structure_metrics.txt 'physics-informed score' SoftSpinLLGChain/data/soft_spin_llg_chain.h5 80000 GPU:2 5070
```

## Metrics

Native selected-observable metric from the accepted 11-family validation:

```text
M_best all mobility11 C rel.RMSE = 0.0531850004
M_best all mobility11 C corr     = 0.998877911
```

Old accepted full-coordinate manuscript cache:

```text
M_NN full coordinate C rel.RMSE = 0.0731980025
M_NN full coordinate C corr     = 0.997430674
```

New branch-local 0-to-100 coordinate cache:

```text
cache time range = 0.0 to 99.9735
target times     = 235
Phi full coordinate C rel.RMSE    = 0.421224513
M_NN full coordinate C rel.RMSE   = 0.131772981
M_phys full coordinate C rel.RMSE = 0.118080883
```

The new full-coordinate value is larger than the old accepted value because the
updated figure cache scores all finite long-grid target times out to 100.  The
old `0.0731980025` value is still reproduced from the old accepted cache.

Residual/A metrics:

```text
pairs_per_lag = 60000
active_lag_indices = 7:24
true_m_rel_rmse = 0.111476853
phys_rel_rmse   = 0.111697072
nn_rel_rmse     = 0.117826075
```

## No-Cheating Audit

The generated figures use saved observation and forward trajectories, learned
stationary/conditional scores, learned M-NN checkpoints, and data-driven Phi/A
targets.  True mobility/analytic quantities appear only in labeled ex-post
reference curves and were not used to train or select this figure set.
