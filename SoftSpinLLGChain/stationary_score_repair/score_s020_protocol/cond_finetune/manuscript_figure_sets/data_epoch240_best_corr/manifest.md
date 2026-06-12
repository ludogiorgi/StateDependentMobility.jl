# Soft-Spin Manuscript Figure Set: Data-Only Epoch-240 Best-Correlation Branch

Generated in:

```text
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/
```

## Selected Objects

```text
stationary score:
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/models/score_protocol_gpu2_score_vr_big.bson

conditional source:
SoftSpinLLGChain/stationary_score_repair/models/cond_repaired_score_gpu2_main_epoch0720.bson

old conditional config:
SoftSpinLLGChain/stationary_score_repair/configs/cond_repaired_score_gpu2_main.toml

M config:
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/configs/fit_M_epoch240_transfer_warm_best_forward_gpu2.toml

M-NN model:
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/models/M_epoch240_transfer_warm_gpu1_best_snapshot_forward.bson

M-NN forward trajectory:
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/data/forward_M_epoch240_transfer_warm_best_gpu2.h5

matched Phi artifact:
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/models/fit_Phi_gpu2_vr_epoch240_seed0505_artifacts.bson

matched Phi forward trajectory:
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/data/forward_phi_epoch240_transfer_gpu1.h5

A target:
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/models/A_epoch240_noncheating_globalscale.bson
```

This branch uses the transferred learned residual

```text
r_transfer = r_old + s_old - s_epoch240
```

and is selected by forward correlation reconstruction, not by final A-loss.
The rejected final A-best forward file was not used:

```text
SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/data/forward_M_epoch240_transfer_warm_final_best_gpu2.h5
```

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

`spin_trueM_condscore_compare.png` was copied from the shared accepted figure.
SHA256:

```text
05f41ffa875a9236d625ee5e5f118d4cdd311b3b1bc24d3f8dafbc15b61ff2f5
```

## Commands

Coordinate correlations:

```bash
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_softspin_coordinate_summary.jl --force --phi=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/data/forward_phi_epoch240_transfer_gpu1.h5 --nn=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/data/forward_M_epoch240_transfer_warm_best_gpu2.h5 --phys=SoftSpinLLGChain/data/forward_dM_phys_ansatz_clean37_directcond_floor001_dt0015.h5 --out=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/spin_correlations.png --debug=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/logs/spin_correlations_debug.png --cache=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/logs/spin_coordinate_correlations.h5 --metrics=SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/logs/spin_coordinate_metrics.txt --heatmap-t=1
```

PDFs, Hovmoller, and mobility field:

```bash
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_pdf_nn_compare.jl SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/configs/render_pdf_nn_compare.toml
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_hovmoller_forward_compare.jl SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/configs/render_hovmoller_forward_compare.toml
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_mobility_field_compare.jl SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/configs/render_mobility_field_compare.toml
```

Residual/A figure:

```bash
xvfb-run -a julia --project=. --startup-file=no SoftSpinLLGChain/code/render_mobility11_A_fourway.jl SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/configs/fit_M_epoch240_transfer_warm_best_forward_gpu2.toml SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/models/M_epoch240_transfer_warm_gpu1_best_snapshot_forward.bson SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/models/A_epoch240_noncheating_globalscale.bson SoftSpinLLGChain/models/dM_phys_ansatz_clean37_directcond_mean1e5_floor001_gpu2.bson same SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/spin_residuals.png SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/logs/spin_residual_fourway.bson 60000 GPU:2 5070
```

Integrated constant-mobility structure diagnostic:

```bash
xvfb-run -a julia --project=. --threads 4 --startup-file=no SoftSpinLLGChain/code/render_phitilde_structure_diagnostic.jl SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/models/score_protocol_gpu2_score_vr_big.bson SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/spin_phitilde_structure.png SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/logs/spin_phitilde_structure.bson SoftSpinLLGChain/stationary_score_repair/score_s020_protocol/cond_finetune/manuscript_figure_sets/data_epoch240_best_corr/logs/spin_phitilde_structure_metrics.txt 'data-only epoch-240 score' SoftSpinLLGChain/data/soft_spin_llg_chain.h5 80000 GPU:2 5070
```

## Metrics

Native 11-family forward metric:

```text
epoch240 all mobility11 C rel.RMSE = 0.149058607
epoch240 all mobility11 C corr     = 0.996450356
epoch240 covariance rel.RMSE       = 0.133056566
```

New branch-local 0-to-100 coordinate cache:

```text
cache time range = 0.0 to 99.9735
target times     = 235
Phi full coordinate C rel.RMSE    = 0.415701983
M_NN full coordinate C rel.RMSE   = 0.179201228
M_phys full coordinate C rel.RMSE = 0.118080883
```

Residual/A metrics:

```text
pairs_per_lag = 60000
active_lag_indices = 7:24
true_m_rel_rmse = 0.0852274311
phys_rel_rmse   = 0.0900121237
nn_rel_rmse     = 0.0858225946
```

## No-Cheating Audit

The branch figures use saved observation and forward trajectories, the learned
epoch-240 stationary score, the frozen learned old residual transferred by
`r_old + s_old - s_epoch240`, the learned M-NN snapshot, and data-only Phi/A
artifacts.  True mobility/analytic quantities appear only in labeled ex-post
reference curves and were not used to train or select this branch.
