# Notebook-Style Post-Run Report (Wide Scout, Balance-Oriented)

- Source CSV: `RESEARCH/reports/astro_tabular_nn_quick_scout_wide_balance_e6_s4_20260208_055644.csv`
- Generated at: `2026-02-08T05:59:15`
- Ranking objective: maximize `test_recall_min`, then minimize `test_recall_gap`, then maximize `test_mcc`, then `test_acc`.

## Selected Latest Best Models

### Rank 1: `dcn_dropout_high` (`dcn`), seed `45`

- Test: recall_min=`0.6682`, recall_gap=`0.1096`, MCC=`0.1314`, ACC=`0.6704`
- Diagnostics plot: `model_rank01_dcn_dropout_high_seed45_diagnostics.png`
- Split metrics table: `model_rank01_dcn_dropout_high_seed45_split_metrics.csv`

### Rank 2: `dcn_cross_heavy` (`dcn`), seed `44`

- Test: recall_min=`0.6667`, recall_gap=`0.2272`, MCC=`0.2434`, ACC=`0.8894`
- Diagnostics plot: `model_rank02_dcn_cross_heavy_seed44_diagnostics.png`
- Split metrics table: `model_rank02_dcn_cross_heavy_seed44_split_metrics.csv`

### Rank 3: `deepfm_embed128` (`deepfm`), seed `45`

- Test: recall_min=`0.5892`, recall_gap=`0.1886`, MCC=`0.1039`, ACC=`0.5929`
- Diagnostics plot: `model_rank03_deepfm_embed128_seed45_diagnostics.png`
- Split metrics table: `model_rank03_deepfm_embed128_seed45_split_metrics.csv`

### Rank 4: `deepfm_narrow` (`deepfm`), seed `43`

- Test: recall_min=`0.5556`, recall_gap=`0.3970`, MCC=`0.3049`, ACC=`0.9447`
- Diagnostics plot: `model_rank04_deepfm_narrow_seed43_diagnostics.png`
- Split metrics table: `model_rank04_deepfm_narrow_seed43_split_metrics.csv`

### Rank 5: `deepfm_narrow` (`deepfm`), seed `45`

- Test: recall_min=`0.4444`, recall_gap=`0.2734`, MCC=`0.0502`, ACC=`0.7124`
- Diagnostics plot: `model_rank05_deepfm_narrow_seed45_diagnostics.png`
- Split metrics table: `model_rank05_deepfm_narrow_seed45_split_metrics.csv`

## Leaderboard (Top 15)

- Saved table: `leaderboard_top15.csv`

```text
           model model_type  seed  best_margin  best_val_score  test_recall_down  test_recall_up  test_recall_min  test_recall_gap  test_mcc  test_acc
dcn_dropout_high        dcn    45         0.59          0.4046            0.6682          0.7778           0.6682           0.1096    0.1314    0.6704
 dcn_cross_heavy        dcn    44         0.48          0.4699            0.8939          0.6667           0.6667           0.2272    0.2434    0.8894
 deepfm_embed128     deepfm    45         0.90          0.0306            0.5892          0.7778           0.5892           0.1886    0.1039    0.5929
   deepfm_narrow     deepfm    43         0.80          0.4627            0.9526          0.5556           0.5556           0.3970    0.3049    0.9447
   deepfm_narrow     deepfm    45         0.54          0.3669            0.7178          0.4444           0.4444           0.2734    0.0502    0.7124
dcn_high_dropout        dcn    45         0.61          0.4590            0.7381          0.4444           0.4444           0.2937    0.0578    0.7323
  deepfm_embed96     deepfm    43         0.69          0.4596            0.7743          0.4444           0.4444           0.3298    0.0726    0.7677
 dcn_cross_heavy        dcn    43         0.85          0.3219            0.4628          0.3333           0.3333           0.1294   -0.0571    0.4602
  deepfm_embed64     deepfm    43         0.62          0.5099            0.5305          0.3333           0.3333           0.1971   -0.0381    0.5265
     deepfm_base     deepfm    43         0.38          0.5576            0.7111          0.3333           0.3333           0.3777    0.0137    0.7035
 dcn_cross_light        dcn    44         0.86          0.3775            0.7562          0.3333           0.3333           0.4229    0.0291    0.7478
 dcn_dropout_low        dcn    45         0.68          0.4336            0.7607          0.3333           0.3333           0.4274    0.0307    0.7522
 dcn_cross_light        dcn    42         0.73          0.4358            0.8194          0.3333           0.3333           0.4861    0.0551    0.8097
deepfm_wider_mlp     deepfm    43         0.50          0.4597            0.8442          0.3333           0.3333           0.5109    0.0678    0.8341
 dcn_cross_light        dcn    45         0.89          0.4066            0.8600          0.3333           0.3333           0.5267    0.0770    0.8496
```