# Models — README

Machine learning experiments for crypto asset ranking.
All notebooks predict which coins will outperform the cohort over the next 7–30 days.

For access to dataset pls text me on telegram @pal1ychuk

---

## Structure

```
Models/
├── Data/                        — model-ready datasets (gitignored, not in repo)
│   └── data_for_experiments.csv
│
├── Modeling/                    — baseline models
│   ├── first_model_weekly.ipynb         — first end-to-end weekly model
│   ├── different_model_comparison.ipynb — RF / XGBoost / LightGBM / GBM comparison
│   └── Other/
│       ├── model_comparison_results.csv — full metrics table
│       └── model_comparison_top30.txt   — top-30 features by importance
│
├── Experiments/                 — feature and strategy experiments
│   ├── 1_feature_experiments.ipynb      — 4 experiments testing data hypotheses
│   ├── 1.1_hybrid_strategy.ipynb        — hybrid signal strategy
│   ├── 2_removal_stablecoins.ipynb      — effect of removing stablecoins
│   ├── 3_finding_improvements.ipynb     — iterative model improvements
│   └── 4_hybrid_strategies_v2.ipynb     — refined hybrid strategies
│
├── Shap analysis/
│   └── main_shap_analysis.ipynb         — SHAP feature importance and interpretation
│
└── Figures/                     — all experiment output plots (76 PNGs)
```

---

## Notebooks

### Modeling

| Notebook | Description |
|----------|-------------|
| `first_model_weekly.ipynb` | Baseline weekly model: loads `data_for_experiments.csv`, trains RF/XGBoost/LightGBM/GBM, evaluates bucket returns and cumulative P&L |
| `different_model_comparison.ipynb` | Side-by-side comparison of all four model types across metrics |

### Experiments

| Notebook | Experiment | Question |
|----------|-----------|----------|
| `1_feature_experiments.ipynb` | 4 feature set experiments | What signal types drive performance? |
| `1.1_hybrid_strategy.ipynb` | Hybrid signal strategy v1 | Can combining signals beat individual ones? |
| `2_removal_stablecoins.ipynb` | Stablecoin removal | Does including stablecoins hurt the model? |
| `3_finding_improvements.ipynb` | Iterative improvements | Where are the biggest gains? |
| `4_hybrid_strategies_v2.ipynb` | Hybrid strategies v2 | Refined approach after v1 findings |

### SHAP Analysis

`main_shap_analysis.ipynb` — interprets the best model using SHAP values.
Shows which features drive Strong Buy / Avoid predictions globally and per-coin.

---

## Setup

```bash
pip install scikit-learn xgboost lightgbm shap pandas numpy matplotlib

# Data file is not in the repo — obtain data_for_experiments.csv
# from the pipeline in ../Building dataset/ and place it in Data/
```

---

## Figures

All plots are saved to `Figures/` and organised by experiment:

| Prefix | Source |
|--------|--------|
| `exp0_full_*` | Baseline full feature set |
| `exp1_technical_*` | Technical indicators only |
| `exp2_decorr_*` | Decorrelated feature set |
| `exp3_sentiment_*` | Sentiment + derivatives only |
| `exp4_*` | Regime-conditioned (bull/bear/sideways) |
| `hybrid_*` | Hybrid strategy experiments |
| `shap_*` | SHAP analysis plots |
| `final_*` | Summary comparisons |
