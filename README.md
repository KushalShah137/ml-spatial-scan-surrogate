# Spatial Scan Surrogate

This repository builds and evaluates a structured surrogate for spatial scan statistics. The surrogate predicts each dataset's Gumbel location and scale for the null-maximum distribution of the spatial scan statistic from dataset-level features, then computes window p-values by formula. This replaces repeated null simulation for each window with a fitted dataset-level distribution while retaining per-window p-value estimates.

## Repository layout

- `config/`: dataset and scan settings.
- `data/raw/`: local CDC WONDER exports; not committed.
- `data/processed/`: prepared county data, scan windows, and saved null-maximum distributions.
- `data/synth/`: generated synthetic datasets and metadata.
- `data/train/`: assembled model-training tables.
- `src/`: data preparation, scan, simulation, modeling, evaluation, reporting, figure, and table scripts.
- `results/tables/`: CSV metrics and intermediate summaries.
- `results/tables_tex/`: LaTeX tables.
- `results/figures/paper/`: paper figures as PDF and PNG.
- `legacy_scripts/`: the preserved single-dataset implementation.

## Data

The three CDC WONDER exports are not committed. Download them from CDC WONDER using these settings for each export:

- Underlying Cause of Death, 1999–2020.
- Group by County only.
- Show totals, show zero values, and show suppressed values.
- Use ICD-10 codes C45, J60–J64, and G12.2.
- Export as tab-delimited text.

Save the files with these exact names:

- `data/raw/mesothelioma.txt.tsv`
- `data/raw/pneumoconioses.txt.tsv`
- `data/raw/als.txt.tsv`

## Install

Use Python 3.10 or newer, create and activate a virtual environment, then install the dependencies:

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. Run commands from the repository root. For commands that accept `--workers`, replace `W` with the number of worker processes appropriate for the machine.

## Full run order

Run these commands in order. Times are rough wall-clock estimates and depend heavily on CPU speed, memory, and the selected worker count; the simulation and scan steps are the longest.

1. Prepare the CDC data: `python src/data_prep.py --all` (about 1–5 minutes).
2. Calculate observed scans and null distributions: `python src/run_scan.py --all` (about 10–60 minutes).
3. Validate scan calculations: `python src/validate_scan.py` (about 1–5 minutes).
4. Generate 400 regional synthetic datasets: `python src/synth.py --count 400 --workers W --start 0` (about 2–12 hours).
5. Generate 50 national synthetic datasets: `python src/synth.py --count 50 --workers W --start 5000 --national` (about 1–4 hours).
6. Assemble training tables: `python src/build_table.py` (about 5–30 minutes).
7. Train and compare full models: `python src/train_compare.py --full` (about 30 minutes–4 hours).
8. Benchmark computational cost: `python src/benchmark_cost.py` (about 5–30 minutes).
9. Calibrate null p-values: `python src/null_calibration.py --count 2000 --workers W --mc99-count 300` (about 1–6 hours).
10. Run the 9,999-replicate reference scan for each dataset (about 1–4 hours each):

    - `python src/run_scan.py --name mesothelioma --replicates 9999 --seed 2026 --tag ref`
    - `python src/run_scan.py --name pneumoconioses --replicates 9999 --seed 2026 --tag ref`
    - `python src/run_scan.py --name als --replicates 9999 --seed 2026 --tag ref`

11. Evaluate against the reference: `python src/eval_reference.py` (about 5–30 minutes).
12. Print result summaries: `python src/summarize.py` (under 1 minute).
13. Generate figures: `python src/make_figures.py --real-source reference` (about 1–5 minutes).
14. Generate LaTeX tables: `python src/make_tables_tex.py --real-source reference` (about 1–5 minutes).

The options above match the argument definitions in the corresponding scripts. `run_scan.py --name` accepts one dataset name, so run it once for each of the three configured datasets. `validate_scan.py`, `build_table.py`, `benchmark_cost.py`, `eval_reference.py`, and `summarize.py` take no command-line options.

## Results

`results/tables/` contains:

- `baseline_repeats.csv`
- `cost.csv` and `cost_by_n.csv`
- `dataset_metrics.csv`
- `gumbel_regression_quality.csv` and `gumbel_tuning.csv`
- `learning_curve.csv`
- `null_calibration.csv` and `null_calibration_summary.csv`
- `paired_bootstrap.csv`
- `real_all_windows.csv` and `real_vs_reference.csv`
- `tuning.csv`
- `type1_power.csv`
- `window_metrics.csv`

`results/tables_tex/` contains `cost_break_even.tex`, `null_calibration.tex`, `real_hard_comparison.tex`, and `synthetic_test_comparison.tex`.

`results/figures/paper/` contains PDF and PNG versions of `fig_break_even`, `fig_clusters`, `fig_learning_curve`, `fig_method_comparison`, `fig_null_calibration`, `fig_pred_vs_true`, `fig_real_hard_f1`, and `fig_runtime`.

The original single-dataset version is preserved in Git tag `v0-single-dataset` and in `legacy_scripts/`.
