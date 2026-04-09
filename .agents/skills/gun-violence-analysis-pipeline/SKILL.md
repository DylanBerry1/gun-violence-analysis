---
name: gun-violence-analysis-pipeline
description: "Repository-specific workflow for the gun-violence-analysis project. Use when Codex needs to run, debug, or modify the Chicago crime pipeline in this repo: refreshing infrastructure data, generating hex-map outputs, recomputing the all-crimes correlation analysis, training the XGBoost hotspot or count models, or rebuilding homicide rank-order plots. Also use when verifying required raw inputs, locating generated CSV or HTML or metrics artifacts, or deciding which script in src/ is the source of truth. Do not use for generic Python or data-science tasks outside this repository."
---

# Gun Violence Analysis Pipeline

Use this skill to work on the operational pipeline without re-deriving commands, inputs, and artifact paths from notebooks or stale prose.

## Source Of Truth

- Prefer scripts in `src/` over notebooks and `README.md` when they disagree.
- Treat generated files under `data/processed/` and `reports/` as outputs, not design docs.
- Keep the default `500m` hex size unless the user explicitly asks to regenerate downstream outputs at another scale.
- Edit source scripts and regenerate outputs. Do not hand-edit generated CSV, HTML, JSON, PNG, or model artifacts.

## Workflow

1. Run commands from the repository root.
2. Choose the stage that matches the task:
   - Refresh infrastructure data only when `data/raw/infrastructure_locations.csv` must be regenerated: `python3 src/build_infrastructure_data.py`
   - Generate hex outputs and the interactive crime map: `python3 src/build_hex_maps.py`
   - Recompute the all-crimes and infrastructure correlation analysis: `python3 src/build_correlation_analysis.py`
   - Train hotspot and or count models: `python3 src/train_xgboost_hex_model.py`
   - Rebuild homicide rank-order plots after homicide hex counts exist: `python3 src/build_rank_order_plot.py`
3. If the user wants the full pipeline, run the stages in that order.
4. Before running a stage, confirm the expected raw inputs exist. Read `references/artifacts.md` if you need the exact file list or output paths.
5. After code changes, run `ruff format` and `ruff check --fix` if `ruff` is available before rerunning the affected stage.

## Verification

- For map work, confirm the combined HTML map and the default-size per-crime CSV outputs were written without runtime errors.
- For correlation-analysis work, confirm the merged hex table plus the expected figure and summary outputs were regenerated under `data/processed/` and `reports/figures/`.
- For modeling work, confirm the modeling table plus task-specific metrics, holdout predictions, feature-importance CSVs, and plots were regenerated under `reports/modeling/hotspot/` and or `reports/modeling/count/`.
- For rank-order work, confirm the PNG and interactive HTML outputs were regenerated under `reports/figures/`.
- For data-pipeline work, sanity-check key fields such as `hex_id`, centroid coordinates, counts, and derived features in the generated tables.
- If the task involves browser review of Folium outputs, ask the operator to start a live server and inspect the page with Chrome DevTools instead of opening the HTML file ad hoc.

## Gotchas

- `build_hex_maps.py` prefers a full Chicago crimes CSV in `data/raw/` and auto-discovers crime types from `Primary Type`; otherwise it falls back to the per-crime CSVs already in `data/raw/`.
- `build_correlation_analysis.py` currently expects the full crimes dataset at `data/raw/Crimes_-_2001_to_Present_20260408.csv`; if that filename changes, the script or symlink must change too.
- The persisted CSV outputs from `build_hex_maps.py` are only for the default `500m` hex size even though the HTML map supports multiple sizes.
- `train_xgboost_hex_model.py` writes reports into task-specific subdirectories under `reports/modeling/`, even when older prose references flatter paths.
- `build_infrastructure_data.py` depends on live OpenStreetMap access and overwrites `data/raw/infrastructure_locations.csv`.
