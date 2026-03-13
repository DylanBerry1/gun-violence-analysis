# AGENTS.md

## Project Overview
- This repository is a Python data-analysis project focused on analyzing gun violence trends and related crime patterns in Chicago using hex-based spatial aggregation.
- The large upstream source dataset is the City of Chicago "Crimes - 2001 to Present" dataset: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data

## Repository Structure
- `src/`: operational scripts for data preparation, map generation, plotting, and XGBoost modeling.
- `data/raw/`: required source CSV inputs.
- `data/processed/hex/`: generated hex-level tables.
- `data/processed/modeling/`: generated modeling table.
- `reports/maps/` and `reports/maps/crime_hex_maps/`: generated HTML maps.
- `reports/modeling/hotspot/` and `reports/modeling/count/`: generated model metrics, holdout predictions, and feature importance outputs.
- `reports/figures/`: generated plots and interactive figures.
- `notebooks/`: exploratory notebooks with some stale assumptions and brittle relative paths.
- `report/`: project writeup and background context, not an implementation spec.

## Canonical Workflow
Run scripts from the repository root.

1. Refresh infrastructure data only when needed:
   - `python3 src/build_infrastructure_data.py`
2. Build hex outputs from raw crime data:
   - `python3 src/build_hex_maps.py`
3. Train the XGBoost models after raw inputs and hex-derived tables are available:
   - `python3 src/train_xgboost_hex_model.py`
4. Build rank-order plots after homicide hex counts exist:
   - `python3 src/build_rank_order_plot.py`

## Required Inputs
Expected raw inputs in `data/raw/`:
- `chicago_violence_homicides.csv`
- `chicago_drug_crimes.csv`
- `infrastructure_locations.csv`
- `chicago_socioeconomic_neighborhoods.csv`

Notes:
- For map generation, the preferred optional input is `data/raw/chicago_crimes_2001_to_present.csv`. When present, `src/build_hex_maps.py` auto-discovers crime types from `Primary Type`.
- The crime CSVs are expected to contain Chicago-style columns such as `Date`, `Latitude`, `Longitude`, `Community Area`, `Block`, `Location Description`, `Arrest`, and `Domestic`.

## Operational Source Of Truth
- Prefer the scripts in `src/` over the notebooks when behavior conflicts.
- Prefer script constants and output paths over `README.md` when they disagree. The README is useful, but it is not fully up to date with the current output layout.
- Do not treat generated files in `reports/` or `data/processed/` as authoritative design docs; they are artifacts of the current scripts.

## Map Builder Conventions
- `src/build_hex_maps.py` is the single source of truth for hex-map generation.
- The script prefers a full Chicago crimes dataset in `data/raw/` and will auto-discover crime types from `Primary Type` when that file is present.
- If no full dataset is present, the script falls back to auto-discovering the per-crime CSV files already stored in `data/raw/`.
- The map scripts auto-detect latitude and longitude columns, coerce them to numeric, drop null coordinates, and filter rows to a fixed Chicago bounding box.
- The interactive combined map renders multiple hex sizes and crime types, but the persisted CSV outputs are based on the default `500m` hex size.

## Modeling Conventions
- `src/train_xgboost_hex_model.py` is the canonical modeling entrypoint.
- The modeling table is one row per hex cell and is written to `data/processed/modeling/chicago_hex_modeling_table.csv`.
- The modeling pipeline joins:
  - homicide counts as targets
  - drug-crime counts and temporal/location features
  - infrastructure counts by type
  - socioeconomic fields mapped from the dominant `Community Area` represented in each hex
- The script supports `--task all|hotspot|count`, but the default project workflow runs both tasks.
- Keep the default `--hex-size-m 500` unless the user explicitly wants downstream outputs regenerated at another scale.

## Artifact Ownership
- `src/build_hex_maps.py` writes the combined crime-selector map HTML under `reports/maps/crime_hex_maps/` and the per-crime hex CSV outputs in `data/processed/hex/`.
- `src/train_xgboost_hex_model.py` writes the modeling table plus task-specific outputs under `reports/modeling/hotspot/` and `reports/modeling/count/`.
- `src/build_rank_order_plot.py` writes the homicide rank-order PNG and interactive HTML outputs in `reports/figures/`.

## Editing Rules For Agents
- Prefer changing source scripts and regenerating outputs rather than hand-editing generated CSV, JSON, PNG, or HTML artifacts.
- Be careful with output-path changes. The combined map HTML lives under `reports/maps/crime_hex_maps/`, while downstream modeling still depends on per-crime CSV outputs in `data/processed/hex/`.
- If you add a Python package dependency, add it to `requirements.txt` in the same change.
- Avoid broad cleanup changes unless asked. This repository contains generated artifacts and some stale outputs that are useful for comparison.
- If your work creates temporary debug files or obviously stale generated outputs that are no longer part of the intended result, delete them before marking the task finished.

## Visual Review
- If you need to review maps or other visual outputs in a browser, use Chrome DevTools MCP for inspection.
- Before doing browser-based review, ask the operator to start a live server and share the local URL.
- Prefer reviewing the generated HTML through that live server rather than opening files ad hoc, especially for interactive Folium outputs.

## Documentation Lookup
- Use Context7 MCP when you need current library or API documentation.
- If the target library is known, pin it with slash syntax such as `/org/project` and mention the version you are targeting when relevant.
- Fetch only the minimal documentation needed for the task and summarize it instead of dumping large excerpts.

## Verification Expectations
There is no automated test suite. Verify work by running the affected script and checking the relevant artifacts.

- After code changes, run `ruff format` and `ruff check --fix` if `ruff` is available in the environment.
- For map changes, run the relevant builder and confirm the expected HTML and CSV outputs were written without obvious runtime errors.
- For modeling changes, run `python3 src/train_xgboost_hex_model.py` and inspect the regenerated metrics, holdout predictions, and feature importance outputs.
- For data-pipeline changes, confirm the raw input columns still match code expectations and the generated tables still contain key fields such as `hex_id`, coordinates, counts, and derived features.

## Known Sharp Edges
- `README.md` and some script docstrings lag the current output layout. Trust the code over top-level prose when they conflict.
- `src/build_infrastructure_data.py` depends on live OpenStreetMap access through `osmnx`.
- `src/build_hex_maps.py` can produce a very large HTML file if many crime types are embedded from a full source dataset.
