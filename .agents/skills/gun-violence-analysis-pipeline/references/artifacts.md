# Pipeline Artifacts

Use this reference when you need exact input names, output paths, or script ownership.

## Raw Inputs

- Required:
  - `data/raw/chicago_violence_homicides.csv`
  - `data/raw/chicago_drug_crimes.csv`
  - `data/raw/infrastructure_locations.csv`
  - `data/raw/chicago_socioeconomic_neighborhoods.csv`
- Optional full crime dataset candidates for `src/build_hex_maps.py`:
  - `data/raw/chicago_crimes_2001_to_present.csv`
  - `data/raw/Crimes_-_2001_to_Present.csv`
  - `data/raw/chicago_crimes.csv`

## Script Ownership

- `src/build_infrastructure_data.py`
  - Reads live OSM data via `osmnx`
  - Writes `data/raw/infrastructure_locations.csv`
- `src/build_hex_maps.py`
  - Writes `reports/maps/crime_hex_maps/chicago_hex_map.html`
  - Writes default-size CSV outputs under `data/processed/hex/` such as:
    - `chicago_<crime>_hex_counts.csv`
    - `chicago_<crime>_with_hex.csv`
    - `chicago_<crime>_hex_time_season_counts.csv`
- `src/train_xgboost_hex_model.py`
  - Writes `data/processed/modeling/chicago_hex_modeling_table.csv`
  - Writes task-specific reports under:
    - `reports/modeling/hotspot/`
    - `reports/modeling/count/`
- `src/build_rank_order_plot.py`
  - Reads `data/processed/hex/chicago_homicides_hex_counts.csv`
  - Writes:
    - `reports/figures/homicide_rank_order_plot.png`
    - `reports/figures/interactive_rank_order.html`

## Behavioral Notes

- `src/build_hex_maps.py` auto-detects latitude and longitude columns, coerces them to numeric, drops null coordinates, and filters rows to a fixed Chicago bounding box.
- `src/train_xgboost_hex_model.py` defaults to `--task all` and `--hex-size-m 500`.
- `notebooks/` may contain stale assumptions or brittle paths. Prefer `src/` scripts for current behavior.
