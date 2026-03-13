# gun-violence-analysis

Class project for analyzing gun violence trends and related crime patterns in Chicago.

The repository focuses on spatial analysis of Chicago homicides and related crime data using hex-based aggregation, interactive maps, and XGBoost models.

Primary source dataset:
- Chicago Crimes, 2001 to Present: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data

The operational source of truth is the code in `src/`. Generated artifacts under `data/processed/` and `reports/` should usually be regenerated from the scripts instead of edited manually.

## Map Builders

Generate the homicide hex map and homicide hex CSV outputs:

```bash
python3 src/build_homicides_hex_map.py
```

Generate the drug crime hex map and drug hex CSV outputs:

```bash
python3 src/build_drug_hex_map.py
```

Outputs are written to `reports/maps/` and `data/processed/hex/`.

Social infrastructure library:
- `osmnx`

Slides:
- https://docs.google.com/presentation/d/1r3yGWe5nyEBj4HpeNcQ_DzlclLKmw4AoyAubE0k-A0w/edit?usp=sharing

## XGBoost Modeling

Train hex-level XGBoost models for:
- homicide hotspot classification
- homicide count regression

```bash
python3 src/train_xgboost_hex_model.py
```

The script builds one row per 500m hex cell using:
- homicide counts as the target
- drug crime density, timing, and location mix as features
- infrastructure counts by type as features
- socioeconomic indicators mapped by the dominant community area in each hex

Primary modeling outputs:
- `data/processed/modeling/chicago_hex_modeling_table.csv`
- `reports/modeling/hotspot/xgboost_feature_importance.csv`
- `reports/modeling/hotspot/xgboost_feature_groups.csv`
- `reports/modeling/hotspot/xgboost_metrics.json`
- `reports/modeling/count/xgboost_feature_importance.csv`
- `reports/modeling/count/xgboost_feature_groups.csv`
- `reports/modeling/count/xgboost_metrics.json`

## Notes

- The project assumes the large Chicago crime dataset has already been filtered into the raw CSVs stored in `data/raw/`.
- If you change shared map-generation behavior, check all map-builder scripts in `src/` because there is duplicated logic.
- If a task produces temporary debug files or stale generated outputs, remove them before considering the task finished.
