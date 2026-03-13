# gun-violence-analysis

Class project analyzing possible causes and patterns of gun violence in Chicago.

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

Chicago dataset:
- https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data

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
