# gun-violence-analysis

Chicago dataset:
- https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data

Social infrastructure library:
- `osmnx`

Slides:
- https://docs.google.com/presentation/d/1r3yGWe5nyEBj4HpeNcQ_DzlclLKmw4AoyAubE0k-A0w/edit?usp=sharing

## Repository Layout

```
data/
  raw/                # source data (do not edit)
  interim/            # optional cleaned intermediate files
  processed/hex/      # team-ready hex outputs

src/chicago_analysis/ # reusable Python modules
scripts/              # runnable pipeline steps
notebooks/            # exploratory + reporting notebooks
reports/              # generated maps/figures
docs/                 # definitions, framing, dictionary
```

## Main Hex Workflow

Run:

```bash
python3 scripts/03_build_maps.py
```

Outputs:
- `reports/maps/chicago_homicides_hex_map.html`
- `data/processed/hex/chicago_hex_counts.csv`
- `data/processed/hex/chicago_homicides_with_hex.csv`
- `data/processed/hex/chicago_hex_time_season_counts.csv`
