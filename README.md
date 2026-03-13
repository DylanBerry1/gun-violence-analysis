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
