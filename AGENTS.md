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
- `.agents/skills/`: repo-local skills available to Codex in this project.
- `AGENTS.md`: canonical project-specific agent instructions at the repository root for Codex discovery.
- `CONTINUITY.md`: canonical continuity log at the repository root.

## Repo Skills
- Use `.agents/skills/gun-violence-analysis-pipeline/` for the operational pipeline workflow.
- That skill is the source of truth for:
  - script order and canonical commands
  - expected raw inputs
  - pipeline output paths and artifact ownership
  - workflow verification steps
  - repo-specific pipeline gotchas and source-of-truth rules

## Editing Rules For Agents
- Be careful with output-path changes. The combined map HTML lives under `reports/maps/crime_hex_maps/`, while downstream modeling still depends on per-crime CSV outputs in `data/processed/hex/`.
- If you add a Python package dependency, add it to `requirements.txt` in the same change.
- Avoid broad cleanup changes unless asked. This repository contains generated artifacts and some stale outputs that are useful for comparison.
- If your work creates temporary debug files or obviously stale generated outputs that are no longer part of the intended result, delete them before marking the task finished.

## Documentation Lookup
- Use Context7 MCP when you need current library or API documentation.
- If the target library is known, pin it with slash syntax such as `/org/project` and mention the version you are targeting when relevant.
- Fetch only the minimal documentation needed for the task and summarize it instead of dumping large excerpts.

## CONTINUITY.md (REQUIRED)

Maintain a single continuity file for the current workspace: `CONTINUITY.md`.

- `CONTINUITY.md` is a living document and canonical briefing designed to survive compaction; do not rely on earlier chat/tool output unless it's reflected there.

- At the start of each assistant turn: read `CONTINUITY.md` before acting.
