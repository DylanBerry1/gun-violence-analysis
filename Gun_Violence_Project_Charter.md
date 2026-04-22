# Project Charter: The Firebreak Theory
## Social Infrastructure as a Structural Barrier to Gun Violence Spread in Chicago

**Team:** Senior Data Science Capstone Team (7 students)  
**Duration:** 15 weeks  
**Focus:** Explanatory Policy Analysis  
**Date:** February 6, 2026

---

## 1. The High-Level Research Question

### The "Why": From Suppression to Resilience

Traditional approaches to gun violence prevention focus on **suppression**—identifying and stopping the shooter, increasing police presence, and reactive interventions. While these tactics address the immediate threat, they operate within a paradigm that treats violence as isolated incidents rather than understanding its **spatial dynamics**.

This project shifts the lens from suppression to **resilience**. We ask: Can neighborhoods be made more resilient to the *spread* of violence, even when individual incidents occur? This question recognizes that gun violence exhibits **contagion-like properties**—clusters emerge, hotspots form, and violence can "spill over" from one area to adjacent neighborhoods. Understanding what *stops* this spread is fundamentally different from understanding what *causes* individual shootings.

### The Core Hypothesis

**The Firebreak Theory:** Gun violence spreads spatially like a contagion, but **Social Infrastructure** (Third Places—libraries, parks, community centers) acts as a structural "firebreak" that dampens this spread.

Just as firebreaks in forestry prevent wildfires from jumping between areas, we hypothesize that these community assets create **social and physical barriers** that:
- Reduce opportunities for conflict escalation
- Increase informal social control and "eyes on the street"
- Provide alternative gathering spaces that reduce street-level tensions
- Create positive social networks that interrupt negative feedback loops

### The Explanatory Goal

Our primary objective is to **measure the causal effect** of social infrastructure density on neighborhood-level violence outcomes, while rigorously controlling for **structural covariates** that create baseline risk (poverty, unemployment, housing instability).

We are not building a predictive model to forecast where shootings will occur. Instead, we seek to **explain** whether investments in social infrastructure can reduce the *spatial spread* of violence, independent of underlying socioeconomic conditions. This explanatory focus makes our findings directly actionable for policy: if social infrastructure acts as a firebreak, then investing in libraries and parks becomes a violence-reduction strategy.

---

## 2. The Data Ecosystem

Chicago serves as our "Digital Twin"—a city with exceptional open data infrastructure that allows us to construct a rigorous natural experiment. Below are the specific data "ingredients" required to build this analysis.

### Ingredient A: The Outcome Variable
**Incident-Level Shooting Data**

**Source:** [Chicago Data Portal - Crimes Dataset](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)  
**Specific Fields Required:**
- `Latitude` and `Longitude` (precise geocoding)
- `Date` and `Time` (timestamp for temporal analysis)
- `Primary Type` (filtered to shooting incidents)
- `Location Description` (contextual information)

**Why This Matters:** We need **precise spatial coordinates** (not just block-level) to identify micro-clusters and measure how violence spreads across space. The temporal component allows us to construct **spatial-temporal models** that can distinguish between "background noise" and "contagious outbreaks." Without precise lat/lon, we cannot measure spillover effects between adjacent census tracts.

**Data Access Method:** Direct download via API or CSV export from Chicago Data Portal. Expected volume: ~10,000+ shooting incidents over a multi-year period.

---

### Ingredient B: The "Endemic" Risk Factors
**Structural Covariates at Census Tract Level**

**Source:** U.S. Census Bureau / American Community Survey (ACS)  
**Python Libraries:** `cenpy` or `censusdis` for programmatic access  
**Geographic Unit:** **Census Tract** (required for neighborhood-level analysis)

**Specific Variables Needed:**
- **Poverty Rate:** Percentage of population below poverty line
- **Unemployment Rate:** Labor force participation and employment status
- **Housing Stability:** Percentage of owner-occupied vs. renter-occupied units, vacancy rates
- **Demographics:** Age distribution, educational attainment
- **Population Density:** Total population per square mile

**Why This Matters:** These variables capture the **baseline risk** that exists independent of social infrastructure. A neighborhood with high poverty and unemployment may have more violence simply due to structural factors. By controlling for these at the **Census Tract level**, we can isolate the effect of social infrastructure from the effect of socioeconomic disadvantage. Without this control, we risk **confounding**—attributing effects to libraries when poverty is the real driver.

**Data Access Method:** 
- `cenpy`: `cenpy.products.ACS(2019).from_place('Chicago, IL', level='tract')`
- `censusdis`: `censusdis.data.download(dataset='acs/acs5', year=2019, variables=['B17001_001E'], state='IL', county='Cook', tract='*')`

**Note:** Census tracts are the ideal unit because they represent ~4,000 people—large enough for stable estimates but small enough to capture neighborhood-level variation.

---

### Ingredient C: The "Treatment" Variable
**Social Infrastructure Locations and Density**

**Source:** OpenStreetMap (OSM) via the `osmnx` Python library  
**Infrastructure Types:**
- **Libraries:** Public libraries and branch locations
- **Parks:** Public parks, green spaces, playgrounds
- **Community Centers:** Recreation centers, community centers, youth centers

**Spatial Metrics to Calculate:**
- **Density:** Number of social infrastructure assets per square kilometer within each census tract
- **Proximity:** Average distance from tract centroid to nearest library/park/center
- **Accessibility:** Number of assets within a 0.5-mile or 1-mile buffer of tract boundary

**Why This Matters:** We need to quantify the **presence and accessibility** of social infrastructure for every neighborhood. Density measures capture whether a tract is "rich" or "poor" in these assets. Proximity measures capture whether residents can easily access these spaces. Both metrics matter: a library in a tract is different from a library three tracts away.

**Data Access Method:**
```python
import osmnx as ox
# Get libraries
libraries = ox.features_from_place('Chicago, IL', tags={'amenity': 'library'})
# Get parks
parks = ox.features_from_place('Chicago, IL', tags={'leisure': 'park'})
# Get community centers
centers = ox.features_from_place('Chicago, IL', tags={'amenity': 'community_centre'})
```

**Validation:** Cross-reference OSM data with official Chicago city datasets (e.g., Chicago Public Library locations) to ensure completeness.

---

### Ingredient D: The Spatial Context
**Census Tract Shapefiles and Adjacency Matrix**

**Source:** U.S. Census Bureau TIGER/Line Shapefiles  
**Python Libraries:** `geopandas`, `libpysal` for spatial analysis

**Required Components:**
- **Census Tract Boundaries:** Shapefile (.shp) for Chicago/Cook County tracts
- **Spatial Weights Matrix:** A matrix defining "who is neighbors with whom"
  - **Queen Contiguity:** Tracts that share a border or vertex
  - **Distance-Based:** Tracts within X kilometers of each other
  - **K-Nearest Neighbors:** Each tract's K closest neighbors

**Why This Matters:** To measure **spillover effects**, we must define the spatial graph. Does violence in Tract A increase the risk in Tract B? The adjacency matrix encodes this relationship. Without it, we cannot test whether social infrastructure in Tract A reduces violence spillover to Tract B—the core of the Firebreak Theory.

**Data Access Method:**
```python
import geopandas as gpd
import libpysal
# Load tracts
tracts = gpd.read_file('tl_2019_17_tract.shp')
# Filter to Cook County
cook_tracts = tracts[tracts['COUNTYFP'] == '031']
# Create spatial weights
w = libpysal.weights.Queen.from_dataframe(cook_tracts)
```

---

## 3. Potential Analytical Paths

Below are high-level methodological approaches that could answer our research question. The team should select one primary path based on data availability and statistical assumptions, but understanding multiple options provides flexibility.

### Path 1: Spatial Autoregressive Models (SAR)
**Concept:** Model violence as a function of local social infrastructure *and* spillover from neighboring tracts.

**Mathematical Intuition:** 
- Violence in Tract i = f(Social Infrastructure in i, Violence in Neighbors of i, Controls)
- The spatial lag term captures contagion; the infrastructure term captures firebreak effect

**Advantages:** 
- Explicitly models spatial dependence
- Can separate "local" effects from "spillover" effects
- Well-established econometric framework

**Challenges:** 
- Requires strong assumptions about spatial structure
- Endogeneity concerns (infrastructure placement may be endogenous to violence)

---

### Path 2: Causal Inference via Matching
**Concept:** Compare "twin neighborhoods" that are similar on all observable characteristics except social infrastructure.

**Mathematical Intuition:**
- Use propensity score matching or coarsened exact matching
- Match tracts with high infrastructure density to tracts with low density (controlling for poverty, demographics, etc.)
- Compare violence outcomes between matched pairs

**Advantages:**
- Addresses confounding by design
- Intuitive interpretation: "What would violence look like if we added a library?"
- Can use difference-in-differences if infrastructure changes over time

**Challenges:**
- Requires sufficient overlap in propensity scores
- Cannot control for unobservable confounders
- May reduce sample size significantly

---

### Path 3: Bayesian Hierarchical Spatial Models
**Concept:** Separate "background risk" (endemic) from "contagious outbreaks" (epidemic) using probabilistic modeling.

**Mathematical Intuition:**
- Model violence as a Poisson process with two components:
  - **Endemic:** Baseline risk driven by structural covariates
  - **Epidemic:** Contagious spread from nearby incidents
- Social infrastructure modifies the *transmission rate* of the epidemic component

**Advantages:**
- Explicitly models the contagion mechanism
- Provides uncertainty quantification
- Can handle sparse data (tracts with zero incidents)

**Challenges:**
- Computationally intensive
- Requires careful prior specification
- Less familiar to policy audiences

---

### Recommendation: Hybrid Approach
Consider combining Paths 1 and 2: Use spatial models to identify spillover patterns, then use matching to estimate causal effects within a subset of well-matched tracts. This provides both **explanatory power** (how does it work?) and **causal identification** (does it work?).

---

## 4. The "So What?": Policy Implications

### If the Hypothesis is Supported

If we find robust evidence that social infrastructure acts as a firebreak—reducing violence spillover even after controlling for poverty—then the policy implications are profound:

**1. Reallocation of Public Safety Resources**
- Shift from purely reactive policing to **proactive infrastructure investment**
- Libraries and parks become **violence prevention tools**, not just amenities
- Cost-benefit analysis: Is a $2M library more effective than $2M in additional police presence?

**2. Spatial Targeting of Investments**
- Identify "firebreak gaps"—neighborhoods with high violence risk but low infrastructure density
- Prioritize infrastructure investments in these areas to create "circuit breakers" in violence networks
- Use spatial analysis to optimize placement (where will one library have the largest spillover-reduction effect?)

**3. Evidence for "Third Place" Theory**
- Validates urban planning principles that emphasize community gathering spaces
- Provides quantitative support for policies like "15-minute cities" and walkable neighborhoods
- Connects to broader research on social capital and community resilience

### If the Hypothesis is Not Supported

Even a null finding is valuable:
- Suggests that structural factors (poverty, inequality) are the primary drivers, and infrastructure alone cannot overcome these forces
- Implies that violence reduction requires **systemic economic change**, not just place-based interventions
- Prevents misallocation of resources toward ineffective solutions

### The Deliverable

The final report should provide:
- **Clear effect size:** "Each additional library per square mile reduces violence spillover by X%"
- **Spatial maps:** Visualizations showing "firebreak gaps" and recommended investment locations
- **Policy brief:** Executive summary for city planners and public safety officials
- **Reproducible code:** All analysis scripts and data pipelines for future researchers

---

## Next Steps

1. **Data Acquisition (Weeks 1-2):** Download and validate all four data ingredients
2. **Data Integration (Weeks 3-4):** Merge datasets at census tract level, create spatial weights matrix
3. **Exploratory Analysis (Weeks 5-6):** Visualize violence clusters, infrastructure distribution, and spatial patterns
4. **Model Development (Weeks 7-11):** Implement primary analytical path, robustness checks
5. **Policy Analysis (Weeks 12-13):** Translate findings into actionable recommendations
6. **Report Writing (Weeks 14-15):** Finalize written report, visualizations, and presentation

---

**Document Version:** 1.0  
**Last Updated:** February 6, 2026
