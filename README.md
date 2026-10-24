# ML Spatial Scan Surrogate

ML surrogate for Monte Carlo significance testing in Kulldorff's 
spatial scan statistic — accelerating rare disease cluster detection 
in spatial epidemiology.

## Overview
Kulldorff's spatial scan statistic detects statistically significant 
disease clusters by comparing observed vs expected case counts across 
candidate geographic windows, using Monte Carlo simulation to establish 
significance. This project replaces the expensive Monte Carlo step with 
a trained ML surrogate (gradient boosted regression) that predicts 
p-values directly from window features, validated against baseline 
models (MLR, Random Forest, SVR, MLP).

## Structure
- /data — input datasets (not committed to the repo)
- /src — pipeline: data prep, scan engine, MC labeling, surrogate model, 
  baseline models, evaluation
- /results — trained models, metrics, evaluation plots

## Status
In development.
