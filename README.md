# Predicting Electric Vehicles Energy Consumption (MMA3001)

Predict trip energy consumption (kWh) of Roewe E50 electric vehicles from trip characteristics and vehicle operating data.
Linear regression is the baseline, compared against decision tree and random forest regression.

**Research question:** How accurately can regression and machine-learning models predict EV trip energy consumption from recorded
trip characteristics and vehicle operating data, and which variables contribute most to predictive performance?

## Data
Electric Vehicle Trip Energy Consumption Data, Yang Liu, Chalmers University of Technology (10,151 trips, 13 variables, 50 vehicles).
Stored in `data/raw/ev_trip_energy.csv` (raw data). downloaded from: https://ets-data.sciopen.com/ETSD.2023.9190020

## Project structure
| Path | Contents |
|---|---|
| `notebooks/` | Analysis notebooks based off workflow (01 explore → 02 baseline → 03 models → 04 sensitivity) |
| `src/ev_energy/` | Reusable, documented functions (loading, checks, modelling) |
| `tests/` | Simple tests for the functions |
| `docs/` | Report draft, decisions log, AI-use log |
| `figures/` | Saved plots that will be used in report |

## Reproduce
```bash
pip install -r requirements.txt
jupyter notebook          # run notebooks in order
PYTHONPATH=src pytest     # run tests
```

## Workflow
1. Load and explore 
2. Clean and flag invalid data 
3. Relationships 
4. Grouped train/test split 
5. Linear baseline 
6. Decision tree / random forest + tuning 
7. MAE, RMSE, R² 
8. Feature importance, sensitivity, residuals 
9. Interpretation, limitations, AI reflection.

## Status
- [x] Repo set up, data loaded, exploration notebook
- [ ] Cleaning rules decided
- [ ] Baseline model
- [ ] Tree / forest models and tuning
- [ ] Sensitivity and residual analysis
- [ ] Report
