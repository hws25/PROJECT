# Decisions log
Each row: what was decided, why, and where the evidence is. This feeds the "justification" and "lessons" parts of the report.

## Data and cleaning
| Decision | Why | Evidence |
|---|---|---|
| Use the Chalmers EV trip dataset (10,151 trips, 50 vehicles) | Real data with trip-level energy and operating conditions; scope approved on Ed | README; notebook 01 |
| Target = trip energy in kWh | Directly meaningful for range and energy planning | |
| Energy range 0 to 22.4 kWh | A trip cannot use more than the 22.4 kWh pack (dataset explanatory file); data max is 20.83 | `PLAUSIBLE_RANGES` in `data.py` |
| Distance range 0 to 170 km | Reported driving range is 170 km; data max is 97 | same |
| Hour 0 to 24, weekday 0 to 6 | Definitions of the variables | same |
| Voltage 250 to 400 V | Deliberately loose; about 99.8% of values lie in 290 to 317 V; flags a single 435 V row | notebook 01 |
| Temperatures -30 to 60 C | Generic lithium-ion limits; data spans about -3 to 34 C | |
| Duration 1 to 600 min | Data max is 393 min | |
| Latitude 30 to 32, longitude 120 to 123 | Loose bounds around Shanghai; 11 rows have impossible coordinates | notebook 01 |
| Drop raw `Speed` and use distance / duration instead | Documented as average km/h but reaches 930 km/h, has 285 zeros on trips of at least 6 km, and is not proportional to distance / duration; derived speed never exceeds 90 km/h | `figures/speed_check.png` |
| Drop latitude and longitude from the models | Impossible values in 11 rows and hard to interpret as inputs | notebook 01 |
| Remove the 1 row with 435 V | Obvious sensor error; one row only | `clean_data` |
| Keep all other rows, including extreme trips | A long or high-energy trip can be genuine, so extremes are not removed just for being extreme | |
| Dataset documentation says 10,000 trips but the file has 10,151 | Noted as a discrepancy; the file as supplied is used | |

## Validation and models
| Decision | Why | Evidence |
|---|---|---|
| Split by vehicle (80% / 20% of vehicles, seed 42): 40 train, 10 test | Prevents the same car appearing in both sets, so the score reflects unseen vehicles | `split_by_vehicle` |
| Main metric MAE (kWh), also RMSE and R² | MAE is in kWh and easy to interpret; RMSE penalises large errors; R² gives the share of variance explained | notebook 02 |
| Baselines: fixed rate x distance, and linear regression | The problem statement is that a fixed rate is too crude; linear regression is the simplest ML baseline | notebooks 02, 03 |
| Fixed rate computed from training data only | Avoids using test information | `fixed_rate_predict` |
| Tune with 5-fold cross-validation grouped by vehicle on training data only; test vehicles used once | Keeps the test result unbiased | notebook 03 |
| Final settings: tree max_depth 6, min_samples_leaf 20; forest 200 trees, max_depth 12, min_samples_leaf 5 | Lowest cross-validated MAE; best forest settings differ by under 0.001 kWh | notebook 03 |
| Permutation importance with the two temperature columns shuffled together | They are almost identical (correlation 0.99) | notebook 04 |
| Sensitivity: add inputs step by step | Answers trip-only versus trip + electrical | `docs/results_04_sensitivity.csv` |
| Residual analysis by distance, hour, temperature and vehicle | Shows where the model is unreliable and how well it transfers to unseen cars | notebook 04 |

## Results to date (seed 42, 10 test vehicles)
| Model (trip + electrical) | MAE | RMSE | R² |
|---|---|---|---|
| Fixed rate | 0.867 | 1.282 | 0.819 |
| Linear regression | 0.785 | 1.164 | 0.851 |
| Decision tree | 0.806 | 1.186 | 0.845 |
| Random forest | 0.740 | 1.105 | 0.866 |

Random forest, trip-only inputs: MAE 0.784, R² 0.851.

## Lessons learned (extend in the report)
- Check data documentation against the data: the Speed column contradicted its own description.
- A simple baseline (fixed kWh per km) should be added before judging complex models; here it explains most of the variation.
- A split by vehicle gives a more honest score than a random split by trip.
- Small differences between models can be as large as the variation between splits, so repeat the split before claiming a winner.
- Under-prediction on long trips matters for range estimation.
