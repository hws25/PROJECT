# Predicting EV Trip Energy Consumption (MMA3001 project)

How accurately can regression and machine-learning models predict the energy used on an electric-vehicle trip,
and which inputs matter most? Real trip records from Roewe E50 electric cars are used to compare a fixed kWh-per-km rate,
linear regression, a decision tree and a random forest, with all models tested on **vehicles the models have never seen**.

## Key results
Test set: 10 vehicles held out from training (random split, seed 42). Inputs: trip distance, duration, average speed,
hour, weekday, battery temperature, current and voltage. Target: trip energy in kWh (mean 3.95 kWh).

| Model (trip + electrical inputs) | MAE (kWh) | RMSE (kWh) | R² |
|---|---|---|---|
| Fixed rate x distance (baseline) | 0.867 | 1.282 | 0.819 |
| Linear regression | 0.785 | 1.164 | 0.851 |
| Decision tree (tuned) | 0.806 | 1.186 | 0.845 |
| Random forest (tuned) | 0.740 | 1.105 | 0.866 |

- Distance explains most of the variation; a fixed rate of about 0.21 kWh/km is already hard to beat.
- The random forest is best, but only modestly better than linear regression, and it takes far longer to train.
- Errors grow with trip length and the model tends to under-predict slightly (mean residual +0.14 kWh).
- Without current and voltage (which are only known during a trip) the random forest scores MAE 0.784 kWh, R² 0.851.

Full discussion: see the notebooks and `docs/`.

## Data
*Electric Vehicle Trip Energy Consumption Data*, Yang Liu, Chalmers University of Technology
(10,151 trips, 50 vehicles, 13 variables). Stored unmodified in `data/raw/ev_trip_energy.csv`.
**TODO: add the exact citation, DOI/link and licence terms from the dataset page.**

## Project structure
| Path | Contents |
|---|---|
| `notebooks/` | `01` explore and clean, `02` baseline, `03` models, tuning and timing, `04` importance, sensitivity, residuals |
| `src/ev_energy/` | `data.py` (loading, checks, cleaning), `modelling.py` (split, metrics, timing, importance, error summaries) |
| `tests/` | pytest tests for the functions in `src/` |
| `docs/` | decisions log, AI-use log, report draft, saved result tables, generated HTML docs and test report |
| `figures/` | plots used in the report |

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook            # run the notebooks in order (01 to 04)
```
Select the `.venv` kernel in the notebook. Notebook 03 takes a few minutes (grid search).

### Tests and test report
```bash
python -m pytest --html=docs/test_report.html --self-contained-html
```

### HTML documentation of the source code
```bash
PYTHONPATH=src python -m pdoc ev_energy -o docs/html
```
Then open `docs/html/index.html`.

## Method in brief
1. **Clean:** rows outside physically plausible ranges are flagged; the unreliable raw `Speed`, `Latitude` and `Longitude`
   columns are dropped and average speed is recomputed as distance / duration; one row with an impossible 435 V is removed (10,150 trips remain).
2. **Validate:** train/test split by vehicle, so no car appears in both sets; tuning uses 5-fold cross-validation grouped by vehicle on the training set only.
3. **Compare:** fixed-rate baseline, linear regression, decision tree, random forest; metrics MAE, RMSE, R², plus fit and prediction time.
4. **Explain:** grouped permutation importance, step-by-step addition of inputs (sensitivity), and residuals by trip conditions and by vehicle.

## Limitations
- Only 10 test vehicles, one car model (Roewe E50), and one region, so results may not carry over to other EVs.
- Trip energy is recorded in steps of 0.224 kWh, which limits attainable accuracy.
- Current and voltage are measured during a trip, so models using them are an upper bound for pre-trip prediction.
- Errors grow on long trips and the model slightly under-predicts, the risky direction for range estimation.
- Tuned settings were chosen on one split, so repeat-split checks are slightly favourable to the tree and forest.

## AI use acknowledgement
- **Tool:** Claude (Anthropic).
- **What it was used for:** setting up the repository structure; writing the helper functions in `src/ev_energy/`, the tests, and
  the notebook templates; suggesting model comparisons, validation and sensitivity analyses; drafting interpretations of results that I then edited; and explaining git, GitHub and the code. 
- **Approximate contribution:** high for code scaffolding and test writing, medium for interpretation drafts and result analysis; low for the initial project idea, dataset, scope and cleaning decisions as they were my own ideas.
- **How the output was checked:** I ran every notebook and test myself, compared the numbers in the text with my own output, and read the code until I could explain each function
- **Decisions that remained mine:** choice of project and dataset (approved on Ed), the plausible ranges and cleaning rules, the model list, and the wording of the conclusions.
- **Errors and unhelpful suggestions:**
  - Claude gave the Roewe E50 battery capacity as about 18 kWh from memory. The dataset's explanatory file states 22.4 kWh, so I corrected the energy limit to 22.4 kWh. Its first energy bound (60 kWh) was also a loose guess that did not match the data.
  - The first set of plausibility ranges were generic estimates. I revised them using the dataset documentation and the data, and wrote the justification for each range myself.
  - Some of Claude's practical instructions did not work for my setup: the notebook assumed a folder layout that differed from mine and gave an import error, the Python package install went to the wrong environment, and GitHub's website upload would not keep folders, so I had to reorganise files and change approach.
  - Claude wrote draft interpretations of the plots before seeing them. I had to check them against my own plots and rewrite the descriptions. 
  - It also first suggested binning residuals by actual energy, which exaggerates the bias on high-energy trips, and later corrected this. So the high-energy group will always look under-predicted, and the low-energy group over-predicted, we fixed this by grouping trips by something the model is given, such as distance, hour or temperature.

- **Effect on my understanding:**
  - AI sped up setting up the repository, writing the helper functions and tests, and producing the notebooks, which let me spend time on the results. I asked follow up questions to understand parts I was unsure about, such as what `src` and `.py` files are, what the diagonal and horizontal lines in the residual plots mean, and what holding out unseen vehicles does.
  - It also made it easy to accept code I had not written, so I read through the cleaning, splitting and evaluation functions until I could explain each step myself. 
  - Debugging the environment and Git problems took longer than the analysis, and I still need to practice explaining the less obvious results, such as the under-prediction on long trips. 

## Licence
Code released under the MIT licence (see `LICENSE`). The dataset has its own terms; see the Data section.
