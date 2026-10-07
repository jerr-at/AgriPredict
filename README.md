# AgriPredict – AI-Based Rice Yield & Agricultural Decision Support System

College mini-project for **IT in Agricultural Systems (ITA)**.
Stack: **Python · scikit-learn · FastAPI · React (Vite)**.

> **Academic prototype.** It predicts *country-level, annual* rice yield from historical data. It is **not** a
> farm-level tool and its recommendations are **not** professional agricultural advice.

---

## 1. Project Overview

AgriPredict takes five inputs — **Country, Year, average annual rainfall, pesticide use, average temperature** — and returns two
clearly separate outputs:

| Output | Produced by | What it is |
|---|---|---|
| **Predicted rice yield** (tonnes/hectare) | **Machine-learning model** (Random Forest) trained on real historical data | A statistical prediction |
| **Environmental Suitability Score** (0–100), Status, Risk, Recommendations | **Rule engine** (`backend/app/recommendations.py`) | A transparent, hand-written decision-support score. **Not ML.** |

## 2. Problem Statement

Rice is a staple crop and yields vary strongly between countries and years. Planners and students need a simple way to
explore how historical yield relates to country, year, climate and input use, and to present the result with clear
caveats. This project builds an end-to-end system (data → model → API → web app) that does this using a real public dataset.

## 3. Objectives

1. Clean and prepare a real rice yield dataset (unique country-year records).
2. Compare four regression algorithms fairly and select one without using the test set.
3. Evaluate honestly with a **random hold-out** *and* a **temporal hold-out** (predicting later years).
4. Compare against a **country-average baseline**.
5. Serve the saved model through a **FastAPI** backend and a **React** dashboard.
6. Add a **rule-based** suitability score and recommendations, kept separate from the ML model.
7. Warn users when inputs fall outside the range the model has seen.

## 4. Key Features

- Rice yield prediction in **tonnes/hectare**; inputs are exactly the features the model uses (no invented soil/NPK fields).
- Environmental Suitability Score (0–100) + Status (Favorable / Moderate / Poor) + Risk (Low / Medium / High) — **rule-based**.
- 3–4 short recommendations with an academic-prototype disclaimer.
- **Extrapolation warnings** (year, rainfall, pesticides, temperature, and the selected country's own range).
- Dashboard: dataset summary, model summary, latest prediction, actual-vs-predicted, feature importance, yield distribution.
- Analytics page: full model comparison (CV, random hold-out, temporal hold-out), baseline, correlation heatmap.
- Input validation with useful HTTP 422 errors; loading and error states in the UI; responsive layout.

## 5. System Architecture

```
                         ┌──────────────────────────────┐
                         │   React (Vite) web app       │
                         │ Dashboard · Predict · Analytics · About │
                         └──────────────┬───────────────┘
                                        │ JSON over HTTP
                         ┌──────────────▼───────────────┐
                         │   FastAPI  (backend/app)     │
                         │ validation (Pydantic) + warnings │
                         └───────┬──────────────┬───────┘
                                 │              │
                    ┌────────────▼───┐   ┌──────▼─────────────────────┐
                    │   ML MODEL     │   │   RULE ENGINE (not ML)     │
                    │ saved sklearn  │   │ rainfall + temperature     │
                    │ Pipeline       │   │   → Suitability score      │
                    │ (preprocessing │   │   → Status → Risk          │
                    │  + Random Forest)  │   │   → Recommendations        │
                    └────────┬───────┘   └──────┬─────────────────────┘
                             │ Predicted yield  │
                             └───────► (yield is also used by the risk/recommendation
                                        rules to compare with the country's history)
```

The suitability score itself is computed **only** from rainfall and temperature with fixed rules — it does not use the ML
prediction. (The *risk level* and one recommendation additionally compare the predicted yield with the selected country's
historical quartiles.)

## 6. Technology Stack

| Layer | Tools |
|---|---|
| Data / ML | Python, pandas, NumPy, scikit-learn, joblib, matplotlib |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React 18, Vite, Tailwind CSS, Recharts |

## 7. Dataset

| Item | Value |
|---|---|
| Name | **Crop Yield Prediction Dataset** (file used here: `backend/data/dataset.csv`, a `yield_df.csv`-style file) |
| Source platform | Kaggle |
| URL | https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset |
| Owner (Kaggle user name) | `patelris` |
| Subset used | **Rice, paddy** (the file covers 10 crops) |
| Raw rows in file | 28,242 (all crops) |
| Raw rice rows | **3,388** |
| Final unique country-year records | **1,502** |
| Countries | **68** |
| Years | **1990–2013** (**2003 is missing**) |
| India | **23** records |
| Original columns | `Area` (country), `Item` (crop), `Year`, `hg/ha_yield`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp` |

### Source verification (what is and is not confirmed)

**Verified on 7 Oct 2026 by opening the Kaggle page:** the page exists at the URL above; its title is
*"Crop Yield Prediction Dataset"*; the owner slug in the URL is `patelris`; its short description reads
*"Predicting crop yield of 10 most consumed crops in World"*. The project file has 28,242 data rows, the same count that
mirror sites report for this dataset, and the same column layout.

**Not verified — do not claim these:**
- The full Kaggle description text, licence and file list (the page body could not be retrieved).
- The author's real name (only the Kaggle user name `patelris` is confirmed).
- Mirror-site descriptions credit **FAO and the World Bank** as original sources, but we could not confirm this on the Kaggle
  page itself, and **no source documentation we saw says which variable came from which organisation**, so we do not claim a mapping.
- That `dataset.csv` is byte-identical to Kaggle's file (it is consistent in size and columns, but was not byte-compared).

**Please open the Kaggle page once before submission and copy the exact description/licence yourself.**

## 8. Dataset Preprocessing (done by us)

| Step | Result (from `train.py` / `models/metadata.json`) |
|---|---|
| Select `Item == "Rice, paddy"` | 3,388 rows |
| Remove exact duplicate rows | **297** removed |
| Missing values | none in the rice rows |
| Collapse repeated country-year rows | the file repeats the same country-year several times; **1,589** repeated rows were averaged → **1,502 unique country-years** |
| Why averaging is safe | within every country-year, yield, rainfall and pesticides have exactly **one** distinct value; only temperature differs (in **460** country-years), so we average temperature |
| Sanity filters | yield > 0, rainfall > 0, pesticides ≥ 0, temperature −5…45 °C (0 rows removed) |
| **Target conversion** | the file stores yield in **hg/ha**. 1 hg = 100 g = 0.1 kg = 0.0001 t, so **tonnes/ha = hg/ha ÷ 10,000**. The model is trained on tonnes/ha directly; predictions are not converted again. |

After cleaning, yield averages 3.75 t/ha (median 3.36; range 0.20–10.39).

## 9. Feature Engineering

**Model inputs (exactly five):** `Area` (country, one-hot encoded), `Year`, `rainfall_mm`, `pesticides_tonnes`, `avg_temp`
(the numeric ones are standardised). Everything lives inside one scikit-learn `Pipeline`
(`OneHotEncoder` + `StandardScaler` + model), so encoders and scalers are fitted **only on training rows**, including
inside each cross-validation fold, and the **identical** pipeline is used at prediction time.

**Why Year is included** — decided from evidence, not by default (`python backend/year_ablation.py`):
mean yield rises from 3.29 t/ha (1990) to 4.40 t/ha (2013), and adding Year lowered RMSE for **all four models in all three
evaluations** (CV, random hold-out, temporal hold-out; e.g. Random Forest CV RMSE 0.694 → 0.562, temporal RMSE 1.137 → 0.924).
Year is known when a prediction is requested, so it is not leakage. Caveat: tree-based models cannot extrapolate a trend
past the years they saw, so the app warns for years outside 1990–2013.

## 10. ML Methodology

```
Dataset → Cleaning → Deduplication → Feature preparation → Train/test split (80/20, seed 42)
→ 5-fold cross-validation on the training split → Model comparison → Select by CV RMSE
→ Final random hold-out evaluation → Temporal hold-out evaluation → Saved model → FastAPI → React
```

- **Random split:** 1,201 training / 301 test rows. The test rows are not used for fitting or for choosing the model.
- **Cross-validation:** 5-fold on the training split only, for every model and the baseline.
- **Temporal hold-out:** train on **1990–2008 (1,170 rows)**, test on **2009–2013 (332 rows)**. Used as a stricter check, **not** for selection.
- **Leakage checks:** preprocessing inside the Pipeline; baseline country means computed from training rows only; permutation
  importance computed on the test set after the model was fixed.
- **Saved model:** fitted on the 80 % training split, so the reported hold-out metrics describe exactly the model that is served.

## 11. Models Compared

| Model | Settings |
|---|---|
| Linear Regression | defaults |
| Decision Tree | `max_depth=10`, `min_samples_leaf=5` |
| **Random Forest** | `n_estimators=300`, `min_samples_leaf=2` |
| Gradient Boosting | `n_estimators=300`, `learning_rate=0.05`, `max_depth=3` |
| *Country-average baseline* | predicts the country's mean training yield (not an ML model; reference only) |

## 12. Model Selection

**Random Forest was selected because it had the lowest mean 5-fold cross-validation RMSE (0.561) on the training data.**
No test-set result was used to choose it. It also had the best random and temporal hold-out scores of the four models.

## 13. Evaluation Results

All errors are in **tonnes/hectare**. CV = 5-fold on the training split (RMSE).

| Model | CV RMSE | Random MAE | Random RMSE | Random R² | Temporal MAE | Temporal RMSE | Temporal R² |
|---|---|---|---|---|---|---|---|
| Linear Regression | 0.689 | 0.481 | 0.668 | 0.876 | 0.717 | 1.005 | 0.763 |
| Decision Tree | 0.714 | 0.469 | 0.687 | 0.869 | 0.751 | 1.038 | 0.747 |
| **Random Forest (selected)** | **0.561** | **0.344** | **0.505** | **0.929** | **0.669** | **0.924** | **0.800** |
| Gradient Boosting | 0.701 | 0.518 | 0.669 | 0.875 | 0.749 | 1.000 | 0.766 |
| Country-average baseline | 0.806 | 0.549 | 0.753 | 0.842 | 0.956 | 1.233 | 0.643 |

Charts: `backend/outputs/model_comparison.png`, `actual_vs_predicted.png`, `actual_vs_predicted_temporal.png`.

Numbers are reproducible with `python train.py` (identical to three decimals in a fresh environment with newer library versions).

## 14. Temporal Validation (random vs temporal — please read)

- The **random hold-out** mixes years: for a test row (e.g. India 2011) the model has usually seen India 2010 and 2012 in training.
  This is optimistic. Random Forest scores **R² 0.929** here.
- The **temporal hold-out** trains on 1990–2008 and tests on 2009–2013, which is the realistic question
  *"how well can it predict later years?"*. Random Forest scores **R² 0.800** (RMSE 0.924 t/ha, roughly 1.8× the random-split RMSE).
- **0.929 is not the only result; 0.800 is the more cautious estimate for future years.** Both are reported everywhere.
- One test-period country (Sudan) has no training rows; the pipeline ignores the unseen country and the baseline falls back to the overall mean.

## 15. Baseline Comparison

Country identity captures large, stable differences in yield (for example Egypt's median is 9.28 t/ha), so the
**country-average baseline is strong** (random R² 0.842). The ML model adds predictive information from the other features and from the
upward trend over years, which a plain country average cannot capture.

| | Random R² | Temporal R² |
|---|---|---|
| Country-average baseline | 0.842 | 0.643 |
| Random Forest | 0.929 | 0.800 |

Honest note: part of the gain comes from Year. Even a plain Linear Regression (country + year + climate) reaches R² 0.876 (random) / 0.763 (temporal).

## 16. Feature Importance

Permutation importance on the random test set = **how much R² drops when a feature's information is shuffled**
(*predictive* importance):

| Feature | Importance (drop in R²) |
|---|---|
| Pesticide use | 0.71 |
| Temperature | 0.57 |
| Rainfall | 0.37 |
| Country | 0.21 |
| Year | 0.14 |
| *Country + rainfall shuffled together* | *0.65* |

**These are predictive importances, not causal effects.** Do not read them as "pesticides cause higher yield" or
"temperature lowers yield". Also, **rainfall has exactly one value per country in this dataset**, so it overlaps strongly with
country identity; shuffling either one alone understates how much "which country" matters (hence the combined 0.65 row).
Importances of correlated features are not additive. (Values can differ in the third decimal across scikit-learn versions.)

## 17. Rule-Based Suitability Score (not ML)

`Score = 100 × (0.5 × TemperatureFit + 0.5 × RainfallFit)`, where each fit is 1 inside a reference band and falls linearly to 0:

| Variable | Full score | Falls to 0 at |
|---|---|---|
| Temperature | 22–30 °C | 12 °C and 38 °C |
| Rainfall | 1000–2000 mm/year | 300 mm and 3500 mm |

Status: **≥ 75 Favorable**, **50–74 Moderate**, **< 50 Poor**. Risk: from the score (≥ 75 Low, 50–74 Medium, < 50 High),
raised one level if the predicted yield is below that country's historical lower quartile.

> **These thresholds are assumptions chosen by the project authors for illustration. They were not derived from the dataset
> and are not scientifically validated or universal agronomic standards.**
> Limitation example: Egypt has a single rainfall value of 51 mm yet recorded yields of 7.27–10.08 t/ha (median 9.28), so a
> rainfall-based rule rates it "Poor"; the dataset has no irrigation variable to explain this.

## 18. Recommendation Engine

`recommendations.py` returns 3–4 short messages, e.g. *"Rainfall is relatively low compared with the configured reference
range. Consider monitoring irrigation requirements."*, *"Temperature is relatively high. Monitor potential heat stress."*,
*"High rainfall detected. Monitor waterlogging/drainage conditions."*, *"Environmental conditions may be less favorable
according to the prototype's configured rules."* Every response carries: **"Academic prototype; recommendations are
illustrative and should not replace professional agricultural guidance."**

## 19. FastAPI Backend (`backend/app`)

- `model.py` – data loading/cleaning, shared feature lists, pipeline builder, baseline estimator.
- `predictor.py` – loads the saved pipeline once; one code path for the model call, one for the rules; builds warnings.
- `recommendations.py` – the rule engine. `schemas.py` – Pydantic request/response models. `main.py` – routes.
- Validation: unknown/extra fields rejected; country must exist; year 1900–2100 (integer); rainfall 0–6000 mm;
  pesticides 0–1,000,000 t; temperature −10…50 °C; `NaN`/`Infinity` rejected. Errors return **HTTP 422** with a readable message.
- `/health` returns **503** if the model failed to load.

## 20. React Frontend (`frontend/src`)

- **Dashboard** – dataset summary, model summary (random and temporal), latest prediction, four charts.
- **Predict** – form with units/helper text, pre-filled with the selected country's latest real record; results labelled
  *ML model* vs *Rule-based, not ML*; amber reliability-warnings box; recommendations.
- **Analytics** – comparison table, R²/RMSE charts, actual-vs-predicted (random or temporal), importance, distribution, correlation.
- **About** – architecture diagram, dataset, methodology, suitability rules, limitations.
- All numbers come from the API (no hard-coded results). Loading, error and retry states are handled.

## 21. API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `200` when API and model are ready, else `503` |
| GET | `/model-info` | model name, features, countries, ranges, metrics, dataset and training info |
| GET | `/analytics` | chart data (comparison, actual-vs-predicted, importance, histogram, correlation) |
| POST | `/predict` | ML prediction **plus** rule-based suitability/risk/recommendations/warnings |
| POST | `/recommendation` | rule-engine output (reuses the same model call and warning logic as `/predict`) |

Interactive docs: http://localhost:8000/docs

## 22. Example Prediction

Request (`POST /predict`) — India 2011, a real row from the **random test set** (not seen by the saved model; recorded yield 3.59 t/ha):

```json
{"country":"India","year":2011,"rainfall_mm":1083,"pesticides_tonnes":55540,"avg_temp":25.82}
```

Response (abridged):

```json
{
  "crop": "Rice (paddy)",
  "predicted_yield": 3.54,
  "unit": "tonnes/hectare",
  "suitability_score": 100,
  "suitability_breakdown": {"temperature_fit": 100, "rainfall_fit": 100},
  "status": "Favorable",
  "risk_level": "Low",
  "recommendations": [
    "Predicted yield is above the upper quartile of India's historical records in the dataset.",
    "Soil, nutrient (N-P-K) and moisture data are not part of this dataset; check them locally.",
    "Continue routine monitoring of water supply and crop condition."
  ],
  "warnings": []
}
```

### Tested demo inputs

| Case | Country, Year | Rainfall (mm) | Pesticides (t/yr) | Temp (°C) | Predicted (t/ha) | Recorded (t/ha) | Suitability / Status / Risk |
|---|---|---|---|---|---|---|---|
| 1 | India, 2011 | 1083 | 55540 | 25.82 | **3.54** | 3.59 | 100 / Favorable / Low |
| 2 | Japan, 2009 | 1668 | 60971 | 15.36 | **6.63** | 6.52 | 67 / Moderate / Medium |
| 3 | Bangladesh, 2001 | 2666 | 3296 | 25.93 | **3.47** | 3.40 | 78 / Favorable / Low |
| 4 (warning demo) | India, 2012 | 1083 | 50000 | **40** | 3.40 | — | 50 / Moderate / Medium + warning |

Cases 1–3 are real rows from the random test set; rainfall/pesticides/temperature are rounded as shown and the predictions above were
obtained through the API with exactly these values. Case 4 is deliberately out of range (dataset temperature maximum 30.42 °C)
and returns: *"Warning: Average temperature (40 °C) is outside the historical range observed in the training dataset (3.57 °C to 30.42 °C). Prediction may be unreliable."*

## 23. Installation

Requirements: **Python 3.10+** and **Node.js 18+**. (Tested here with Python 3.12, Node 22, scikit-learn 1.8 and 1.9.)

```bash
git clone <your-repo-or-unzip-the-folder>
cd <project-folder>
```

## 24. Running the Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python eda.py                        # optional: EDA charts + outputs/eda_summary.txt (~2 s)
python train.py                      # trains, evaluates, saves models/ and outputs/ (~25 s)
uvicorn app.main:app --port 8000
```

**Run `python train.py` on your own machine.** A model saved with a different scikit-learn version may not load; if so the API
tells you to re-run it. `python year_ablation.py` (optional, ~30 s) reproduces the "Why Year?" comparison.

## 25. Running the Frontend

```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173  (API expected at http://localhost:8000)
npm run build                        # optional production build check
```

## 26. Project Structure

```
backend/
  app/            main.py, model.py, predictor.py, recommendations.py, schemas.py
  data/           dataset.csv
  models/         yield_model.pkl, preprocessor.pkl, metadata.json, analytics.json   (generated by train.py)
  outputs/        EDA + evaluation charts, eda_summary.txt                           (generated)
  train.py        full training / evaluation pipeline
  eda.py          exploratory analysis
  year_ablation.py  with-vs-without-Year comparison
  requirements.txt
frontend/
  src/            App.jsx, api.js, components/{UI,Charts}.jsx, pages/{Dashboard,Predict,Analytics,About}.jsx
docs/screenshots/ application screenshots (for slides)
README.md  SLIDES.md  VIVA.md
```

## 27. Limitations

- **Global, country-level, annual data** — not India-specific (India has only 23 records) and **not farm-level**. It cannot predict an individual farmer's yield.
- **No soil, N-P-K, soil-moisture, irrigation, variety or management data.**
- **Rainfall is a single fixed value per country**, so it overlaps with country identity; pesticide use is a **national total** (tonnes/year), not per hectare.
- **Country identity explains a large part of the variation** (the baseline is strong); the model can only predict for the 68 countries in the data.
- **Historical observational data** — patterns are associations, not causes; changing one input alone may give non-intuitive results.
- **Performance is lower for later years** (temporal R² 0.800 vs random 0.929); 2003 is missing; three countries have fewer than 10 records (Albania 4, Mauritius 9, Sudan 2).
- **Tree-based models do not extrapolate** beyond 1990–2013 or beyond observed ranges (the app warns).
- **Suitability thresholds are our own assumptions**, not validated; the rainfall rule can disagree with reality where irrigation is used (e.g. Egypt).
- Recommendations are illustrative only.

## 28. Future Enhancements (not implemented)

India- or district-level datasets; soil properties; N-P-K; soil moisture; weather forecasts; satellite imagery; IoT field sensors;
irrigation data; time-series models; a leave-one-country-out evaluation.

## 29. Disclaimer

This is an academic prototype built for coursework. Predictions come from a statistical model trained on historical country-level
data and are **not** guarantees. The suitability score and recommendations are rule-based illustrations using assumed thresholds
and **do not replace professional agricultural guidance**.

## 30. Team / Academic Project

- **Course:** IT in Agricultural Systems (ITA)
- **Project title:** AgriPredict – AI-Based Rice Yield & Agricultural Decision Support System
- **Team members:** _<add names / roll numbers>_
- **Guide / Faculty:** _<add name>_
- **Institution / Academic year:** _<add>_
