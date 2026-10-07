# AgriPredict – Presentation Content (PPT-ready)

**How to use:** one `##` section = one slide. Paste the **On-slide text** into the slide, insert the **Visual**, and read the
**Speaker notes**. All numbers come from the current project (`backend/models/metadata.json`). Images are in
`docs/screenshots/` and `backend/outputs/`.

Golden rule for the whole talk: **ML yield prediction and the rule-based suitability score are two separate things.**

---

## Slide 1 — Title

**On-slide text**
- **AgriPredict**
- AI-Based Rice Yield & Agricultural Decision Support System
- IT in Agricultural Systems (ITA) — College Mini-Project
- Team: _<names / roll numbers>_ · Guide: _<name>_ · _<institution, year>_

**Visual:** `docs/screenshots/01_dashboard.png` (cropped to the top) as a background or right-hand image.

**Speaker notes:** "Our project predicts country-level rice yield from historical data and adds a simple rule-based decision-support layer. It is an academic prototype, and I will be clear about what it can and cannot do."

---

## Slide 2 — Introduction

**On-slide text**
- Rice is a staple crop; yield differs a lot between countries and years
- Data + machine learning can show how yield relates to country, year, climate and input use
- AgriPredict = real data → ML model → API → web dashboard

**Visual:** Simple 4-icon flow: Data → Model → API → Web app.

**Speaker notes:** "We wanted a complete pipeline, not just a notebook: real data, a properly evaluated model, a saved model served by an API, and a usable web interface."

---

## Slide 3 — Problem Statement

**On-slide text**
- Yield varies widely: 0.20 to 10.39 t/ha in our data
- Planners and students need a quick, data-based estimate with honest caveats
- Predictions without validation or warnings can mislead

**Visual:** `backend/outputs/eda_yield_distribution.png`.

**Speaker notes:** "Yield ranges from 0.2 to 10.4 tonnes per hectare across country-years. The problem is giving a data-driven estimate that is validated honestly and that warns the user when it is unreliable."

---

## Slide 4 — Objectives

**On-slide text**
1. Prepare a real rice dataset (unique country-year records)
2. Compare 4 ML models; choose by cross-validation
3. Evaluate on random **and** temporal hold-out; compare with a baseline
4. Serve the saved model via FastAPI + React dashboard
5. Add a rule-based suitability score and recommendations (separate from ML)
6. Warn when inputs are outside the training range

**Visual:** Numbered list only.

**Speaker notes:** "Objectives 2 and 3 are about doing the ML honestly; objective 5 is about keeping the decision-support rules separate from the model."

---

## Slide 5 — Existing System & Disadvantages

**On-slide text** _(general framing, not a literature survey)_
- Manual estimates, experience, or simple historical averages
- No validation on unseen years; no way to flag unreliable inputs
- Estimates and "advice" often mixed together
- Hard to explain or reproduce

**Visual:** Two-column "Existing vs Proposed" table (continues on slide 6).

**Speaker notes:** "This slide is a general description of the common approach, not a formal survey. The point is the gaps we address: validation on later years, warnings, and separating prediction from advice."

---

## Slide 6 — Proposed System

**On-slide text**
- **ML model:** Random Forest predicts rice yield (t/ha)
- **Rule engine:** suitability score → status → risk → recommendations (not ML)
- **Warnings** when inputs are outside the observed range
- **Web app:** Dashboard · Predict · Analytics · About
- Honest evaluation: CV + random + temporal hold-out + baseline

**Visual:** `docs/screenshots/03_predict_japan_2009.png`.

**Speaker notes:** "The screenshot shows the key design choice: the yield card is tagged 'ML model' and the suitability card is tagged 'Rule-based, not ML'."

---

## Slide 7 — System Architecture / Workflow

**On-slide text**
- Inputs: Country · Year · Rainfall · Pesticides · Temperature
- → **ML model** (saved scikit-learn Pipeline) → Predicted yield
- → **Rule engine** (rainfall + temperature) → Suitability → Status → Risk → Recommendations
- React ↔ FastAPI ↔ model / rules

**Visual:** Draw the diagram from README §5 (two parallel branches from the input box); or use the architecture card on the About page (`docs/screenshots/05_about.png`, top).

**Speaker notes:** "The suitability score uses only rainfall and temperature and fixed rules. The ML branch and the rule branch only meet at the end, where the risk level compares the predicted yield with the country's history."

---

## Slide 8 — Dataset & Preprocessing

**On-slide text**
- Kaggle "Crop Yield Prediction Dataset" (user `patelris`)
- 28,242 rows, 10 crops → **3,388 rice rows**
- Removed **297** exact duplicates; averaged repeated country-years → **1,502 unique country-years**
- **68 countries · 1990–2013** (2003 missing) · India: 23 records
- Target: yield, **hg/ha ÷ 10,000 = tonnes/ha**
- Inputs: Country, Year, rainfall (mm), pesticides (tonnes/yr), temperature (°C)

**Visual:** Small funnel: 28,242 → 3,388 → 3,091 → 1,502.

**Speaker notes:** "Repeats of the same country-year would leak near-identical rows into train and test, so we collapsed them. Yield, rainfall and pesticides are identical within a country-year; only temperature varies, so we average it. Source note: we confirmed the Kaggle page title and owner; mirror sites credit FAO and the World Bank, but we do not claim which variable came from where."

---

## Slide 9 — Exploratory Data Analysis

**On-slide text**
- Mean yield: **3.29 t/ha (1990) → 4.40 t/ha (2013)**
- Highest median yields: Egypt 9.28, Australia 8.80, Greece 7.39 t/ha
- Correlation with yield: pesticides +0.28 · Year +0.19 · rainfall −0.12 · temperature −0.31
- Rainfall has **one value per country** → overlaps with country identity

**Visual:** `backend/outputs/eda_correlation.png` and `eda_india_yield_trend.png`.

**Speaker notes:** "Correlations are modest; most variation is between countries. The upward trend over time is why we tested Year as a feature. Correlation is not causation."

---

## Slide 10 — Machine Learning Methodology

**On-slide text**
- Pipeline: one-hot country + scaled numerics + model (fitted on training data only)
- Random 80/20 split: 1,201 train / 301 test
- **5-fold CV on training data** → choose model by lowest RMSE
- Final test: **random hold-out**
- Extra test: **temporal hold-out** — train 1990–2008, test 2009–2013
- Baseline: country-average yield (from training data only)

**Visual:** Flowchart: Split → CV → Select → Random test → Temporal test.

**Speaker notes:** "The test set never influenced model selection. The temporal split is a stricter check of 'can we predict later years'. Year was included because it improved every model in every evaluation — reproducible with year_ablation.py."

---

## Slide 11 — Model Comparison & Evaluation

**On-slide text** _(errors in t/ha)_

| Model | CV RMSE | Random R² | Temporal R² |
|---|---|---|---|
| Linear Regression | 0.689 | 0.876 | 0.763 |
| Decision Tree | 0.714 | 0.869 | 0.747 |
| **Random Forest** | **0.561** | **0.929** | **0.800** |
| Gradient Boosting | 0.701 | 0.875 | 0.766 |
| Country-average baseline | 0.806 | 0.842 | 0.643 |

**Visual:** `backend/outputs/model_comparison.png`.

**Speaker notes:** "Random Forest wins on cross-validation, and also on both hold-outs. Note the drop from 0.929 to 0.800 on later years — the temporal number is the cautious one. The baseline is strong because country identity explains a lot."

---

## Slide 12 — Proposed Model: Random Forest

**On-slide text**
- 300 decision trees, averaged (`min_samples_leaf = 2`)
- **Random hold-out:** MAE 0.344 · RMSE 0.505 · R² 0.929
- **Temporal hold-out:** MAE 0.669 · RMSE 0.924 · R² 0.800
- Predictive feature importance (drop in R²): pesticides 0.71 · temperature 0.57 · rainfall 0.37 · country 0.21 · year 0.14
- **Not causal.** Rainfall overlaps with country.

**Visual:** `docs/screenshots/01_dashboard.png` (actual-vs-predicted + feature-importance charts) or `backend/outputs/actual_vs_predicted_temporal.png`.

**Speaker notes:** "Feature importance means: how much performance drops when we shuffle that feature. It does not mean temperature causes anything. Because rainfall is fixed per country, 'country' and 'rainfall' share information; shuffling both together drops R² by 0.65."

---

## Slide 13 — Agricultural Suitability & Recommendations (rule-based, not ML)

**On-slide text**
- Score = 100 × (0.5 × temperature fit + 0.5 × rainfall fit)
- Reference bands (**assumptions**): 22–30 °C · 1000–2000 mm
- ≥75 Favorable · 50–74 Moderate · <50 Poor → Risk Low / Medium / High
- 3–4 short messages, e.g. "Monitor waterlogging/drainage conditions"
- _Illustrative only — not professional agricultural advice_

**Visual:** `docs/screenshots/03_predict_japan_2009.png` (Japan 2009: suitability 67, Moderate, Medium).

**Speaker notes:** "These thresholds are our own assumptions for illustration; they are not validated agronomic standards. A known weakness: Egypt has 51 mm rainfall but yields of 7 to 10 tonnes per hectare, so the rule rates it Poor, because the dataset has no irrigation information."

---

## Slide 14 — Application Screens / Modules

**On-slide text**
- **Dashboard:** dataset + model summary, latest prediction, charts
- **Predict:** form, prediction, suitability, risk, recommendations, **warnings**
- **Analytics:** full model comparison, actual vs predicted, importance, correlation
- **About:** architecture, dataset, methodology, limitations
- API: `/health` `/model-info` `/analytics` `/predict` `/recommendation`

**Visual:** 2×2 grid: `01_dashboard.png`, `03_predict_japan_2009.png`, `02_analytics.png`, `04_predict_warning_temp40.png`.

**Speaker notes:** "Screenshot 4 shows the warning feature: India with 40 °C is outside the observed range of 3.57 to 30.42 °C, so the app still answers but flags the result as unreliable."

---

## Slide 15 — Results, Limitations & Future Scope

**On-slide text**
- **Results:** RF R² 0.929 (random) / **0.800 (temporal)**; baseline 0.842 / 0.643
- **Limitations:** global country-level annual data; only 23 India records; no soil/NPK/moisture; not causal; not farm-level; tree models don't extrapolate; thresholds are assumptions
- **Future scope (not implemented):** India/district data, soil + NPK, soil moisture, weather forecasts, satellite imagery, IoT sensors, time-series models

**Visual:** Three-column layout: Results | Limitations | Future scope.

**Speaker notes:** "Say the limitations yourself before you are asked. Future work is a list of ideas; none of it is implemented."

---

## Slide 16 — Conclusion & References

**On-slide text**
- Built an end-to-end system: real data → validated ML → API → web app
- Random Forest selected by cross-validation; temporal R² **0.800** is the cautious estimate
- ML prediction and rule-based suitability are clearly separated
- Useful as a learning/decision-support prototype, **not** for farm-level decisions

**References**
1. Kaggle — *Crop Yield Prediction Dataset* — https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
3. scikit-learn documentation — https://scikit-learn.org
4. FastAPI documentation — https://fastapi.tiangolo.com
5. React documentation — https://react.dev

**Visual:** "Thank you / Questions?" with the project name.

**Speaker notes:** "Close by restating the honest summary: good on historical patterns, cautious on later years, and the suitability score is an illustration, not science."
