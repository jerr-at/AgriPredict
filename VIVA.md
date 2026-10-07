# AgriPredict – Viva Preparation

All numbers below come from the current project (`backend/models/metadata.json`, `README.md`).
**Three sentences to memorise:**
1. *"Random Forest predicts country-level rice yield; the suitability score is a separate rule-based illustration, not ML."*
2. *"Random hold-out R² is 0.929, but on later years (temporal hold-out) it is 0.800 — that is the cautious number."*
3. *"It is an academic prototype on country-level historical data; it cannot predict an individual farm's yield."*

---

## A. Fundamentals

**1. What does your project do?**
It predicts rice yield (tonnes/hectare) for a country and year from rainfall, pesticide use and temperature using a Random Forest. It also shows a separate rule-based suitability score, risk level and short recommendations, through a FastAPI backend and a React dashboard.

**2. Why did you choose rice?**
Rice is a staple crop and the dataset has enough rice records (1,502 unique country-years) to evaluate models properly. Keeping to one crop also keeps the problem focused for a one-day-scale mini-project.

**3. What is the target variable?**
Rice (paddy) yield per hectare. The raw file stores it in hg/ha; we divide by 10,000 to get tonnes/ha (1 hg = 0.1 kg = 0.0001 t).

**4. Why is this a regression problem?**
The output is a continuous number (tonnes/ha), not a category. So we use regression models and regression metrics (MAE, RMSE, R²).

**5. What are the inputs of the model?**
Exactly five: Country, Year, average annual rainfall (mm), pesticide use (tonnes/year, national total) and average temperature (°C). The app shows only these; we do not use soil, NPK or moisture because the dataset has none.

**6. Is the suitability score machine learning?**
No. It is a hand-written rule: 50 % temperature fit + 50 % rainfall fit against reference bands that are our own assumptions. The ML model only produces the yield.

---

## B. Dataset

**7. Which dataset did you use and where is it from?**
The Kaggle "Crop Yield Prediction Dataset" by user `patelris`, covering 10 crops; we use the "Rice, paddy" rows. We confirmed the Kaggle page title and owner name; mirror sites credit FAO and the World Bank, but we do not claim which variable came from which organisation.

**8. How big is it?**
28,242 raw rows (all crops), 3,388 raw rice rows, and 1,502 unique country-year records after cleaning. It covers 68 countries from 1990 to 2013, with 2003 missing.

**9. Why did you remove duplicates and collapse repeated country-years?**
The file repeats the same country-year several times (297 exact duplicates plus 1,589 repeated rows). Near-identical rows in both the training and test sets would inflate scores, so we kept one row per country-year.

**10. Why average the repeats instead of picking one?**
Within each country-year, yield, rainfall and pesticides have exactly one distinct value; only temperature varies (in 460 country-years). Averaging the temperature loses nothing else.

**11. Is the dataset India-specific?**
No, it is global; India has only 23 records. So the app is a country-level demonstration, not an India-specific tool.

**12. What are the dataset's weaknesses?**
It is country-level annual data with no soil, NPK, moisture or irrigation information. Rainfall is one fixed number per country, pesticides are a national total, and it is observational, so patterns are not causes.

---

## C. Preprocessing and features

**13. Why is Country a feature, and how is it encoded?**
Countries differ a lot in typical yield, so country identity carries much of the signal. It is one-hot encoded because countries have no natural order.

**14. Why did you use Year? (and "Why not Year?")**
We did use it. Mean yield rises from 3.29 t/ha (1990) to 4.40 t/ha (2013), and adding Year lowered RMSE for all four models in cross-validation, random hold-out and temporal hold-out. Year is known when you ask for a prediction, so it is not leakage. The caveat is that tree models cannot extrapolate beyond 2013, so the app warns.

**15. Why standardise the numeric features?**
It helps Linear Regression and is harmless for tree models. It is done inside the Pipeline so the scaler is fitted on training data only.

**16. Why did you use a scikit-learn Pipeline?**
So the same preprocessing is applied in training, in each cross-validation fold and at prediction time. It prevents leakage and removes the risk of training/serving mismatch.

---

## D. Machine learning and validation

**17. Which models did you compare and why?**
Linear Regression, Decision Tree, Random Forest and Gradient Boosting, plus a country-average baseline. Comparing simple to ensemble models on the same folds shows whether the extra complexity is worth it.

**18. Why did you choose Random Forest?**
It had the lowest 5-fold cross-validation RMSE (0.561) on the training data, and it also scored best on both hold-outs. It averages many decision trees, which reduces overfitting compared with a single tree.

**19. What are MAE, RMSE and R²?**
MAE is the average absolute error (0.344 t/ha on the random test set). RMSE is the square root of the mean squared error, so it punishes large errors more (0.505). R² is the share of yield variation explained: 1 is perfect, 0 is no better than predicting the mean (0.929).

**20. What is overfitting and how did you check for it?**
Overfitting is memorising training data and failing on new data. We use cross-validation on training data, an untouched random test set, and a temporal test on later years.

**21. What is data leakage and where could it have happened here?**
Information from test data influencing training. Risks were encoders/scalers fitted on all data, baseline averages computed with test rows, repeated country-year rows landing in both sets, and choosing the model on the test set. We avoided each of these (Pipeline, train-only baseline, deduplication, selection by CV).

**22. How did you choose the model without using the test set?**
By mean 5-fold cross-validated RMSE on the 1,201 training rows only. The test set and the temporal split were used afterwards, only for reporting.

**23. Why do you have both a random hold-out and a temporal hold-out?**
In a random split, the model has usually seen the neighbouring years of the same country, which is optimistic. The temporal hold-out trains on 1990–2008 and tests on 2009–2013, which answers "how well can it predict later years?".

**24. Which result should we trust: 0.929 or 0.800?**
Both are real; they answer different questions. 0.929 is the random hold-out; 0.800 (RMSE 0.924 t/ha) is the more cautious estimate for later years, and we report both.

**25. Why is the country-average baseline so strong?**
Country identity captures large, stable differences in yield (for example Egypt's median is 9.28 t/ha), so just predicting each country's average gives R² 0.842. The ML model adds information from the other features and the upward trend over years, giving 0.929 randomly and 0.800 temporally against the baseline's 0.643.

**26. What does the model add over the baseline, honestly?**
A real but partly Year-driven gain: even plain Linear Regression with Year reaches 0.876 randomly. So the improvement is not magic from the Random Forest alone.

**27. What does feature importance mean here?**
We use permutation importance: how much R² drops when a feature's information is shuffled (pesticides 0.71, temperature 0.57, rainfall 0.37, country 0.21, year 0.14). It is predictive importance, not a causal effect. Rainfall overlaps with country because it has one value per country; shuffling both together gives 0.65.

**28. Why is pesticide use the most important feature? Does it increase yield?**
We cannot say it causes anything. Pesticides vary by country and over time, so they carry information the model finds useful for prediction; we did not investigate the mechanism.

---

## E. Backend and API

**29. How does the saved model reach the web app?**
`train.py` saves the whole fitted Pipeline (preprocessing + Random Forest) with joblib. FastAPI loads it once at startup and uses it for every `/predict` call, so prediction uses exactly the training preprocessing.

**30. What endpoints do you have?**
`GET /health`, `GET /model-info`, `GET /analytics`, `POST /predict` and `POST /recommendation`. `/predict` returns the ML yield plus the rule-based outputs and warnings; `/recommendation` returns the rule-engine output using the same model call.

**31. How do you validate inputs?**
Pydantic rejects missing or extra fields, wrong types, `NaN`/`Infinity`, negative rainfall or pesticides, temperature outside −10…50 °C and non-integer or impossible years. An unknown country also gives HTTP 422 with a readable message. (We found and fixed a bug where `NaN` caused a 500 error.)

**32. What is the extrapolation warning?**
If an input is outside the range seen in the training data (for example temperature above 30.42 °C or a year after 2013) or far from the selected country's own recorded values, the app still answers but shows a reliability warning. Tree-based models cannot extrapolate, so those answers may be unreliable.

---

## F. Frontend, recommendations and limitations

**33. How does the React app work?**
React with Vite calls the API with `fetch`; pages are Dashboard, Predict, Analytics and About, with Recharts for charts and Tailwind for styling. It has loading, error and retry states, and every number on screen comes from the API.

**34. How do you generate the recommendations?**
A small rule engine compares rainfall and temperature with the configured reference bands and, in some cases, the predicted yield with the country's historical quartiles. It returns 3–4 short messages plus the disclaimer that they are illustrative and not professional advice.

**35. Can this predict an individual farmer's yield?**
No. It uses annual, country-level averages and has no soil, nutrient, moisture, irrigation, variety or management data. A farm-level system would need farm- or district-level data.

**36. What are the main limitations and how would you improve the system?**
Global country-level data (23 India records), no soil/NPK/moisture, associations not causes, lower accuracy on later years, and assumed suitability thresholds (for example, Egypt has 51 mm rainfall but yields of 7–10 t/ha, which the rainfall rule rates "Poor"). Future work, not implemented: India/district data, soil and NPK, soil moisture, weather forecasts, satellite imagery, IoT sensors and time-series models.

---

## G. Tricky "gotcha" questions

**37. "Your R² is 0.93 — isn't that too good to be true?"**
It is a random split of country-years, so the model has usually seen neighbouring years of the same country. That is why we also report the temporal hold-out (R² 0.800) and the baseline (0.842 randomly), and why we do not present 0.929 alone.

**38. "Why not use deep learning or LSTM?"**
There are only 1,502 tabular rows, and tree ensembles are strong on small tabular data and easier to explain. Deep learning would add complexity without a clear benefit here; we did not test it.

**39. "Did you test on unseen countries?"**
No. We evaluated on unseen rows and unseen years, but not leave-one-country-out, and the app only accepts the 68 known countries. A leave-country-out test would be a good extension; we did not run it and do not claim a result.

**40. "Are your suitability thresholds scientifically validated?"**
No. They are assumptions chosen by us for illustration (22–30 °C and 1000–2000 mm), not derived from the dataset or validated agronomic standards. That is why the app labels them as assumptions.

**41. "Why did temperature get a high importance but only a −0.31 correlation with yield?"**
Correlation measures a simple linear relationship across all rows, while permutation importance measures how much the model's predictions depend on the feature, including combinations with country and year. Neither one is a causal effect.

**42. "What happens if I enter a country that is not in the dataset?"**
The UI lists only the 68 known countries, and the API returns HTTP 422 saying the country is unknown.

---

# 5-Minute Demo Script

**Before you start (2 min, do this before the session):** run `python train.py`, start the API (`uvicorn app.main:app --port 8000`), start the UI (`npm run dev`), open http://localhost:5173, and check the Dashboard loads. Keep `docs/screenshots/` open as a fallback if anything breaks.

| Time | Action | What to say |
|---|---|---|
| 0:00–0:30 | Dashboard top | "AgriPredict predicts rice yield from historical data. 1,502 country-years, 68 countries, 1990–2013. Random Forest was chosen by cross-validation." |
| 0:30–1:00 | Point to Model summary | "Random hold-out R² 0.929, MAE 0.344 t/ha. On later years (temporal hold-out) R² is 0.800 — this is the cautious number." |
| 1:00–2:00 | **Predict → India, 2011**, rainfall 1083, pesticides 55540, temp 25.82 → Predict Yield | "Predicted **3.54 t/ha**; the recorded value is 3.59, from the held-out test set. Suitability **100, Favorable, Low risk** — note this tag says *Rule-based, not ML*." |
| 2:00–2:45 | **Japan, 2009**, 1668, 60971, 15.36 | "Predicted **6.63** (recorded 6.52). Suitability **67, Moderate, Medium**: temperature fit is low (34/100) under our assumed bands." |
| 2:45–3:15 | **Bangladesh, 2001**, 2666, 3296, 25.93 | "Predicted **3.47** (recorded 3.40). Rainfall is above our reference band, so the rule engine says *High rainfall detected. Monitor waterlogging/drainage conditions.*" |
| 3:15–4:00 | **India, 2012**, 1083, 50000, **temperature 40** | "Out-of-range input. The app still answers (3.40) but shows a **reliability warning**: 40 °C is above the training maximum of 30.42 °C." |
| 4:00–4:30 | Analytics tab | "Full comparison. The country-average baseline is strong (R² 0.842), the model beats it (0.929) and also on later years (0.800 vs 0.643). Feature importance here is predictive, not causal." |
| 4:30–5:00 | About tab → Limitations | "Limitations: global country-level data, no soil or NPK, 23 India records, assumed thresholds. It is a prototype, not farm-level advice." |

**Quick extras if asked (30 s each):**
- *Validation:* type `2012.5` in Year → "Enter a whole number". In the API docs (`/docs`), send rainfall `-5` → HTTP 422.
- *Show reproducibility:* in a terminal run `python train.py` and show the results table (~25 s).

**If something breaks:** (1) UI shows "cannot reach API" → start `uvicorn` in `backend/`; (2) model-load error → run `python train.py`; (3) otherwise show the screenshots in `docs/screenshots/`.
