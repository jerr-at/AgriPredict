"""Reproducible training script. Run: python train.py

Stages (the test data is never used to fit anything or to choose the model):
  1. Load + clean the real dataset (unique country-year rows).
  2. Random 80/20 split. The 20% test set is untouched until step 5.
  3. 5-fold cross-validation on the TRAINING part only, for the 4 models + the
     country-mean baseline. Preprocessing sits inside a Pipeline, so encoders/scalers
     are re-fitted on each fold's training rows.
  4. Select the best model by mean CV RMSE (training data only).
  5. Final evaluation on the random hold-out test set.
  6. Temporal hold-out: train on years <= 2008, test on 2009-2013 (a stricter,
     "predict later years" check; NOT used for model selection).
  7. Permutation importance on the random test set, save model + metadata + analytics.
"""
import datetime
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.tree import DecisionTreeRegressor

from app.model import (CAT_FEATURES, FEATURE_LABELS, FEATURES, MODELS_DIR, NUM_FEATURES,
                       OUTPUTS_DIR, TARGET, TEMPORAL_CUTOFF_YEAR, CountryMeanBaseline,
                       build_pipeline, clean, load_raw)

warnings.filterwarnings("ignore")
SEED = 42
N_FOLDS = 5
GREEN, BROWN, GREY = "#2f7d32", "#8d6e63", "#9e9e9e"
BASELINE = "Country-average baseline"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fmt(m: dict) -> str:
    return f"MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  R2 {m['R2']:.3f}"


# ---------------------------------------------------------------- 1. load + clean
raw = load_raw()
df, log = clean(raw)
print("Cleaning log:", json.dumps(log, indent=2))
print(f"Features used by the model: {FEATURES}   Target: {TARGET} (tonnes/hectare)")

X, y = df[FEATURES], df[TARGET]

# ---------------------------------------------------------------- 2. random split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"\nRandom split -> train rows: {len(X_train)}  test rows: {len(X_test)}")

candidates = {
    "Linear Regression": build_pipeline(LinearRegression()),
    "Decision Tree": build_pipeline(DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=SEED)),
    "Random Forest": build_pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
    "Gradient Boosting": build_pipeline(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED)),
}

# ---------------------------------------------------------------- 3. cross-validation (train only)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}


def cv_summary(est) -> dict:
    s = cross_validate(est, X_train, y_train, cv=kf, scoring=scoring)
    out = {}
    for k in ("MAE", "RMSE", "R2"):
        v = s[f"test_{k}"] * (-1 if k != "R2" else 1)
        out[k] = float(v.mean())
        out[f"{k}_std"] = float(v.std())
    return out


cv = {name: cv_summary(est) for name, est in candidates.items()}
cv[BASELINE] = cv_summary(CountryMeanBaseline())

print(f"\n{N_FOLDS}-fold cross-validation on the training split (mean ± std over folds, tonnes/ha)")
print("-" * 92)
for name, s in cv.items():
    print(f"{name:<26} MAE {s['MAE']:.3f}±{s['MAE_std']:.3f}  RMSE {s['RMSE']:.3f}±{s['RMSE_std']:.3f}  R2 {s['R2']:.3f}±{s['R2_std']:.3f}")

# ---------------------------------------------------------------- 4. model selection (CV only)
best_name = min(candidates, key=lambda n: cv[n]["RMSE"])
print(f"\nBest model (lowest mean CV RMSE, training data only): {best_name}")

# ---------------------------------------------------------------- 5. random hold-out
random_res, fitted = {}, {}
for name, est in candidates.items():
    fitted[name] = clone(est).fit(X_train, y_train)
    random_res[name] = metrics(y_test, fitted[name].predict(X_test))
baseline_random_model = CountryMeanBaseline().fit(X_train, y_train)  # country means from TRAIN rows only
random_res[BASELINE] = metrics(y_test, baseline_random_model.predict(X_test))

print("\nRANDOM HOLD-OUT (80/20 split, test rows never used for fitting or selection)")
print("-" * 92)
for name, m in random_res.items():
    print(f"{name:<26} {fmt(m)}")

# ---------------------------------------------------------------- 6. temporal hold-out
tr_mask, te_mask = df["Year"] <= TEMPORAL_CUTOFF_YEAR, df["Year"] > TEMPORAL_CUTOFF_YEAR
Xt_tr, yt_tr, Xt_te, yt_te = X[tr_mask], y[tr_mask], X[te_mask], y[te_mask]
unseen_countries = sorted(set(Xt_te["Area"]) - set(Xt_tr["Area"]))
temporal_res, temporal_pred = {}, {}
for name, est in candidates.items():
    m_ = clone(est).fit(Xt_tr, yt_tr)
    temporal_pred[name] = m_.predict(Xt_te)
    temporal_res[name] = metrics(yt_te, temporal_pred[name])
bt = CountryMeanBaseline().fit(Xt_tr, yt_tr)  # country means from the EARLY years only
temporal_res[BASELINE] = metrics(yt_te, bt.predict(Xt_te))

print(f"\nTEMPORAL HOLD-OUT (train {df.Year[tr_mask].min()}-{TEMPORAL_CUTOFF_YEAR}: {int(tr_mask.sum())} rows | "
      f"test {TEMPORAL_CUTOFF_YEAR + 1}-{df.Year.max()}: {int(te_mask.sum())} rows)")
print(f"(countries only in the test years, handled via fallback: {unseen_countries or 'none'}; model selection was NOT based on this)")
print("-" * 92)
for name, m in temporal_res.items():
    print(f"{name:<26} {fmt(m)}")
best_temporal = min(candidates, key=lambda n: temporal_res[n]["RMSE"])
if best_temporal != best_name:
    print(f"Note: {best_temporal} has the lowest temporal RMSE, but the selected model stays {best_name} "
          "(selection was fixed using CV on training data only).")

best = fitted[best_name]
pred_best = best.predict(X_test)

# ---------------------------------------------------------------- 7. importance (test set, selected model)
pi = permutation_importance(best, X_test, y_test, n_repeats=20, random_state=SEED, n_jobs=-1)
importance = sorted(
    [{"feature": f, "label": FEATURE_LABELS[f], "importance": float(m), "std": float(s)}
     for f, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std)],
    key=lambda d: d["importance"], reverse=True,
)


def grouped_permutation_importance(model, Xe, ye, cols, n_repeats=20, seed=SEED):
    """Shuffle several columns TOGETHER (rows stay consistent) and report the mean R2 drop."""
    rng = np.random.RandomState(seed)
    base = r2_score(ye, model.predict(Xe))
    drops = []
    for _ in range(n_repeats):
        idx = rng.permutation(len(Xe))
        Xp = Xe.copy()
        for c in cols:
            Xp[c] = Xe[c].to_numpy()[idx]
        drops.append(base - r2_score(ye, model.predict(Xp)))
    return float(np.mean(drops))


# Rainfall in this dataset has exactly ONE value per country, so rainfall and country carry the
# same information; shuffling either alone under-states it. Shuffle them together too.
country_group = grouped_permutation_importance(best, X_test, y_test, ["Area", "rainfall_mm"])
print(f"\nPredictive (permutation) importance on the random test set - drop in R2 when a feature is shuffled")
for d in importance:
    print(f"  {d['label']:<46} {d['importance']:.3f} ± {d['std']:.3f}")
print(f"  {'Country + rainfall shuffled together':<46} {country_group:.3f}")

# ---------------------------------------------------------------- save model + metadata
joblib.dump(best, MODELS_DIR / "yield_model.pkl", compress=3)  # full pipeline = preprocessor + model
joblib.dump(best.named_steps["pre"], MODELS_DIR / "preprocessor.pkl")  # fitted preprocessor alone (for reference)

num_stats = {
    c: {k: float(v) for k, v in df[c].describe(percentiles=[0.05, 0.5, 0.95]).items()
        if k in ("min", "5%", "50%", "95%", "max")}
    for c in NUM_FEATURES
}
country_stats = {}
for area, g in df.groupby("Area"):
    last = g.sort_values("Year").iloc[-1]
    country_stats[area] = {
        "n_records": int(len(g)),
        "yield_p25": float(g[TARGET].quantile(0.25)),
        "yield_median": float(g[TARGET].median()),
        "yield_p75": float(g[TARGET].quantile(0.75)),
        "rainfall_mm": float(g["rainfall_mm"].median()),
        "pesticides_tonnes": float(g["pesticides_tonnes"].median()),
        "avg_temp": float(g["avg_temp"].median()),
        # country's own observed range - used to warn when inputs are unrealistic for it
        "ranges": {c: [float(g[c].min()), float(g[c].max())] for c in NUM_FEATURES},
        # the country's most recent real record, used to pre-fill the form
        "latest": {"Year": int(last["Year"]), "rainfall_mm": float(last["rainfall_mm"]),
                   "pesticides_tonnes": float(last["pesticides_tonnes"]), "avg_temp": float(last["avg_temp"]),
                   "yield_t_ha": float(last[TARGET])},
    }

all_models = [
    {"model": n, "cv": cv[n], "random_holdout": random_res[n], "temporal_holdout": temporal_res[n]}
    for n in list(candidates) + [BASELINE]
]

metadata = {
    "crop": "Rice (paddy)",
    "target": TARGET,
    "target_description": "Rice (paddy) yield per hectare",
    "unit": "tonnes/hectare",
    "unit_conversion": "Original column hg/ha. 1 hg = 100 g = 0.1 kg = 0.0001 t, so t/ha = hg/ha / 10,000. "
                       "The model is trained on t/ha directly; predictions are NOT converted again.",
    "features": FEATURES,
    "feature_labels": FEATURE_LABELS,
    "categorical_features": CAT_FEATURES,
    "numeric_features": NUM_FEATURES,
    "countries": sorted(df["Area"].unique().tolist()),
    "numeric_ranges": num_stats,
    "country_stats": country_stats,
    "best_model": best_name,
    "selection_method": f"Lowest mean {N_FOLDS}-fold cross-validated RMSE on the training split (test data not used)",
    "best_model_metrics": random_res[best_name],            # random hold-out (kept for backward compatibility)
    "best_model_temporal_metrics": temporal_res[best_name],
    "all_models": all_models,
    "baseline_country_mean": random_res[BASELINE],
    "baseline_country_mean_temporal": temporal_res[BASELINE],
    "n_records": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "temporal_split": {
        "cutoff_year": TEMPORAL_CUTOFF_YEAR,
        "train_years": [int(df.Year[tr_mask].min()), TEMPORAL_CUTOFF_YEAR],
        "test_years": [TEMPORAL_CUTOFF_YEAR + 1, int(df.Year.max())],
        "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
        "countries_only_in_test_years": unseen_countries,
    },
    "training": {
        "random_state": SEED, "cv_folds": N_FOLDS, "test_size": 0.2,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "note": "The saved model is fitted on the random training split only, so the reported "
                "hold-out metrics describe exactly the model that is served.",
    },
    "cleaning_log": log,
    "sklearn_version": sklearn.__version__,
    "dataset": {
        "name": "Crop Yield Prediction Dataset (file: yield_df.csv)",
        "source": "Kaggle dataset by owner 'patelris'. Dataset descriptions on mirror sites credit FAO and the World Bank as original sources; the per-variable mapping is not documented, so it is not claimed here",
        "url": "https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset",
        "subset": "Rice, paddy",
        "raw_rows": log["raw_rows"], "rice_rows": log["rice_rows"],
        "unique_country_years": log["final_rows"], "countries": log["countries"],
        "year_range": [log["year_min"], log["year_max"]],
    },
}
(MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

# ---------------------------------------------------------------- analytics JSON (real results only)
corr = df[NUM_FEATURES + [TARGET]].corr().round(3)
hist, edges = np.histogram(df[TARGET], bins=20)
india = df[df["Area"] == "India"].sort_values("Year")
analytics = {
    "models": all_models,
    "best_model": best_name,
    "baseline": random_res[BASELINE],
    "baseline_temporal": temporal_res[BASELINE],
    "actual_vs_predicted": [
        {"actual": round(float(a), 3), "predicted": round(float(p), 3)} for a, p in zip(y_test, pred_best)
    ],
    "temporal_actual_vs_predicted": [
        {"actual": round(float(a), 3), "predicted": round(float(p), 3)}
        for a, p in zip(yt_te, temporal_pred[best_name])
    ],
    "feature_importance": importance,
    "grouped_importance": [{"features": ["Country", "Average annual rainfall (mm)"], "importance": country_group}],
    "yield_histogram": [
        {"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "count": int(hist[i])} for i in range(len(hist))
    ],
    "correlation": {"labels": list(corr.columns), "matrix": corr.values.tolist()},
    "india_trend": [{"year": int(r.Year), "yield": round(float(r[TARGET]), 3)} for _, r in india.iterrows()],
}
(MODELS_DIR / "analytics.json").write_text(json.dumps(analytics))

# ---------------------------------------------------------------- charts for the report
names = list(candidates) + [BASELINE]
x = np.arange(len(names)); w = 0.38
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, title in [(axes[0], "R2", "R² (higher is better)"), (axes[1], "RMSE", "RMSE, t/ha (lower is better)")]:
    ax.bar(x - w / 2, [random_res[n][key] for n in names], w, label="Random hold-out", color=GREEN)
    ax.bar(x + w / 2, [temporal_res[n][key] for n in names], w, label="Temporal hold-out (2009-2013)", color=BROWN)
    ax.set_xticks(x, names, rotation=25, ha="right"); ax.set_title(title); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=150); plt.close(fig)


def scatter_png(actual, predicted, title, fname):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(actual, predicted, s=12, alpha=0.5, color=GREEN)
    lim = [0, max(np.max(actual), np.max(predicted)) * 1.05]
    ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual yield (t/ha)"); ax.set_ylabel("Predicted yield (t/ha)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / fname, dpi=150); plt.close(fig)


scatter_png(y_test, pred_best, f"Actual vs predicted - {best_name} (random test set)", "actual_vs_predicted.png")
scatter_png(yt_te, temporal_pred[best_name], f"Actual vs predicted - {best_name} (temporal test 2009-2013)",
            "actual_vs_predicted_temporal.png")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh([d["label"] for d in importance][::-1], [d["importance"] for d in importance][::-1], color=GREEN)
ax.set_xlabel("Predictive (permutation) importance: drop in R² when shuffled")
ax.set_title(f"Feature importance - {best_name}")
fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150); plt.close(fig)

print("\nSaved: models/yield_model.pkl, preprocessor.pkl, metadata.json, analytics.json; charts in outputs/")
