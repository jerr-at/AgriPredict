"""Shared data loading / cleaning / pipeline logic.

Used by train.py, eda.py and (for paths/feature names) the API, so training and
inference always agree on column names and preprocessing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

CROP_ITEM = "Rice, paddy"  # name used in the FAO-derived dataset

TARGET = "yield_t_ha"
CAT_FEATURES = ["Area"]
NUM_FEATURES = ["Year", "rainfall_mm", "pesticides_tonnes", "avg_temp"]
FEATURES = CAT_FEATURES + NUM_FEATURES

# Human-readable names + units (single source of truth for API/UI/README wording).
FEATURE_LABELS = {
    "Area": "Country",
    "Year": "Year",
    "rainfall_mm": "Average annual rainfall (mm)",
    "pesticides_tonnes": "Pesticide use (tonnes/year, national total)",
    "avg_temp": "Average temperature (°C)",
}

# Temporal hold-out: train on years <= cutoff, test on years > cutoff.
# The data covers 1990-2013 (2003 absent); 2009-2013 is the last 5 years (~22% of rows,
# similar to the 20% used for the random hold-out).
TEMPORAL_CUTOFF_YEAR = 2008

# 1 hectogram = 100 g = 0.1 kg = 0.0001 tonne  ->  hg/ha ÷ 10,000 = tonnes/ha
HG_PER_HA_TO_T_HA = 1.0 / 10_000.0

RENAME = {
    "hg/ha_yield": "yield_hg_ha",
    "average_rain_fall_mm_per_year": "rainfall_mm",
}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def clean(df: pd.DataFrame):
    """Return (clean_df, log).

    Steps: select rice -> drop exact duplicates -> drop missing -> aggregate to
    one row per (country, year) -> basic sanity filters -> convert units.

    Aggregation matters: in the raw file the same country-year is repeated
    several times. Yield, rainfall and pesticides are identical inside every
    country-year group (checked below); only avg_temp can differ, so the repeats
    are averaged. Leaving the repeats would let near-identical rows land in both
    train and test sets and inflate the scores.
    """
    log = {"raw_rows": int(len(df)), "raw_columns": int(df.shape[1])}
    d = df.rename(columns=RENAME)
    d = d[d["Item"] == CROP_ITEM].copy()
    log["rice_rows"] = int(len(d))

    before = len(d)
    d = d.drop_duplicates()
    log["exact_duplicates_removed"] = int(before - len(d))

    log["missing_values_rice"] = int(d.isna().sum().sum())
    d = d.dropna()

    # Evidence that averaging within country-year is safe (recorded in the log).
    grp = d.groupby(["Area", "Year"])
    log["max_distinct_yield_within_country_year"] = int(grp["yield_hg_ha"].nunique().max())
    log["max_distinct_rainfall_within_country_year"] = int(grp["rainfall_mm"].nunique().max())
    log["max_distinct_pesticides_within_country_year"] = int(grp["pesticides_tonnes"].nunique().max())
    log["country_years_with_varying_temperature"] = int((grp["avg_temp"].nunique() > 1).sum())

    g = d.groupby(["Area", "Year"], as_index=False).agg(
        yield_hg_ha=("yield_hg_ha", "mean"),
        rainfall_mm=("rainfall_mm", "mean"),
        pesticides_tonnes=("pesticides_tonnes", "mean"),
        avg_temp=("avg_temp", "mean"),
    )
    log["repeated_rows_collapsed"] = int(len(d) - len(g))
    log["rows_after_aggregation"] = int(len(g))

    before = len(g)
    g = g[
        (g["yield_hg_ha"] > 0)
        & (g["rainfall_mm"] > 0)
        & (g["pesticides_tonnes"] >= 0)
        & (g["avg_temp"].between(-5, 45))
    ]
    log["sanity_rows_removed"] = int(before - len(g))

    g[TARGET] = g["yield_hg_ha"] * HG_PER_HA_TO_T_HA
    g = g.reset_index(drop=True)

    # Hard guarantees the rest of the pipeline relies on.
    assert not g.duplicated(["Area", "Year"]).any(), "country-year must be unique"
    assert np.allclose(g[TARGET] * 10_000, g["yield_hg_ha"]), "unit conversion mismatch"

    log["final_rows"] = int(len(g))
    log["countries"] = int(g["Area"].nunique())
    log["year_min"] = int(g["Year"].min())
    log["year_max"] = int(g["Year"].max())
    log["years_missing_in_range"] = [
        int(y) for y in range(log["year_min"], log["year_max"] + 1) if y not in set(g["Year"])
    ]
    return g, log


def make_preprocessor() -> ColumnTransformer:
    """One-hot country + standardised numerics. Always used INSIDE a Pipeline so it is
    fitted on training data only (also inside every cross-validation fold)."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("pre", make_preprocessor()), ("model", estimator)])


class CountryMeanBaseline(BaseEstimator, RegressorMixin):
    """Reference model: predict the country's mean yield.

    The means are learned in fit(), i.e. from the training rows only (and only from the
    training part of each fold when used inside cross-validation). Unseen countries
    fall back to the overall training mean.
    """

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.country_mean_ = y.groupby(X["Area"]).mean()
        self.global_mean_ = float(y.mean())
        return self

    def predict(self, X):
        return X["Area"].map(self.country_mean_).fillna(self.global_mean_).to_numpy(dtype=float)
