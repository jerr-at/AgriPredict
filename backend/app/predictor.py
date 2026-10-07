"""Loads the trained pipeline once and turns request data into a prediction.

Architecture (kept deliberately simple and explicit):

    request ──┬─► ML model (saved sklearn Pipeline) ─► predicted yield
              └─► rule engine (recommendations.py)  ─► suitability, status, risk, tips

The saved pipeline contains the exact preprocessor (one-hot + scaler) fitted during
training, so inference applies identical preprocessing. There is ONE code path for the
model call (_model_yield) and ONE for the rules (_decision_support); /predict and
/recommendation both reuse them.
"""
import json
import logging

import joblib
import pandas as pd
import sklearn

from . import recommendations as rec
from .model import FEATURES, MODELS_DIR

log = logging.getLogger(__name__)

MIN_COUNTRY_RECORDS = 10   # below this, country-specific ranges are not considered reliable
RAIN_PEST_TOLERANCE = 0.20  # heuristic: warn if > 20% beyond the country's recorded min/max
TEMP_TOLERANCE_C = 2.0      # heuristic: warn if > 2 °C beyond the country's recorded min/max
TREE_MODELS = ("Random Forest", "Decision Tree", "Gradient Boosting")


class ModelNotReady(RuntimeError):
    pass


SHORT_NAME = {"Year": "Year", "rainfall_mm": "Average annual rainfall",
              "pesticides_tonnes": "Pesticide use", "avg_temp": "Average temperature"}
UNIT = {"Year": "", "rainfall_mm": " mm", "pesticides_tonnes": " tonnes/year", "avg_temp": " °C"}


def _num(feature: str, v: float) -> str:
    if feature == "Year":
        return str(int(v))
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}".rstrip("0").rstrip(".")


def _fmt(feature: str, v: float) -> str:
    return f"{_num(feature, v)}{UNIT[feature]}"


class Predictor:
    def __init__(self):
        try:
            self.pipeline = joblib.load(MODELS_DIR / "yield_model.pkl")
            self.meta = json.loads((MODELS_DIR / "metadata.json").read_text())
            self.analytics = json.loads((MODELS_DIR / "analytics.json").read_text())
        except FileNotFoundError as e:
            raise ModelNotReady("Model files not found. Run `python train.py` in the backend folder first.") from e
        except Exception as e:  # e.g. pickle written by a different scikit-learn version
            raise ModelNotReady(f"Could not load the saved model ({type(e).__name__}: {e}). "
                                "Re-run `python train.py` in the backend folder.") from e
        trained_with = self.meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            log.warning("Model was trained with scikit-learn %s but %s is installed; re-run train.py if results look off.",
                        trained_with, sklearn.__version__)

    # ------------------------------------------------------------------ helpers
    def _check_country(self, country: str) -> None:
        if country not in self.meta["countries"]:
            raise ValueError(f"Unknown country '{country}'. See /model-info for the {len(self.meta['countries'])} supported countries.")

    def _model_yield(self, country: str, year: int, rainfall_mm: float, pesticides_tonnes: float, avg_temp: float) -> float:
        """The single place where the ML model is called."""
        row = pd.DataFrame([{"Area": country, "Year": year, "rainfall_mm": rainfall_mm,
                             "pesticides_tonnes": pesticides_tonnes, "avg_temp": avg_temp}])[FEATURES]
        return max(float(self.pipeline.predict(row)[0]), 0.0)

    def _decision_support(self, country: str, rainfall_mm: float, avg_temp: float, predicted_yield: float) -> dict:
        """Rule engine only (NOT machine learning): suitability, status, risk, recommendations."""
        s = rec.suitability(rainfall_mm, avg_temp)
        cstats = self.meta["country_stats"].get(country)
        return {
            "suitability_score": s["score"],
            "suitability_breakdown": {"temperature_fit": s["temperature_fit"], "rainfall_fit": s["rainfall_fit"]},
            "status": rec.status_from_score(s["score"]),
            "risk_level": rec.risk_level(s["score"], predicted_yield, cstats),
            "recommendations": rec.build_recommendations(rainfall_mm, avg_temp, s["score"], predicted_yield, country, cstats),
            "disclaimer": rec.DISCLAIMER,
        }

    def _warnings(self, country: str, vals: dict) -> list[str]:
        """Extrapolation warnings. They never block the prediction."""
        out, flagged = [], set()
        tree = self.meta["best_model"] in TREE_MODELS

        # 1) outside the whole training dataset
        for k, v in vals.items():
            r = self.meta["numeric_ranges"][k]
            if v < r["min"] or v > r["max"]:
                flagged.add(k)
                msg = (f"Warning: {SHORT_NAME[k]} ({_fmt(k, v)}) is outside the historical range observed in the "
                       f"training dataset ({_fmt(k, r['min'])} to {_fmt(k, r['max'])}). Prediction may be unreliable.")
                if k == "Year" and v > r["max"] and tree:
                    msg += " Tree-based models do not extrapolate trends beyond the training years."
                out.append(msg)

        # 2) outside the selected country's own history (only where the country has enough records)
        cstats = self.meta["country_stats"].get(country) or {}
        n = cstats.get("n_records", 0)
        if n and n < MIN_COUNTRY_RECORDS:
            out.append(f"Warning: only {n} records exist for {country} in the dataset, so predictions for this country are less reliable.")
        elif n:
            for k, v in vals.items():
                if k in flagged:
                    continue
                lo, hi = cstats["ranges"][k]
                if k == "Year":
                    lo_t, hi_t = lo, hi
                elif k == "avg_temp":
                    lo_t, hi_t = lo - TEMP_TOLERANCE_C, hi + TEMP_TOLERANCE_C
                else:
                    lo_t, hi_t = lo * (1 - RAIN_PEST_TOLERANCE), hi * (1 + RAIN_PEST_TOLERANCE)
                if v < lo_t or v > hi_t:
                    seen = _fmt(k, lo) if lo == hi else f"{_fmt(k, lo)} to {_fmt(k, hi)}"
                    out.append(f"Warning: {SHORT_NAME[k]} ({_fmt(k, v)}) is far from the values recorded for {country} "
                               f"in the dataset ({seen}). The model mainly learned cross-country patterns, so prediction may be unreliable.")
        return out

    # ------------------------------------------------------------------ public API
    def predict(self, country: str, year: int, rainfall_mm: float, pesticides_tonnes: float, avg_temp: float) -> dict:
        self._check_country(country)
        y = self._model_yield(country, year, rainfall_mm, pesticides_tonnes, avg_temp)
        return {
            "crop": self.meta["crop"],
            "predicted_yield": round(y, 2),
            "unit": self.meta["unit"],
            **self._decision_support(country, rainfall_mm, avg_temp, y),
            "warnings": self._warnings(country, {"Year": year, "rainfall_mm": rainfall_mm,
                                                 "pesticides_tonnes": pesticides_tonnes, "avg_temp": avg_temp}),
        }

    def recommend(self, country: str, year: int, rainfall_mm: float, pesticides_tonnes: float, avg_temp: float) -> dict:
        """Rule-engine output. Reuses the same model call and warning logic as predict()."""
        return self.predict(country, year, rainfall_mm, pesticides_tonnes, avg_temp)
