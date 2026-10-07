"""Rule-based decision support: suitability score, status, risk, recommendations.

IMPORTANT - what this module is and is not:
  * It is a transparent, hand-written RULE ENGINE. It is NOT a machine-learning model
    and it does not use the ML prediction to compute the suitability score.
  * The reference ranges below are ASSUMPTIONS chosen by the project authors for
    illustration. They are not universal agronomic thresholds and were not derived
    from the dataset. Edit them here to change the behaviour.
  * The dataset has no soil, moisture or nutrient (N-P-K) columns, so none are assessed.
"""

# Reference band (full score) and the value at which the score falls to zero (ASSUMED values).
# Between a zero-point and the reference band the score rises/falls linearly.
TEMP = {"zero_low": 12.0, "opt_low": 22.0, "opt_high": 30.0, "zero_high": 38.0}      # °C
RAIN = {"zero_low": 300.0, "opt_low": 1000.0, "opt_high": 2000.0, "zero_high": 3500.0}  # mm / year

DISCLAIMER = (
    "Academic prototype; recommendations are illustrative and should not replace "
    "professional agricultural guidance."
)
MAX_RECOMMENDATIONS = 4


def _band_score(x: float, b: dict) -> float:
    """Trapezoid membership: 0 outside the zero points, 1 inside the reference band."""
    if x <= b["zero_low"] or x >= b["zero_high"]:
        return 0.0
    if x < b["opt_low"]:
        return (x - b["zero_low"]) / (b["opt_low"] - b["zero_low"])
    if x <= b["opt_high"]:
        return 1.0
    return (b["zero_high"] - x) / (b["zero_high"] - b["opt_high"])


def suitability(rainfall_mm: float, avg_temp: float) -> dict:
    """Environmental Suitability Score (0-100) = 50% temperature fit + 50% rainfall fit."""
    t = _band_score(avg_temp, TEMP)
    r = _band_score(rainfall_mm, RAIN)
    score = round(100 * (0.5 * t + 0.5 * r))
    return {
        "score": score,
        "temperature_fit": round(100 * t),
        "rainfall_fit": round(100 * r),
    }


def status_from_score(score: int) -> str:
    if score >= 75:
        return "Favorable"
    if score >= 50:
        return "Moderate"
    return "Poor"


def risk_level(score: int, predicted_yield: float | None = None, country: dict | None = None) -> str:
    """Risk starts from the suitability score and is raised one level if the
    predicted yield is below the country's historical lower quartile."""
    levels = ["Low", "Medium", "High"]
    i = 0 if score >= 75 else 1 if score >= 50 else 2
    if predicted_yield is not None and country and predicted_yield < country["yield_p25"]:
        i = min(i + 1, 2)
    return levels[i]


def build_recommendations(rainfall_mm: float, avg_temp: float, score: int,
                          predicted_yield: float | None = None,
                          country_name: str | None = None,
                          country: dict | None = None) -> list[str]:
    """3-4 short, illustrative messages. Priority order: rainfall, temperature,
    overall suitability, yield-vs-history."""
    recs: list[str] = []

    if rainfall_mm < RAIN["opt_low"]:
        recs.append("Rainfall is relatively low compared with the configured reference range. "
                    "Consider monitoring irrigation requirements.")
    elif rainfall_mm > RAIN["opt_high"]:
        recs.append("High rainfall detected. Monitor waterlogging/drainage conditions.")

    if avg_temp > TEMP["opt_high"]:
        recs.append("Temperature is relatively high. Monitor potential heat stress.")
    elif avg_temp < TEMP["opt_low"]:
        recs.append("Temperature is relatively low compared with the configured reference range. "
                    "Monitor for slower growth or cold stress.")

    if score < 50:
        recs.append("Environmental conditions may be less favorable according to the prototype's configured rules.")

    if predicted_yield is not None and country:
        label = country_name or "this country"
        if predicted_yield < country["yield_p25"]:
            recs.append(f"Predicted yield is below the lower quartile of {label}'s historical records in the dataset.")
        elif predicted_yield > country["yield_p75"]:
            recs.append(f"Predicted yield is above the upper quartile of {label}'s historical records in the dataset.")

    if not recs:
        recs.append("Inputs fall inside the configured reference ranges for temperature and rainfall.")
    if len(recs) < 3:
        recs.append("Soil, nutrient (N-P-K) and moisture data are not part of this dataset; check them locally.")
    if len(recs) < 3:
        recs.append("Continue routine monitoring of water supply and crop condition.")
    return recs[:MAX_RECOMMENDATIONS]
