"""Exploratory data analysis. Run: python eda.py
Saves charts + a text summary into backend/outputs/ for the report/presentation.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.model import NUM_FEATURES, OUTPUTS_DIR, TARGET, clean, load_raw

GREEN = "#2f7d32"
OUTPUTS_DIR.mkdir(exist_ok=True)

raw = load_raw()
df, log = clean(raw)

lines = []
def say(s=""):
    print(s)
    lines.append(str(s))

say("=== 1. DATASET OVERVIEW (raw file) ===")
say(f"Shape: {raw.shape}")
say(raw.dtypes.to_string())
say("\nItems (crops) in raw file:")
say(raw["Item"].value_counts().to_string())

say("\n=== 2. CLEANING LOG (rice only) ===")
for k, v in log.items():
    say(f"{k}: {v}")

say("\n=== 3. MISSING VALUES (raw file) ===")
say(raw.isna().sum().to_string())

say("\n=== 4. DESCRIPTIVE STATISTICS (cleaned rice data) ===")
say(df[NUM_FEATURES + [TARGET]].describe().T.round(3).to_string())

say("\n=== 5. OUTLIER CHECK (IQR rule, reported only - values are real) ===")
for c in [f for f in NUM_FEATURES if f != "Year"] + [TARGET]:
    q1, q3 = df[c].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    say(f"{c}: {int(((df[c] < lo) | (df[c] > hi)).sum())} values outside [{lo:.2f}, {hi:.2f}]")

say("\n=== 5b. DATA FACTS THAT AFFECT MODELLING (computed from the cleaned data) ===")
rain_per_country = raw[raw["Item"] == "Rice, paddy"].groupby("Area")["average_rain_fall_mm_per_year"].nunique()
say(f"Distinct rainfall values per country (max over countries): {int(rain_per_country.max())}  "
    "-> rainfall is ONE fixed value per country, so it overlaps with country identity")
n_per_country = df.groupby("Area").size()
say(f"Records per country: min {int(n_per_country.min())}, median {int(n_per_country.median())}, max {int(n_per_country.max())}; "
    f"countries with < 10 records: {', '.join(n_per_country[n_per_country < 10].index)}")
say(f"India records: {int(n_per_country.get('India', 0))}")
my = df.groupby("Year")[TARGET].mean()
say(f"Mean yield across countries: {my.index.min()} = {my.iloc[0]:.2f} t/ha -> {my.index.max()} = {my.iloc[-1]:.2f} t/ha (upward trend; why Year is a feature)")

corr = df[NUM_FEATURES + [TARGET]].corr()
say("\n=== 6. CORRELATION MATRIX ===")
say(corr.round(3).to_string())

# --- charts ---
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="Greens", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_title("Correlation matrix (rice, country-year)")
fig.colorbar(im)
fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "eda_correlation.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df[TARGET], bins=30, color=GREEN, edgecolor="white")
ax.set_xlabel("Rice yield (tonnes/hectare)"); ax.set_ylabel("Country-year records")
ax.set_title("Distribution of rice yield")
fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "eda_yield_distribution.png", dpi=150); plt.close(fig)

for col, label, name in [
    ("rainfall_mm", "Average annual rainfall (mm)", "eda_rainfall_vs_yield.png"),
    ("avg_temp", "Average temperature (°C)", "eda_temp_vs_yield.png"),
    ("pesticides_tonnes", "Pesticides (tonnes, national total)", "eda_pesticides_vs_yield.png"),
]:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df[col], df[TARGET], s=10, alpha=0.4, color=GREEN)
    ax.set_xlabel(label); ax.set_ylabel("Yield (t/ha)"); ax.set_title(f"{label} vs yield")
    if col == "pesticides_tonnes":
        ax.set_xscale("log")
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / name, dpi=150); plt.close(fig)

ind = df[df["Area"] == "India"].sort_values("Year")
if len(ind):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ind["Year"], ind[TARGET], marker="o", color=GREEN)
    ax.set_xlabel("Year"); ax.set_ylabel("Yield (t/ha)"); ax.set_title("India - rice yield by year")
    fig.tight_layout(); fig.savefig(OUTPUTS_DIR / "eda_india_yield_trend.png", dpi=150); plt.close(fig)

(OUTPUTS_DIR / "eda_summary.txt").write_text("\n".join(lines))
say(f"\nSaved charts + eda_summary.txt to {OUTPUTS_DIR}")
