"""Does adding Year help?  Run: python year_ablation.py

Compares every model WITH and WITHOUT Year using the same three views as train.py:
5-fold CV on the training split, random hold-out, temporal hold-out (train <=2008, test 2009-2013).
Nothing here is saved; it only documents why Year is a model input.
"""
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from app.model import TARGET, TEMPORAL_CUTOFF_YEAR, clean, load_raw

warnings.filterwarnings("ignore")
SEED = 42
df, _ = clean(load_raw())
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))


def pipe(est, num):
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), ["Area"]), ("n", StandardScaler(), num)])
    return Pipeline([("pre", pre), ("model", est)])


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=SEED),
}
base_num = ["rainfall_mm", "pesticides_tonnes", "avg_temp"]
tr, te = df.Year <= TEMPORAL_CUTOFF_YEAR, df.Year > TEMPORAL_CUTOFF_YEAR
kf = KFold(5, shuffle=True, random_state=SEED)

print("RMSE in tonnes/hectare (lower is better)")
print(f"{'model':<18}{'view':<10}{'without Year':>14}{'with Year':>12}")
for name, est in models.items():
    row = {}
    for label, num in [("without", base_num), ("with", base_num + ["Year"])]:
        feats = ["Area"] + num
        Xtr, Xte, ytr, yte = train_test_split(df[feats], df[TARGET], test_size=0.2, random_state=SEED)
        p = pipe(clone(est), num)
        cv = -cross_val_score(p, Xtr, ytr, cv=kf, scoring="neg_root_mean_squared_error").mean()
        rnd = rmse(yte, clone(p).fit(Xtr, ytr).predict(Xte))
        tmp = rmse(df.loc[te, TARGET], clone(p).fit(df.loc[tr, feats], df.loc[tr, TARGET]).predict(df.loc[te, feats]))
        row[label] = (cv, rnd, tmp)
    for i, view in enumerate(["CV", "random", "temporal"]):
        print(f"{name if i == 0 else '':<18}{view:<10}{row['without'][i]:>14.3f}{row['with'][i]:>12.3f}")

print("\nMean yield by year (all countries) - the upward trend Year can capture:")
m = df.groupby("Year")[TARGET].mean()
print(f"  {m.index.min()}: {m.iloc[0]:.2f} t/ha  ->  {m.index.max()}: {m.iloc[-1]:.2f} t/ha")
