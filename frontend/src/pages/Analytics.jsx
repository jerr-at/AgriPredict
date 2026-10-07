import { useEffect, useState } from "react";
import { getAnalytics } from "../api";
import { ActualVsPredicted, BASELINE, FeatureImportance, Heatmap, ModelComparisonChart, YieldHistogram } from "../components/Charts.jsx";
import { Card, ErrorBox, Loading } from "../components/UI.jsx";

const f3 = (v) => v.toFixed(3);

export default function Analytics({ info }) {
  const [an, setAn] = useState(null);
  const [err, setErr] = useState(null);
  const [view, setView] = useState("random"); // which hold-out the scatter shows
  const load = () => { setErr(null); getAnalytics().then(setAn).catch((e) => setErr(e.message)); };
  useEffect(load, []);

  if (err) return <ErrorBox message={err} onRetry={load} />;
  if (!an) return <Loading text="Loading analytics..." />;

  const ts = info.temporal_split;
  const grouped = an.grouped_importance?.[0];
  const scatter = view === "random" ? an.actual_vs_predicted : an.temporal_actual_vs_predicted;

  return (
    <div className="space-y-6">
      <Card title="Model comparison" subtitle={`Selected: ${an.best_model}. ${info.selection_method}.`}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead>
              <tr className="border-b border-emerald-100 text-left text-slate-500">
                <th className="py-2 pr-3" rowSpan={2}>Model</th>
                <th className="px-2 text-center" colSpan={1}>5-fold CV</th>
                <th className="px-2 text-center" colSpan={3}>Random hold-out ({info.n_test} rows)</th>
                <th className="px-2 text-center" colSpan={3}>Temporal hold-out ({ts.test_years[0]}–{ts.test_years[1]}, {ts.n_test} rows)</th>
              </tr>
              <tr className="border-b border-emerald-100 text-right text-xs text-slate-500">
                <th className="px-2 pb-2">RMSE</th>
                <th className="px-2">MAE</th><th className="px-2">RMSE</th><th className="px-2">R²</th>
                <th className="px-2">MAE</th><th className="px-2">RMSE</th><th className="px-2">R²</th>
              </tr>
            </thead>
            <tbody>
              {an.models.map((m) => {
                const sel = m.model === an.best_model;
                const base = m.model === BASELINE;
                return (
                  <tr key={m.model} className={`border-b border-emerald-50 text-right ${sel ? "bg-emerald-50 font-semibold" : ""} ${base ? "italic text-slate-500" : ""}`}>
                    <td className="py-2 pr-3 text-left">{m.model}{sel && " ✓"}</td>
                    <td className="px-2">{f3(m.cv.RMSE)}</td>
                    <td className="px-2">{f3(m.random_holdout.MAE)}</td><td className="px-2">{f3(m.random_holdout.RMSE)}</td><td className="px-2">{f3(m.random_holdout.R2)}</td>
                    <td className="px-2">{f3(m.temporal_holdout.MAE)}</td><td className="px-2">{f3(m.temporal_holdout.RMSE)}</td><td className="px-2">{f3(m.temporal_holdout.R2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Errors are in tonnes/hectare. ✓ = selected using cross-validation on training data only. The temporal hold-out
          (train {ts.train_years[0]}–{ts.train_years[1]}, test {ts.test_years[0]}–{ts.test_years[1]}) was not used to choose the model.
          Scores are lower on the temporal hold-out; that is the more realistic estimate for predicting later years.
        </p>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="R² by model" subtitle="Higher is better. Baseline = predict the country's average training yield.">
          <ModelComparisonChart models={an.models} metric="R2" />
        </Card>
        <Card title="RMSE by model (tonnes/hectare)" subtitle="Lower is better.">
          <ModelComparisonChart models={an.models} metric="RMSE" />
        </Card>
      </div>
      <p className="-mt-2 text-sm text-slate-600">
        Country identity explains a large part of the variation, so the country-average baseline is strong
        (random hold-out R² {f3(info.baseline_country_mean.R2)}; temporal {f3(info.baseline_country_mean_temporal.R2)}).
        The ML model adds predictive information from the other features, including the upward trend over years, which a plain country average cannot capture.
      </p>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Actual vs predicted yield" subtitle={`${an.best_model}. Records the model never saw. Closer to the dashed line is better.`}>
          <div className="mb-2 flex gap-2" role="group" aria-label="Choose hold-out">
            {[["random", "Random hold-out"], ["temporal", `Temporal hold-out (${ts.test_years[0]}–${ts.test_years[1]})`]].map(([id, label]) => (
              <button key={id} onClick={() => setView(id)}
                className={`rounded-full px-3 py-1 text-xs font-medium ${view === id ? "bg-emerald-700 text-white" : "bg-emerald-50 text-emerald-800 hover:bg-emerald-100"}`}>
                {label}
              </button>
            ))}
          </div>
          <ActualVsPredicted data={scatter} />
        </Card>

        <Card title="Feature importance (predictive)" subtitle="Permutation importance on the random test set: how much R² drops when a feature's information is shuffled. This is NOT a causal effect.">
          <FeatureImportance data={an.feature_importance} />
          {grouped && (
            <p className="mt-2 text-xs text-slate-500">
              Rainfall has a single fixed value per country in this dataset, so it overlaps with Country. Shuffling
              Country and Rainfall <i>together</i> drops R² by {f3(grouped.importance)}, which is a better measure of how much
              "which country" matters than either bar alone. Importances of correlated features are not additive.
            </p>
          )}
        </Card>

        <Card title="Yield distribution" subtitle="Rice yield, all country-year records (tonnes/hectare)">
          <YieldHistogram data={an.yield_histogram} />
        </Card>
      </div>

      <Card title="Feature correlation" subtitle="Pearson correlation between numeric columns (green = positive, amber = negative). Correlation is not causation.">
        <Heatmap labels={an.correlation.labels} matrix={an.correlation.matrix} />
      </Card>
    </div>
  );
}
