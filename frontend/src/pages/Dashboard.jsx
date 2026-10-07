import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getAnalytics } from "../api";
import { ActualVsPredicted, FeatureImportance, YieldHistogram } from "../components/Charts.jsx";
import { Badge, Card, ErrorBox, Loading, StatCard, Tag, riskTone, statusTone } from "../components/UI.jsx";

export default function Dashboard({ info, last, go }) {
  const [an, setAn] = useState(null);
  const [err, setErr] = useState(null);
  const load = () => { setErr(null); getAnalytics().then(setAn).catch((e) => setErr(e.message)); };
  useEffect(load, []);

  const p = last?.result;
  const m = info.best_model_metrics;
  const t = info.best_model_temporal_metrics;
  const ds = info.dataset;

  return (
    <div className="space-y-6">
      <section aria-label="Dataset summary">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">Dataset summary</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard icon="🗂️" label="Records" value={ds.unique_country_years.toLocaleString()} sub="unique country-year rows" />
          <StatCard icon="🌍" label="Countries" value={ds.countries} sub="not India-only" />
          <StatCard icon="📅" label="Year range" value={`${ds.year_range[0]}–${ds.year_range[1]}`} sub="2003 is absent" />
          <StatCard icon="🌾" label="Crop" value="Rice" sub={info.crop} />
        </div>
      </section>

      <section aria-label="Model summary">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">Model summary</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard icon="🤖" label="Best model" value={<span className="text-2xl">{info.best_model}</span>} sub="chosen by 5-fold CV (training data)" />
          <StatCard icon="🎯" label="R²" value={m.R2.toFixed(3)} sub={`temporal hold-out: ${t.R2.toFixed(3)}`} />
          <StatCard icon="📏" label="MAE" value={m.MAE.toFixed(3)} sub={`t/ha · temporal: ${t.MAE.toFixed(3)}`} />
          <StatCard icon="📐" label="RMSE" value={m.RMSE.toFixed(3)} sub={`t/ha · temporal: ${t.RMSE.toFixed(3)}`} />
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Large numbers: random 80/20 hold-out ({info.n_test} unseen records). Small print: temporal hold-out, trained on 1990–2008 and tested on 2009–2013 (a stricter test).
        </p>
      </section>

      <section aria-label="Latest prediction">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">Latest prediction</h2>
        <div className="grid gap-4 sm:grid-cols-3">
          <StatCard icon="📈" label="Predicted yield" value={p ? p.predicted_yield.toFixed(2) : "—"}
            sub={p ? `${p.unit} · ${last.input.country}, ${last.input.year}` : "Run a prediction"} />
          <StatCard icon="🌡️" label="Environmental suitability (rule-based)" value={p ? `${p.suitability_score} / 100` : "—"}
            badge={p ? <Badge tone={statusTone(p.status)}>{p.status}</Badge> : "Run a prediction"} />
          <StatCard icon="⚠️" label="Risk" value={p ? p.risk_level : "—"}
            badge={p ? <Badge tone={riskTone(p.risk_level)}>{p.risk_level} risk</Badge> : "Run a prediction"} />
        </div>
        {!p && (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-4">
            <p className="text-emerald-900">No prediction yet. Enter conditions to fill in these cards.</p>
            <button onClick={() => go("predict")} className="rounded-lg bg-emerald-700 px-4 py-2 text-white hover:bg-emerald-800">
              Go to Prediction
            </button>
          </div>
        )}
      </section>

      <section aria-label="Analytics">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">Analytics (computed from the real data and model)</h2>
        {err && <ErrorBox message={err} onRetry={load} />}
        {!an && !err && <Loading text="Loading analytics..." />}
        {an && (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card title={<>Actual vs predicted yield<Tag kind="ml" /></>} subtitle={`${an.best_model} on the random test set (records the model never saw)`}>
              <ActualVsPredicted data={an.actual_vs_predicted} />
            </Card>
            <Card title="Feature importance" subtitle="Predictive importance (permutation): drop in R² when a feature is shuffled. Not a causal effect.">
              <FeatureImportance data={an.feature_importance} />
            </Card>
            <Card title="Yield distribution" subtitle="Rice yield, all country-year records (tonnes/hectare)">
              <YieldHistogram data={an.yield_histogram} />
            </Card>
            <Card title="India: rice yield by year" subtitle="Historical values from the dataset (tonnes/hectare)">
              <div className="h-64">
                <ResponsiveContainer>
                  <LineChart data={an.india_trend} margin={{ top: 5, right: 15, left: 5, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5ece4" />
                    <XAxis dataKey="year" type="number" domain={["dataMin", "dataMax"]} allowDecimals={false} label={{ value: "Year", position: "insideBottom", offset: -12 }} />
                    <YAxis domain={["auto", "auto"]} label={{ value: "Yield (t/ha)", angle: -90, position: "insideLeft", offset: 5 }} />
                    <Tooltip />
                    <Line type="linear" name="Yield (t/ha)" dataKey="yield" stroke="#2f7d32" strokeWidth={2.5} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        )}
      </section>
    </div>
  );
}
