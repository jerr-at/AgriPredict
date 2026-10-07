import { useState } from "react";
import { predictYield } from "../api";
import { Badge, Card, ErrorBox, Loading, Tag, riskTone, statusTone } from "../components/UI.jsx";

// Exactly the features the trained model uses: Country + these four numeric inputs.
const FIELDS = [
  { key: "year", meta: "Year", label: "Year", unit: "", min: 1900, max: 2100, step: 1, integer: true,
    help: "Calendar year of the record. The training data covers 1990–2013 (2003 is missing)." },
  { key: "rainfall_mm", meta: "rainfall_mm", label: "Average Annual Rainfall", unit: "mm", min: 0, max: 6000, step: 1,
    help: "In this dataset rainfall is a single fixed average per country (it does not change by year)." },
  { key: "pesticides_tonnes", meta: "pesticides_tonnes", label: "Pesticide Use", unit: "tonnes/year", min: 0, max: 1000000, step: 1,
    help: "National total in tonnes per year, not per hectare." },
  { key: "avg_temp", meta: "avg_temp", label: "Average Temperature", unit: "°C", min: -10, max: 50, step: 0.1,
    help: "Average temperature in degrees Celsius." },
];

// Pre-fill with the country's most recent REAL record from the dataset.
const defaultsFor = (info, country) => {
  const l = info.country_stats[country].latest;
  return {
    year: String(l.Year),
    rainfall_mm: String(Math.round(l.rainfall_mm)),
    pesticides_tonnes: String(Math.round(l.pesticides_tonnes)),
    avg_temp: String(Number(l.avg_temp.toFixed(1))),
  };
};

const fmtRange = (k, r) => (k === "Year" ? `${r[0]} to ${r[1]}` : r[0] === r[1] ? `${r[0].toFixed(1)} (single value)` : `${r[0].toFixed(1)} to ${r[1].toFixed(1)}`);

export default function Predict({ info, last, setLast }) {
  const [country, setCountry] = useState(last?.input.country ?? (info.countries.includes("India") ? "India" : info.countries[0]));
  const [vals, setVals] = useState(
    last ? Object.fromEntries(FIELDS.map((f) => [f.key, String(last.input[f.key])])) : defaultsFor(info, country)
  );
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState(null);
  const result = last?.result;
  const stats = info.country_stats[country];

  const changeCountry = (c) => {
    setCountry(c);
    setVals(defaultsFor(info, c));
    setErrors({});
  };

  const validate = () => {
    const e = {};
    for (const f of FIELDS) {
      const raw = vals[f.key].trim();
      const n = Number(raw);
      if (raw === "" || Number.isNaN(n)) e[f.key] = "Enter a number";
      else if (f.integer && !Number.isInteger(n)) e[f.key] = "Enter a whole number";
      else if (n < f.min || n > f.max) e[f.key] = `Must be between ${f.min} and ${f.max}`;
    }
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const submit = async (ev) => {
    ev.preventDefault();
    setApiError(null);
    if (!validate()) return;
    const input = { country, ...Object.fromEntries(FIELDS.map((f) => [f.key, Number(vals[f.key])])) };
    setLoading(true);
    try {
      const r = await predictYield(input);
      setLast({ input, result: r });
    } catch (e) {
      setApiError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <Card title="Input conditions" subtitle="These are exactly the features the trained model uses." className="lg:col-span-2">
        <form onSubmit={submit} noValidate className="space-y-4">
          <div>
            <label htmlFor="country" className="block text-sm font-medium text-slate-700">Country</label>
            <select id="country" value={country} onChange={(e) => changeCountry(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 focus:border-emerald-600 focus:outline-none focus:ring-2 focus:ring-emerald-200">
              {info.countries.map((c) => <option key={c}>{c}</option>)}
            </select>
            <p className="mt-1 text-xs text-slate-400">
              {stats.n_records} records in the dataset. Form pre-filled with {country}'s latest real record ({stats.latest.Year}).
            </p>
          </div>

          {FIELDS.map((f) => (
            <div key={f.key}>
              <label htmlFor={f.key} className="block text-sm font-medium text-slate-700">
                {f.label} {f.unit && <span className="text-slate-400">({f.unit})</span>}
              </label>
              <input id={f.key} type="number" inputMode="decimal" step={f.step} value={vals[f.key]}
                onChange={(e) => setVals({ ...vals, [f.key]: e.target.value })}
                aria-invalid={!!errors[f.key]}
                className={`mt-1 w-full rounded-lg border px-3 py-2 focus:outline-none focus:ring-2 ${
                  errors[f.key] ? "border-red-400 focus:ring-red-200" : "border-slate-300 focus:border-emerald-600 focus:ring-emerald-200"
                }`} />
              {errors[f.key]
                ? <p className="mt-1 text-sm text-red-600">{errors[f.key]}</p>
                : <p className="mt-1 text-xs text-slate-400">{f.help} {country} in the data: {fmtRange(f.meta, stats.ranges[f.meta])}.</p>}
            </div>
          ))}

          <button type="submit" disabled={loading}
            className="w-full rounded-lg bg-emerald-700 px-4 py-2.5 font-semibold text-white hover:bg-emerald-800 disabled:opacity-60">
            {loading ? "Predicting..." : "Predict Yield"}
          </button>
        </form>
      </Card>

      <div className="space-y-4 lg:col-span-3">
        {apiError && <ErrorBox message={apiError} />}
        {loading && <Loading text="Running the model..." />}
        {!result && !loading && !apiError && (
          <Card><p className="text-slate-500">Enter conditions and press <b>Predict Yield</b> to see the result.</p></Card>
        )}

        {result && !loading && (
          <>
            <p className="text-xs text-slate-500">
              Result for {last.input.country}, {last.input.year} · {last.input.rainfall_mm} mm · {last.input.pesticides_tonnes.toLocaleString()} t/yr · {last.input.avg_temp} °C
            </p>

            <Card>
              <div className="grid gap-5 sm:grid-cols-2">
                <div>
                  <div className="text-sm text-slate-500">Predicted Rice Yield <Tag kind="ml" /></div>
                  <div className="mt-1 text-4xl font-bold text-emerald-900">{result.predicted_yield.toFixed(2)}</div>
                  <div className="text-sm text-slate-500">{result.unit}</div>
                </div>
                <div>
                  <div className="text-sm text-slate-500">Environmental Suitability <Tag kind="rule" /></div>
                  <div className="mt-1 text-4xl font-bold text-emerald-900">{result.suitability_score} <span className="text-xl text-slate-400">/ 100</span></div>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-sm">
                    <span className="text-slate-500">Status</span> <Badge tone={statusTone(result.status)}>{result.status}</Badge>
                    <span className="text-slate-500">Risk</span> <Badge tone={riskTone(result.risk_level)}>{result.risk_level}</Badge>
                  </div>
                </div>
              </div>
              <p className="mt-4 border-t border-emerald-50 pt-3 text-xs text-slate-500">
                Suitability = 50% temperature fit ({result.suitability_breakdown.temperature_fit}/100) + 50% rainfall fit ({result.suitability_breakdown.rainfall_fit}/100),
                using reference ranges that are <b>assumptions</b> of this prototype (see About). It is computed from rainfall and temperature only, independently of the ML prediction.
              </p>
            </Card>

            {result.warnings.length > 0 && (
              <div className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900" role="alert">
                <p className="font-semibold">Reliability warnings</p>
                <ul className="mt-1 list-disc space-y-1 pl-5">
                  {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </div>
            )}

            <Card title="Recommendations" subtitle={result.disclaimer}>
              <ul className="list-disc space-y-2 pl-5 text-slate-700">
                {result.recommendations.map((r, i) => <li key={i}>{r}</li>)}
              </ul>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
