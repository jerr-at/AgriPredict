import { Card } from "../components/UI.jsx";

const Box = ({ children, tone = "slate" }) => {
  const tones = {
    slate: "border-slate-200 bg-white text-slate-700",
    ml: "border-sky-300 bg-sky-50 text-sky-900",
    rule: "border-violet-300 bg-violet-50 text-violet-900",
  };
  return <div className={`rounded-xl border px-3 py-2 text-center text-sm font-medium ${tones[tone]}`}>{children}</div>;
};
const Arrow = () => <div className="text-center text-slate-400" aria-hidden>↓</div>;

export default function About({ info }) {
  const log = info.cleaning_log;
  const ds = info.dataset;
  const ts = info.temporal_split;
  const best = info.best_model;

  const steps = [
    ["Dataset", `${ds.name}, ${log.raw_rows.toLocaleString()} rows covering 10 crops. We use the "${ds.subset}" subset: ${log.rice_rows.toLocaleString()} rows.`],
    ["Cleaning & deduplication", `Removed ${log.exact_duplicates_removed} exact duplicates. Yield, rainfall and pesticides are identical within each country-year; only temperature varies (in ${log.country_years_with_varying_temperature} country-years), so repeats were averaged into ${log.final_rows.toLocaleString()} unique country-year records (${log.countries} countries, ${log.year_min}–${log.year_max}). No missing values. Yield converted from hg/ha to tonnes/ha (÷ 10,000).`],
    ["Feature preparation", "Country is one-hot encoded; Year, rainfall, pesticides and temperature are standardised. All of this sits inside a scikit-learn Pipeline, so it is fitted on training rows only."],
    ["Train/test split", `Random 80/20 split (${info.n_train} / ${info.n_test} records). The test set is untouched until the final evaluation.`],
    ["Cross-validation & model selection", `Linear Regression, Decision Tree, Random Forest and Gradient Boosting compared with 5-fold CV on the training split. ${best} had the lowest CV RMSE.`],
    ["Final evaluation", `Random hold-out R² ${info.best_model_metrics.R2.toFixed(3)}; temporal hold-out (train ${ts.train_years[0]}–${ts.train_years[1]}, test ${ts.test_years[0]}–${ts.test_years[1]}) R² ${info.best_model_temporal_metrics.R2.toFixed(3)}. A country-average baseline is reported alongside.`],
    ["Saved model → FastAPI → React", "The fitted pipeline is saved with joblib and loaded by the API, so prediction uses the exact preprocessing used in training."],
  ];

  return (
    <div className="space-y-6">
      <Card title="System architecture" subtitle="The prediction and the suitability score come from two separate parts">
        <div className="mx-auto max-w-3xl">
          <Box>User input: Country · Year · Rainfall · Pesticides · Temperature</Box>
          <Arrow />
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1 rounded-xl border-2 border-sky-200 p-3">
              <p className="text-center text-xs font-semibold uppercase tracking-wide text-sky-700">Machine learning</p>
              <Box tone="ml">{best} (trained on real data)</Box>
              <Arrow />
              <Box tone="ml">Predicted yield (tonnes/hectare)</Box>
            </div>
            <div className="space-y-1 rounded-xl border-2 border-violet-200 p-3">
              <p className="text-center text-xs font-semibold uppercase tracking-wide text-violet-700">Rule engine (not ML)</p>
              <Box tone="rule">Suitability score from rainfall + temperature</Box>
              <Arrow />
              <Box tone="rule">Status → Risk level → Recommendations</Box>
            </div>
          </div>
          <p className="mt-3 text-sm text-slate-600">
            The suitability score is <b>not</b> a machine-learning prediction. It is a transparent, hand-written academic decision-support rule.
            (The risk level also compares the predicted yield with the country's historical lower quartile.)
          </p>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Dataset" subtitle="Real, public, historical data">
          <dl className="space-y-1.5 text-sm">
            <Item k="Name" v={ds.name} />
            <Item k="Source" v={ds.source} />
            <Item k="URL" v={<a className="break-all text-emerald-700 underline" href={ds.url} target="_blank" rel="noreferrer">{ds.url}</a>} />
            <Item k="Records used" v={`${ds.unique_country_years.toLocaleString()} unique country-years (from ${log.rice_rows.toLocaleString()} raw rice rows)`} />
            <Item k="Countries / years" v={`${ds.countries} countries · ${ds.year_range[0]}–${ds.year_range[1]} (2003 absent)`} />
            <Item k="Model inputs" v="Country, Year, rainfall (mm), pesticides (tonnes/year), temperature (°C)" />
            <Item k="Target" v="Rice (paddy) yield, tonnes/hectare" />
          </dl>
          <p className="mt-3 text-xs text-slate-500">{info.unit_conversion}</p>
        </Card>

        <Card title="Why Year is a model input" subtitle="Decided from the data, not by default">
          <p className="text-sm text-slate-600">
            Average yield rises from year to year in this dataset, and adding Year improved every model in cross-validation,
            on the random hold-out and on the temporal hold-out. Year is known when a prediction is requested, so using it does not leak the target.
            Caveat: tree-based models cannot extrapolate a trend beyond the years they saw, so the app warns for years outside {ds.year_range[0]}–{ds.year_range[1]}.
          </p>
        </Card>
      </div>

      <Card title="Methodology" subtitle="From raw data to recommendation">
        <ol className="space-y-2">
          {steps.map(([t, d], i) => (
            <li key={t}>
              <div className="flex gap-3 rounded-xl border border-emerald-100 bg-emerald-50/50 p-3">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-emerald-700 text-sm font-bold text-white">{i + 1}</span>
                <div>
                  <p className="font-semibold text-emerald-900">{t}</p>
                  <p className="text-sm text-slate-600">{d}</p>
                </div>
              </div>
              {i < steps.length - 1 && <div className="pl-2.5 text-emerald-600" aria-hidden>↓</div>}
            </li>
          ))}
        </ol>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Environmental Suitability Score" subtitle="Rule-based, not ML. Thresholds are assumptions.">
          <p className="text-sm text-slate-600">
            The dataset has no soil, moisture, nutrient or crop-health columns, so a true growth/health score is not possible.
            Instead the prototype scores how close the two climate inputs are to a <b>configured reference range</b>:
          </p>
          <p className="mt-3 rounded-lg bg-slate-50 p-3 font-mono text-sm">
            Score = 100 × (0.5 × TemperatureFit + 0.5 × RainfallFit)
          </p>
          <ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-slate-600">
            <li>Temperature: full marks at 22–30 °C, falling linearly to 0 at 12 °C and 38 °C.</li>
            <li>Rainfall: full marks at 1000–2000 mm/year, falling linearly to 0 at 300 mm and 3500 mm.</li>
            <li>Status: ≥75 Favorable, 50–74 Moderate, &lt;50 Poor.</li>
            <li>Risk: from the score (≥75 Low, 50–74 Medium, &lt;50 High), raised one level if the predicted yield is below that country's historical lower quartile.</li>
          </ul>
          <p className="mt-2 rounded-lg bg-amber-50 p-2 text-xs text-amber-900">
            These ranges are <b>assumptions chosen by the project authors for illustration</b>. They are not universally valid
            agronomic thresholds and were not derived from the dataset.
          </p>
        </Card>

        <Card title="Limitations" subtitle="Please read before interpreting any number">
          <ul className="list-disc space-y-2 pl-5 text-sm text-slate-600">
            <li>Global country-level annual data, not India-specific (India has only {info.country_stats.India?.n_records ?? "few"} records); it cannot guide an individual farm.</li>
            <li>No soil, N-P-K or soil-moisture data. Rainfall is one fixed value per country; pesticides are a national total, not per hectare.</li>
            <li>Historical observational data: predictions show learned patterns, not cause and effect. Changing one input alone can give non-intuitive results.</li>
            <li>Country identity explains a large share of the variation; the country-average baseline is strong.</li>
            <li>Performance is lower on later years (temporal hold-out) than on a random split.</li>
            <li>Recommendations are illustrative only and are not professional agricultural advice.</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}

const Item = ({ k, v }) => (
  <div className="flex flex-col gap-0.5 border-b border-emerald-50 pb-1.5 sm:flex-row sm:gap-3">
    <dt className="w-32 shrink-0 text-slate-500">{k}</dt>
    <dd className="font-medium text-emerald-900">{v}</dd>
  </div>
);
