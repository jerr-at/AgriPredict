import {
  Bar, BarChart, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";

export const GREEN = "#2f7d32";
export const BROWN = "#8d6e63";
export const BASELINE = "Country-average baseline";

// Short, readable names (with units) for chart axes. Keys are the model's feature names.
export const SHORT = {
  Area: "Country",
  Year: "Year",
  rainfall_mm: "Rainfall (mm)",
  pesticides_tonnes: "Pesticides (t/yr)",
  avg_temp: "Temperature (°C)",
  yield_t_ha: "Yield (t/ha)",
};
export const short = (k) => SHORT[k] || k;

const grid = <CartesianGrid strokeDasharray="3 3" stroke="#e5ece4" />;

/** Actual (x) vs predicted (y) for held-out records; dashed line = perfect prediction. */
export function ActualVsPredicted({ data, height = "h-72" }) {
  const maxV = Math.ceil(Math.max(...data.flatMap((d) => [d.actual, d.predicted])));
  return (
    <div>
      <div className="mb-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
        <span><span className="mr-1 inline-block h-2.5 w-2.5 rounded-full bg-[#2f7d32] opacity-60" />Held-out records (actual vs predicted)</span>
        <span><span className="mr-1 inline-block w-4 border-t-2 border-dashed border-slate-700 align-middle" />Perfect prediction (predicted = actual)</span>
      </div>
    <div className={height}>
      <ResponsiveContainer>
        <ComposedChart margin={{ top: 8, right: 15, left: 15, bottom: 25 }}>
          {grid}
          <XAxis type="number" dataKey="actual" domain={[0, maxV]} name="Actual" unit=" t/ha"
            label={{ value: "Actual yield (t/ha)", position: "insideBottom", offset: -12 }} />
          <YAxis type="number" dataKey="predicted" domain={[0, maxV]} name="Predicted" unit=" t/ha"
            label={{ value: "Predicted yield (t/ha)", angle: -90, position: "insideLeft", offset: -2, style: { textAnchor: "middle" } }} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter name="Held-out records" data={data} fill={GREEN} fillOpacity={0.55} />
          <Line name="Perfect prediction (predicted = actual)" data={[{ actual: 0, predicted: 0 }, { actual: maxV, predicted: maxV }]}
            dataKey="predicted" stroke="#333" strokeDasharray="5 5" dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
    </div>
  );
}

/** Permutation (predictive) importance: drop in R² when a feature's information is shuffled. */
export function FeatureImportance({ data, height = "h-64" }) {
  const rows = data.map((d) => ({ label: short(d.feature), importance: +d.importance.toFixed(3) }));
  return (
    <div className={height}>
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ top: 5, right: 20, left: 10, bottom: 25 }}>
          {grid}
          <XAxis type="number" label={{ value: "Drop in R² when shuffled", position: "insideBottom", offset: -12 }} />
          <YAxis type="category" dataKey="label" width={115} tick={{ fontSize: 12 }} />
          <Tooltip />
          <Bar dataKey="importance" name="Drop in R²" fill={GREEN} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function YieldHistogram({ data, height = "h-64" }) {
  return (
    <div className={height}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 5, right: 10, left: 15, bottom: 45 }}>
          {grid}
          <XAxis dataKey="bin" angle={-40} textAnchor="end" interval={1} height={60} tick={{ fontSize: 10 }}
            label={{ value: "Yield bin (tonnes/hectare)", position: "insideBottom", offset: -38 }} />
          <YAxis label={{ value: "Records", angle: -90, position: "insideLeft", offset: -2, style: { textAnchor: "middle" } }} />
          <Tooltip />
          <Bar dataKey="count" name="Records" fill={GREEN} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Grouped bars: random vs temporal hold-out for every model + the baseline. metric = "R2" | "RMSE". */
export function ModelComparisonChart({ models, metric }) {
  const rows = models.map((m) => ({
    name: m.model === BASELINE ? "Baseline" : m.model,
    random: +m.random_holdout[metric].toFixed(3),
    temporal: +m.temporal_holdout[metric].toFixed(3),
  }));
  return (
    <div className="h-72">
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 5, right: 10, left: 15, bottom: 45 }}>
          {grid}
          <XAxis dataKey="name" angle={-25} textAnchor="end" interval={0} height={70} tick={{ fontSize: 11 }} />
          <YAxis label={{ value: metric === "R2" ? "R²" : "RMSE (t/ha)", angle: -90, position: "insideLeft", offset: 5 }} />
          <Tooltip />
          <Legend verticalAlign="top" height={28} />
          <Bar dataKey="random" name="Random hold-out" fill={GREEN} />
          <Bar dataKey="temporal" name="Temporal hold-out (2009–2013)" fill={BROWN} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Heatmap({ labels, matrix }) {
  const n = labels.length;
  return (
    <div className="overflow-x-auto">
      <div className="inline-grid gap-1 text-sm" style={{ gridTemplateColumns: `120px repeat(${n}, 88px)` }}>
        <div />
        {labels.map((l) => <div key={l} className="text-center text-xs font-medium text-slate-600">{short(l)}</div>)}
        {matrix.map((row, i) => (
          <HeatRow key={labels[i]} label={short(labels[i])} row={row} />
        ))}
      </div>
    </div>
  );
}

function HeatRow({ label, row }) {
  return (
    <>
      <div className="flex items-center text-xs font-medium text-slate-600">{label}</div>
      {row.map((v, j) => (
        <div key={j} className="rounded-md py-3 text-center font-medium"
          style={{
            background: v >= 0 ? `rgba(47,125,50,${Math.abs(v) * 0.85})` : `rgba(217,119,6,${Math.abs(v) * 0.85})`,
            color: Math.abs(v) > 0.5 ? "#fff" : "#1f2a1f",
          }}>
          {v.toFixed(2)}
        </div>
      ))}
    </>
  );
}
