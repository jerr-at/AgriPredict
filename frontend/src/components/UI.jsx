export function Card({ title, subtitle, children, className = "" }) {
  return (
    <section className={`rounded-2xl bg-white border border-emerald-100 shadow-sm p-5 ${className}`}>
      {title && <h3 className="text-base font-semibold text-emerald-900">{title}</h3>}
      {subtitle && <p className="text-sm text-slate-500 mt-0.5 mb-3">{subtitle}</p>}
      {!subtitle && title && <div className="mb-3" />}
      {children}
    </section>
  );
}

const TONES = {
  green: "bg-emerald-100 text-emerald-800",
  amber: "bg-amber-100 text-amber-800",
  red: "bg-red-100 text-red-800",
  slate: "bg-slate-100 text-slate-600",
};

export function Badge({ tone = "slate", children }) {
  return (
    <span className={`inline-block rounded-full px-3 py-1 text-sm font-semibold ${TONES[tone]}`}>
      {children}
    </span>
  );
}

export const statusTone = (s) => ({ Favorable: "green", Moderate: "amber", Poor: "red" }[s] || "slate");
export const riskTone = (r) => ({ Low: "green", Medium: "amber", High: "red" }[r] || "slate");

export function StatCard({ icon, label, value, sub, badge }) {
  return (
    <div className="rounded-2xl bg-white border border-emerald-100 shadow-sm p-5">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
        <span className="text-lg">{icon}</span>
        {label}
      </div>
      <div className="mt-2 text-3xl font-bold text-emerald-900">{value}</div>
      <div className="mt-1 min-h-6 text-sm text-slate-500">{badge || sub}</div>
    </div>
  );
}

export function Loading({ text = "Loading..." }) {
  return (
    <div className="flex items-center gap-3 p-6 text-slate-500" role="status">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-emerald-600 border-t-transparent" />
      {text}
    </div>
  );
}

export function ErrorBox({ message, onRetry }) {
  return (
    <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-red-800" role="alert">
      <p className="font-medium">Something went wrong</p>
      <p className="text-sm mt-1">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-3 rounded-lg bg-red-700 px-3 py-1.5 text-sm text-white hover:bg-red-800">
          Retry
        </button>
      )}
    </div>
  );
}

/** Small label that tells the viewer where a number comes from (ML model vs rule engine). */
export function Tag({ kind }) {
  const ml = kind === "ml";
  return (
    <span className={`ml-2 inline-block rounded px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide align-middle ${
      ml ? "bg-sky-100 text-sky-800" : "bg-violet-100 text-violet-800"}`}>
      {ml ? "ML model" : "Rule-based, not ML"}
    </span>
  );
}
