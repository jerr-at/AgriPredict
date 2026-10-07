import { useCallback, useEffect, useState } from "react";
import { getModelInfo } from "./api";
import { ErrorBox, Loading } from "./components/UI.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Predict from "./pages/Predict.jsx";
import Analytics from "./pages/Analytics.jsx";
import About from "./pages/About.jsx";

const TABS = [
  ["dashboard", "Dashboard"],
  ["predict", "Predict"],
  ["analytics", "Analytics"],
  ["about", "About"],
];

export default function App() {
  const [tab, setTab] = useState("dashboard");
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [last, setLast] = useState(null); // last prediction + the inputs used

  const load = useCallback(() => {
    setError(null);
    getModelInfo().then(setInfo).catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  return (
    <div className="min-h-screen">
      <header className="bg-gradient-to-r from-emerald-800 to-green-600 text-white">
        <div className="mx-auto max-w-6xl px-4 pt-6 pb-4">
          <h1 className="text-3xl font-bold tracking-tight">🌾 AgriPredict</h1>
          <p className="text-emerald-100">AI-Based Rice Yield &amp; Agricultural Decision Support System</p>
          <nav className="mt-5 flex gap-1 overflow-x-auto" aria-label="Main">
            {TABS.map(([id, label]) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={`rounded-t-lg px-4 py-2 text-sm font-medium transition ${
                  tab === id ? "bg-[#f4f8f3] text-emerald-900" : "text-emerald-50 hover:bg-white/10"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        {error && <ErrorBox message={error} onRetry={load} />}
        {!error && !info && <Loading text="Connecting to the AgriPredict API..." />}
        {info && tab === "dashboard" && <Dashboard info={info} last={last} go={setTab} />}
        {info && tab === "predict" && <Predict info={info} last={last} setLast={setLast} />}
        {info && tab === "analytics" && <Analytics info={info} />}
        {info && tab === "about" && <About info={info} />}
      </main>

      <footer className="mx-auto max-w-6xl px-4 pb-8 text-xs text-slate-500">
        Academic prototype. Predictions come from a machine-learning model trained on historical country-level data; the suitability score is a rule-based illustration. Neither replaces professional agricultural guidance.
      </footer>
    </div>
  );
}
