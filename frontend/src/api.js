const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options) {
  let res;
  try {
    res = await fetch(API + path, options);
  } catch {
    throw new Error(`Cannot reach the backend at ${API}. Start it with: uvicorn app.main:app --port 8000`);
  }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = typeof j.detail === "string"
        ? j.detail
        : (j.detail || []).map((d) => `${d.loc.at(-1)}: ${d.msg}`).join("; ");
    } catch {
      /* keep statusText */
    }
    throw new Error(msg || `Request failed (${res.status})`);
  }
  return res.json();
}

export const getModelInfo = () => request("/model-info");
export const getAnalytics = () => request("/analytics");
export const predictYield = (body) =>
  request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
