from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .predictor import ModelNotReady, Predictor
from .schemas import PredictRequest, PredictResponse, RecommendationResponse

app = FastAPI(title="AgriPredict API", version="1.1.0",
              description="Rice yield prediction (ML) and rule-based decision support. Academic prototype.")

# Prototype: allow the Vite dev server (and anything else) to call the API.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """422 with only location/message/type. FastAPI's default body echoes the offending input,
    which cannot be JSON-encoded for NaN/Infinity and used to crash with HTTP 500."""
    errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


try:
    predictor = Predictor()
    load_error = None
except ModelNotReady as e:
    predictor = None
    load_error = str(e)


def _need_model() -> Predictor:
    if predictor is None:
        raise HTTPException(status_code=503, detail=load_error)
    return predictor


@app.get("/health")
def health():
    """200 only when the API is up AND the model is loaded; 503 otherwise."""
    p = _need_model()
    return {"status": "ok", "model": p.meta["best_model"], "features": p.meta["features"]}


@app.get("/model-info")
def model_info():
    p = _need_model()
    m = p.meta
    return {k: m[k] for k in (
        "crop", "target", "target_description", "unit", "unit_conversion", "features", "feature_labels",
        "numeric_features", "categorical_features", "countries", "numeric_ranges", "country_stats",
        "best_model", "selection_method", "best_model_metrics", "best_model_temporal_metrics", "all_models",
        "baseline_country_mean", "baseline_country_mean_temporal", "n_records", "n_train", "n_test",
        "temporal_split", "training", "cleaning_log", "dataset",
    )}


@app.get("/analytics")
def analytics():
    return _need_model().analytics


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return _need_model().predict(req.country, req.year, req.rainfall_mm, req.pesticides_tonnes, req.avg_temp)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/recommendation", response_model=RecommendationResponse)
def recommendation(req: PredictRequest):
    try:
        return _need_model().recommend(req.country, req.year, req.rainfall_mm, req.pesticides_tonnes, req.avg_temp)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
