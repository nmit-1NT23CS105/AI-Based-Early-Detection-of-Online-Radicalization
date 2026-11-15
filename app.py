from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import fetch_alerts, fetch_dashboard_summary, fetch_history, init_db, save_analysis
from model import RadicalizationModelManager
from predict import PredictionService
from schemas import BatchRequest, TextRequest


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Online Radicalization Detection System", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

init_db()
model_manager = RadicalizationModelManager(BASE_DIR)
prediction_service = PredictionService(BASE_DIR)


@app.on_event("startup")
def startup_event() -> None:
    init_db()
    model_manager.ensure_model()
    prediction_service.reload()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    context = {
        "request": request,
        "page_title": "Online Radicalization Detection System",
        "summary": fetch_dashboard_summary(),
        "history": fetch_history(limit=12),
        "alerts": fetch_alerts(limit=8),
        "metrics": prediction_service.get_dashboard_metrics(),
        "metadata": prediction_service.get_model_metadata(),
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/api/analyze")
async def analyze_text(payload: TextRequest) -> JSONResponse:
    try:
        result = prediction_service.predict_one(payload.text)
        stored_record = save_analysis(result)
        response = {
            "analysis": result,
            "stored_record": stored_record,
            "dashboard": fetch_dashboard_summary(),
        }
        return JSONResponse(response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/analyze-batch")
async def analyze_batch(payload: BatchRequest) -> JSONResponse:
    try:
        results = prediction_service.predict_batch(payload.texts)
        stored = [save_analysis(item) for item in results]
        average_risk = round(sum(item["risk"]["score"] for item in results) / len(results), 2)
        return JSONResponse(
            {
                "results": results,
                "stored_records": stored,
                "summary": {
                    "items": len(results),
                    "average_risk": average_risk,
                    "high_risk": sum(1 for item in results if item["risk"]["level"] == "High Risk"),
                },
                "dashboard": fetch_dashboard_summary(),
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/history")
async def get_history() -> JSONResponse:
    return JSONResponse({"history": fetch_history(limit=30)})


@app.get("/api/dashboard")
async def dashboard() -> JSONResponse:
    return JSONResponse(
        {
            "summary": fetch_dashboard_summary(),
            "metrics": prediction_service.get_dashboard_metrics(),
            "metadata": prediction_service.get_model_metadata(),
        }
    )


@app.post("/api/retrain")
async def retrain_model() -> JSONResponse:
    result = model_manager.train_from_csv()
    prediction_service.reload()
    return JSONResponse(
        {
            "message": "Model retraining completed on the academic demo dataset.",
            "metrics": result["metrics"],
            "metadata": result["metadata"],
        }
    )


@app.get("/api/alerts/stream")
async def stream_alerts() -> StreamingResponse:
    async def event_stream():
        while True:
            payload = {"alerts": fetch_alerts(limit=8)}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(6)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
