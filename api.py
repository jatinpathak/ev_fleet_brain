"""Thin REST API over the engines — the 'integration-ready' story.

A small FastAPI app exposing a few endpoints so the platform can be wired into
an ERP/SAP/telematics stack without the Streamlit UI. Deliberately minimal
(per the guardrails: cheap credible integration, not enterprise infra).

Run:
    uvicorn api:app --reload --port 8000
    # interactive docs at http://localhost:8000/docs

Endpoints:
    GET  /                 health/info
    GET  /fleet/summary    readiness + carbon fleet summary
    POST /battery/predict  {"cell_id": "CELL_010"} -> health + recommendation
    POST /scenario/run     {"name": "...", "params": {...}} -> scenario deltas
"""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

import config

try:
    from fastapi import FastAPI, HTTPException
except Exception as exc:  # pragma: no cover - fastapi optional at import time
    raise SystemExit("FastAPI is required for the REST layer: pip install fastapi uvicorn") from exc

app = FastAPI(title="EV Fleet Intelligence Brain API", version="3.0")


class BatteryRequest(BaseModel):
    cell_id: str


class ScenarioRequest(BaseModel):
    name: str
    params: dict = {}


@app.get("/")
def root() -> dict:
    return {
        "service": "EV Fleet Intelligence Brain",
        "version": "3.0",
        "endpoints": ["/fleet/summary", "/battery/predict", "/scenario/run", "/docs"],
        "note": "Synthetic/illustrative demo data; battery model shaped on real Severson cycling.",
    }


@app.get("/fleet/summary")
def fleet_summary() -> dict:
    from engines import engine_readiness as er
    from engines import engine_carbon as ec
    scored = er.score_fleet()
    return {
        "readiness": er.fleet_summary(scored),
        "carbon": ec.fleet_carbon_summary(),
        "recommendation": er.recommendation(scored).as_dict(),
    }


@app.post("/battery/predict")
def battery_predict(req: BatteryRequest) -> dict:
    from engines import engine_battery as eb
    df = pd.read_csv(config.BATTERY_DATA_CSV)
    hist = df[df["cell_id"] == req.cell_id]
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"Unknown cell_id: {req.cell_id}")
    health = eb.predict_health(hist)
    return {"health": health,
            "recommendation": eb.recommendation(health).as_dict(),
            "passport": eb.battery_passport(req.cell_id, df)}


@app.post("/scenario/run")
def scenario_run(req: ScenarioRequest) -> dict:
    from engines import engine_scenario as es
    try:
        return es.run(req.name, **(req.params or {}))
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=f"Bad params: {exc}")
