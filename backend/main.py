"""
FastAPI Production Backend for MarginPulse: Dynamic Markdown & Multi-Store Inventory Allocator.
"""
from __future__ import annotations
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ingest import generate_m5_panel
from backend.engine.audit import run_forensic_audit
from backend.engine.elasticity import fit_sku_elasticities
from backend.engine.optimizer import solve_markdown_and_transfers

app = FastAPI(
    title="MarginPulse M5 Dynamic Markdown & Inventory Allocator API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PANEL_DF = generate_m5_panel()
AUDIT_CACHE = run_forensic_audit(PANEL_DF)
ELASTICITY_CACHE = fit_sku_elasticities(PANEL_DF, n_bootstrap=100, holdout_days=28)


class OptimizeRequest(BaseModel):
    markdown_budget: float = Field(4500.0, ge=0.0, le=50000.0)
    transfer_cost_per_unit: float = Field(0.35, ge=0.0, le=10.0)
    min_service_level_pct: float = Field(92.0, ge=50.0, le=99.9)
    holding_cost_rate: float = Field(0.03, ge=0.0, le=0.25)
    state_filter: str = Field("ALL")
    cat_filter: str = Field("ALL")


@app.get("/api/audit")
def get_audit():
    return AUDIT_CACHE


@app.get("/api/elasticity/{dept_id}")
def get_elasticity(dept_id: str = "ALL"):
    if dept_id == "ALL":
        return ELASTICITY_CACHE
    filtered = [s for s in ELASTICITY_CACHE["skus"] if s["dept_id"] == dept_id or s["cat_id"] == dept_id]
    return {**ELASTICITY_CACHE, "skus": filtered}


@app.post("/api/optimize")
def post_optimize(req: OptimizeRequest):
    return solve_markdown_and_transfers(
        df=PANEL_DF,
        elasticity_report=ELASTICITY_CACHE,
        markdown_budget=req.markdown_budget,
        transfer_cost_per_unit=req.transfer_cost_per_unit,
        min_service_level_pct=req.min_service_level_pct,
        holding_cost_rate=req.holding_cost_rate,
        state_filter=req.state_filter,
        cat_filter=req.cat_filter,
    )
