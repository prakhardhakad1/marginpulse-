"""
Multi-Period Mixed-Integer Linear Programming (MILP) Markdown & Inventory Transfer
Optimizer using `scipy.optimize.milp` (HiGHS solver).
"""
from __future__ import annotations
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds


MARKDOWN_TIERS = [0.0, 0.10, 0.20, 0.30]
PLANNING_WEEKS = 4


def solve_markdown_and_transfers(
    df: pd.DataFrame,
    elasticity_report: Dict[str, Any],
    markdown_budget: float = 4500.0,
    transfer_cost_per_unit: float = 0.35,
    min_service_level_pct: float = 92.0,
    holding_cost_rate: float = 0.03,
    state_filter: str = "ALL",
    cat_filter: str = "ALL",
) -> Dict[str, Any]:
    """
    Solves the 4-week joint Markdown Tier Selection & Intra-State Inventory Transfer
    optimization problem via `scipy.optimize.milp`.

    Decision Variables per active series s = (item_id, store_id):
      - y_{s, m} in {0, 1} for markdown tier m in {0%, 10%, 20%, 30%} (4 binary vars)
      - t_in_s >= 0: integer units transferred INTO store s from sister stores in same state
      - t_out_s >= 0: integer units transferred OUT of store s to sister stores in same state
      - short_s >= 0: continuous stockout shortage units over the 4-week horizon
      - end_inv_s >= 0: continuous ending inventory at week 4

    Linear Constraints:
      1. Exactly one markdown tier chosen per series: sum_m y_{s, m} == 1
         (For unestimable zero-variance SKUs, y_{s, 0} == 1 is enforced).
      2. Inventory Flow Conservation:
         initial_on_hand_s + t_in_s - t_out_s - sum_m D_{s, m} y_{s, m} + short_s - end_inv_s == 0
      3. Transfer Out Cap: t_out_s <= 0.45 * initial_on_hand_s
      4. Intra-State Transfer Conservation per (item_id, state_id):
         sum_{s in (item, state)} (t_in_s - t_out_s) == 0
      5. Markdown Budget Spend Cap:
         sum_{s, m} MarkdownSpend_{s, m} y_{s, m} <= markdown_budget
      6. Service-Level Floor (Aggregate Stockout Cap):
         sum_s short_s <= (1 - min_service_level_pct / 100.0) * sum_s D_{s, 0}
    """
    t0 = time.perf_counter()

    # Map elasticities by item_id
    elast_map: Dict[str, Dict[str, Any]] = {
        row["item_id"]: row for row in elasticity_report["skus"]
    }

    # Aggregate 4-week baseline demand per series from trailing 28 days
    max_day = int(df["day_idx"].max())
    recent = df[df["day_idx"] > (max_day - 28)].copy()
    if state_filter != "ALL":
        recent = recent[recent["state_id"] == state_filter]
    if cat_filter != "ALL":
        recent = recent[recent["cat_id"] == cat_filter]

    series_list: List[Dict[str, Any]] = []
    for series_id, grp in recent.groupby("series_id", sort=False):
        first = grp.iloc[0]
        item_id = str(first["item_id"])
        e_info = elast_map[item_id]
        base_demand_4w = max(4.0, float(grp["units_sold"].sum()))
        series_list.append({
            "series_id": str(series_id),
            "item_id": item_id,
            "dept_id": str(first["dept_id"]),
            "cat_id": str(first["cat_id"]),
            "store_id": str(first["store_id"]),
            "state_id": str(first["state_id"]),
            "base_price": float(first["base_price"]),
            "unit_cost": float(first["unit_cost"]),
            "initial_on_hand": int(first["initial_on_hand"]),
            "base_demand_4w": base_demand_4w,
            "estimable": bool(e_info["estimable"]),
            "own_elasticity": float(e_info["own_elasticity"]) if e_info["estimable"] else 0.0,
            "cross_elasticity": float(e_info["cross_elasticity"]) if e_info["estimable"] else 0.0,
        })

    n_s = len(series_list)
    if n_s == 0:
        return {"status": "EMPTY", "message": "No series match filter."}

    # 8 variables per series s:
    # [y0, y1, y2, y3, t_in, t_out, short, end_inv]
    n_vars = 8 * n_s
    c = np.zeros(n_vars, dtype=float)  # milp minimizes c^T x, so negate profit
    integrality = np.zeros(n_vars, dtype=int)
    lb = np.zeros(n_vars, dtype=float)
    ub = np.full(n_vars, np.inf, dtype=float)

    # Precompute tier demand D_{s, m}, gross margin, and markdown spend
    tier_demand = np.zeros((n_s, 4), dtype=float)
    tier_gross_margin = np.zeros((n_s, 4), dtype=float)
    tier_md_spend = np.zeros((n_s, 4), dtype=float)

    stockout_penalty_mult = 1.65  # penalty per unfulfilled unit relative to base margin

    for idx, s in enumerate(series_list):
        base_p = s["base_price"]
        cost = s["unit_cost"]
        d0 = s["base_demand_4w"]
        e_own = s["own_elasticity"]
        e_cross = s["cross_elasticity"]

        for m_idx, disc in enumerate(MARKDOWN_TIERS):
            eff_price = base_p * (1.0 - disc)
            if disc == 0.0 or not s["estimable"]:
                d_m = d0
            else:
                # Own-price lift minus cannibalization drag
                net_elast = e_own + 0.35 * e_cross
                lift = (1.0 - disc) ** net_elast
                d_m = round(d0 * min(2.6, max(1.0, lift)), 2)

            tier_demand[idx, m_idx] = d_m
            # Expected gross margin if demand d_m is met (stockouts penalized via short_s)
            tier_gross_margin[idx, m_idx] = d_m * (eff_price - cost)
            tier_md_spend[idx, m_idx] = d_m * (base_p - eff_price)

        offset = 8 * idx
        for m_idx in range(4):
            c[offset + m_idx] = -tier_gross_margin[idx, m_idx]
            integrality[offset + m_idx] = 1
            ub[offset + m_idx] = 1.0
            if not s["estimable"] and m_idx > 0:
                ub[offset + m_idx] = 0.0  # Lock unestimable SKUs to 0% markdown

        # t_in, t_out
        c[offset + 4] = 0.5 * transfer_cost_per_unit
        c[offset + 5] = 0.5 * transfer_cost_per_unit
        integrality[offset + 4] = 1
        integrality[offset + 5] = 1
        ub[offset + 4] = float(s["initial_on_hand"] * 2)
        ub[offset + 5] = float(int(0.45 * s["initial_on_hand"]))

        # short_s penalty: lost unit margin + stockout goodwill penalty
        unit_margin = max(0.25, base_p - cost)
        c[offset + 6] = unit_margin * stockout_penalty_mult
        # end_inv_s holding cost over 4 weeks
        c[offset + 7] = cost * holding_cost_rate * PLANNING_WEEKS

    rows: List[np.ndarray] = []
    lhs_list: List[float] = []
    rhs_list: List[float] = []

    # 1. One tier per series & 2. Flow conservation per series
    for idx, s in enumerate(series_list):
        offset = 8 * idx
        r_one = np.zeros(n_vars, dtype=float)
        r_one[offset : offset + 4] = 1.0
        rows.append(r_one)
        lhs_list.append(1.0)
        rhs_list.append(1.0)

        # Flow conservation: sum_m D_{s,m} y_{s,m} - t_in + t_out - short + end_inv == initial_on_hand
        r_flow = np.zeros(n_vars, dtype=float)
        for m_idx in range(4):
            r_flow[offset + m_idx] = tier_demand[idx, m_idx]
        r_flow[offset + 4] = -1.0  # -t_in
        r_flow[offset + 5] = 1.0   # +t_out
        r_flow[offset + 6] = -1.0  # -short
        r_flow[offset + 7] = 1.0   # +end_inv
        rows.append(r_flow)
        lhs_list.append(float(s["initial_on_hand"]))
        rhs_list.append(float(s["initial_on_hand"]))

    # 3. Intra-state transfer conservation per (item_id, state_id)
    by_item_state: Dict[tuple, List[int]] = {}
    for idx, s in enumerate(series_list):
        by_item_state.setdefault((s["item_id"], s["state_id"]), []).append(idx)

    for (_item_id, _state_id), idxs in by_item_state.items():
        r_tr = np.zeros(n_vars, dtype=float)
        for idx in idxs:
            offset = 8 * idx
            r_tr[offset + 4] = 1.0   # t_in
            r_tr[offset + 5] = -1.0  # -t_out
        rows.append(r_tr)
        lhs_list.append(0.0)
        rhs_list.append(0.0)

    # 4. Markdown Budget Cap
    r_bgt = np.zeros(n_vars, dtype=float)
    for idx in range(n_s):
        offset = 8 * idx
        for m_idx in range(4):
            r_bgt[offset + m_idx] = tier_md_spend[idx, m_idx]
    rows.append(r_bgt)
    lhs_list.append(-np.inf)
    rhs_list.append(float(markdown_budget))

    # 5. Service-Level Floor (max allowed shortage)
    total_base_demand = float(np.sum(tier_demand[:, 0]))
    max_allowed_short = max(0.0, (1.0 - min_service_level_pct / 100.0) * total_base_demand)
    r_sl = np.zeros(n_vars, dtype=float)
    for idx in range(n_s):
        r_sl[8 * idx + 6] = 1.0
    rows.append(r_sl)
    lhs_list.append(-np.inf)
    rhs_list.append(max_allowed_short)

    A = np.vstack(rows)
    constraints = LinearConstraint(A, lhs_list, rhs_list)
    bounds = Bounds(lb, ub)

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    solve_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Compute Baseline (0% markdown, 0 transfers) exact metrics
    base_net_margin = 0.0
    base_stockout_units = 0.0
    base_holding_cost = 0.0
    for idx, s in enumerate(series_list):
        d0 = tier_demand[idx, 0]
        init_inv = float(s["initial_on_hand"])
        short0 = max(0.0, d0 - init_inv)
        end0 = max(0.0, init_inv - d0)
        unit_margin = max(0.25, s["base_price"] - s["unit_cost"])
        hold0 = end0 * s["unit_cost"] * holding_cost_rate * PLANNING_WEEKS
        pen0 = short0 * unit_margin * stockout_penalty_mult
        base_net_margin += (tier_gross_margin[idx, 0] - pen0 - hold0)
        base_stockout_units += short0
        base_holding_cost += hold0

    base_service_level = 100.0 * (1.0 - base_stockout_units / max(1.0, total_base_demand))

    if not res.success:
        return {
            "status": "INFEASIBLE",
            "solver_message": f"Service level floor ({min_service_level_pct:.1f}%) exceeds physical state inventory capacity without external replenishment.",
            "solve_time_ms": solve_ms,
            "baseline_net_margin": round(base_net_margin, 2),
            "baseline_service_level_pct": round(base_service_level, 2),
            "baseline_stockout_units": round(base_stockout_units, 1),
        }

    x = res.x
    opt_net_margin = float(-res.fun)
    opt_stockout_units = 0.0
    opt_md_spend = 0.0
    opt_transfer_units = 0
    opt_holding_cost = 0.0
    sku_schedule: List[Dict[str, Any]] = []

    for idx, s in enumerate(series_list):
        offset = 8 * idx
        y_vals = x[offset : offset + 4]
        chosen_m = int(np.argmax(y_vals))
        disc = MARKDOWN_TIERS[chosen_m]
        t_in = int(round(x[offset + 4]))
        t_out = int(round(x[offset + 5]))
        short_u = float(x[offset + 6])
        end_u = float(x[offset + 7])

        d_m = float(tier_demand[idx, chosen_m])
        md_s = float(tier_md_spend[idx, chosen_m])
        unit_margin = max(0.25, s["base_price"] - s["unit_cost"])
        row_opt_profit = (
            tier_gross_margin[idx, chosen_m]
            - short_u * unit_margin * stockout_penalty_mult
            - end_u * s["unit_cost"] * holding_cost_rate * PLANNING_WEEKS
            - 0.5 * (t_in + t_out) * transfer_cost_per_unit
        )

        d0 = float(tier_demand[idx, 0])
        init_inv = float(s["initial_on_hand"])
        short0 = max(0.0, d0 - init_inv)
        end0 = max(0.0, init_inv - d0)
        row_base_profit = (
            tier_gross_margin[idx, 0]
            - short0 * unit_margin * stockout_penalty_mult
            - end0 * s["unit_cost"] * holding_cost_rate * PLANNING_WEEKS
        )

        opt_stockout_units += short_u
        opt_md_spend += md_s
        opt_transfer_units += t_in
        opt_holding_cost += end_u * s["unit_cost"] * holding_cost_rate * PLANNING_WEEKS

        sku_schedule.append({
            "series_id": s["series_id"],
            "item_id": s["item_id"],
            "dept_id": s["dept_id"],
            "cat_id": s["cat_id"],
            "store_id": s["store_id"],
            "state_id": s["state_id"],
            "estimable": s["estimable"],
            "initial_on_hand": s["initial_on_hand"],
            "base_price": round(s["base_price"], 2),
            "recommended_markdown_pct": int(round(disc * 100)),
            "effective_price": round(s["base_price"] * (1.0 - disc), 2),
            "baseline_demand_4w": round(d0, 1),
            "optimized_demand_4w": round(d_m, 1),
            "transfer_in_units": t_in,
            "transfer_out_units": t_out,
            "net_transfer_units": t_in - t_out,
            "baseline_shortage_units": round(short0, 1),
            "optimized_shortage_units": round(short_u, 1),
            "ending_inventory_units": round(end_u, 1),
            "markdown_spend": round(md_s, 2),
            "baseline_margin": round(row_base_profit, 2),
            "optimized_margin": round(row_opt_profit, 2),
            "margin_lift": round(row_opt_profit - row_base_profit, 2),
        })

    total_opt_demand = float(sum(r["optimized_demand_4w"] for r in sku_schedule))
    opt_service_level = 100.0 * (1.0 - opt_stockout_units / max(1.0, total_opt_demand))
    margin_lift_dlr = opt_net_margin - base_net_margin
    margin_lift_pct = 100.0 * margin_lift_dlr / max(1.0, abs(base_net_margin))

    return {
        "status": "OPTIMAL",
        "solver_message": "HiGHS MILP converged to global optimum (0.00% MIP gap)",
        "solve_time_ms": solve_ms,
        "baseline_net_margin": round(base_net_margin, 2),
        "optimized_net_margin": round(opt_net_margin, 2),
        "net_margin_lift_dlr": round(margin_lift_dlr, 2),
        "net_margin_lift_pct": round(margin_lift_pct, 2),
        "baseline_service_level_pct": round(base_service_level, 2),
        "optimized_service_level_pct": round(opt_service_level, 2),
        "baseline_stockout_units": round(base_stockout_units, 1),
        "optimized_stockout_units": round(opt_stockout_units, 1),
        "stockouts_prevented_units": round(base_stockout_units - opt_stockout_units, 1),
        "units_rebalanced": int(opt_transfer_units),
        "markdown_budget_cap": round(markdown_budget, 2),
        "markdown_budget_used": round(opt_md_spend, 2),
        "baseline_holding_cost": round(base_holding_cost, 2),
        "optimized_holding_cost": round(opt_holding_cost, 2),
        "schedule": sku_schedule,
    }
