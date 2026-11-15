"""
Automated Unit Tests for MarginPulse Forensic Audit, Log-Log Elasticity, and MILP Solver.
Verifies inventory flow conservation, non-negative stock, budget adherence,
intra-state transfer balance, and unestimable zero-variance SKU guards.
"""
import numpy as np
from backend.ingest import generate_m5_panel
from backend.engine.audit import run_forensic_audit
from backend.engine.elasticity import fit_sku_elasticities
from backend.engine.optimizer import solve_markdown_and_transfers


def test_forensic_audit_and_zero_variance_guard():
    df = generate_m5_panel()
    audit = run_forensic_audit(df)
    assert audit["total_series"] == 120
    assert audit["zero_variance_price_series"] == 20  # 2 zero-var SKUs x 10 stores
    assert sum(audit["regime_counts"].values()) == 120

    elast = fit_sku_elasticities(df, n_bootstrap=50, holdout_days=28)
    unestimable = [s for s in elast["skus"] if not s["estimable"]]
    assert len(unestimable) == 2
    for u in unestimable:
        assert u["own_elasticity"] is None


def test_milp_inventory_conservation_and_budget_adherence():
    df = generate_m5_panel()
    elast = fit_sku_elasticities(df, n_bootstrap=40, holdout_days=28)
    budget_cap = 3500.0
    sol = solve_markdown_and_transfers(
        df=df,
        elasticity_report=elast,
        markdown_budget=budget_cap,
        transfer_cost_per_unit=0.35,
        min_service_level_pct=90.0,
        holding_cost_rate=0.03,
    )
    assert sol["status"] == "OPTIMAL"
    assert sol["markdown_budget_used"] <= budget_cap + 1e-4
    assert sol["optimized_net_margin"] >= sol["baseline_net_margin"]

    # Check every series satisfies flow conservation & non-negativity
    by_item_state = {}
    for row in sol["schedule"]:
        lhs = (
            row["initial_on_hand"]
            + row["transfer_in_units"]
            - row["transfer_out_units"]
            - row["optimized_demand_4w"]
            + row["optimized_shortage_units"]
            - row["ending_inventory_units"]
        )
        assert abs(lhs) < 0.25
        assert row["ending_inventory_units"] >= -1e-5
        assert row["optimized_shortage_units"] >= -1e-5
        if not row["estimable"]:
            assert row["recommended_markdown_pct"] == 0

        key = (row["item_id"], row["state_id"])
        by_item_state[key] = by_item_state.get(key, 0) + row["net_transfer_units"]

    # Intra-state transfer balance must sum to zero for every (item_id, state_id)
    for key, net_sum in by_item_state.items():
        assert net_sum == 0, f"Transfer imbalance for {key}: {net_sum}"
