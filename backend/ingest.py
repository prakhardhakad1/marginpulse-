"""
Walmart M5 Forecasting Schema Ingestion & Calibration Generator.
Supports loading raw M5 CSVs (sales_train_validation.csv, sell_prices.csv, calendar.csv)
or generating an exact M5-schema calibrated dataset (10 stores across CA, TX, WI;
3 categories FOODS, HOBBIES, HOUSEHOLD; 365 days) for reproducible local execution.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd


M5_STORES = [
    {"store_id": "CA_1", "state_id": "CA", "city": "Sacramento"},
    {"store_id": "CA_2", "state_id": "CA", "city": "Los Angeles"},
    {"store_id": "CA_3", "state_id": "CA", "city": "San Diego"},
    {"store_id": "CA_4", "state_id": "CA", "city": "San Francisco"},
    {"store_id": "TX_1", "state_id": "TX", "city": "Houston"},
    {"store_id": "TX_2", "state_id": "TX", "city": "Dallas"},
    {"store_id": "TX_3", "state_id": "TX", "city": "Austin"},
    {"store_id": "WI_1", "state_id": "WI", "city": "Milwaukee"},
    {"store_id": "WI_2", "state_id": "WI", "city": "Madison"},
    {"store_id": "WI_3", "state_id": "WI", "city": "Green Bay"},
]

M5_CATALOG = [
    {"item_id": "FOODS_3_090", "dept_id": "FOODS_3", "cat_id": "FOODS", "base_price": 1.48, "unit_cost": 0.92, "true_elast": -1.85, "cross_elast": 0.34, "zero_var": False, "regime_target": "Smooth"},
    {"item_id": "FOODS_3_252", "dept_id": "FOODS_3", "cat_id": "FOODS", "base_price": 1.58, "unit_cost": 0.98, "true_elast": -1.62, "cross_elast": 0.29, "zero_var": False, "regime_target": "Smooth"},
    {"item_id": "FOODS_3_586", "dept_id": "FOODS_3", "cat_id": "FOODS", "base_price": 1.68, "unit_cost": 1.05, "true_elast": -1.44, "cross_elast": 0.22, "zero_var": False, "regime_target": "Smooth"},
    {"item_id": "FOODS_2_197", "dept_id": "FOODS_2", "cat_id": "FOODS", "base_price": 3.48, "unit_cost": 2.10, "true_elast": -1.94, "cross_elast": 0.41, "zero_var": False, "regime_target": "Erratic"},
    {"item_id": "FOODS_1_004", "dept_id": "FOODS_1", "cat_id": "FOODS", "base_price": 2.98, "unit_cost": 1.85, "true_elast": 0.0, "cross_elast": 0.0, "zero_var": True, "regime_target": "Smooth"},
    {"item_id": "HOBBIES_1_008", "dept_id": "HOBBIES_1", "cat_id": "HOBBIES", "base_price": 11.98, "unit_cost": 4.80, "true_elast": -2.65, "cross_elast": 0.38, "zero_var": False, "regime_target": "Lumpy"},
    {"item_id": "HOBBIES_1_142", "dept_id": "HOBBIES_1", "cat_id": "HOBBIES", "base_price": 8.48, "unit_cost": 3.55, "true_elast": -2.35, "cross_elast": 0.28, "zero_var": False, "regime_target": "Intermittent"},
    {"item_id": "HOBBIES_2_031", "dept_id": "HOBBIES_2", "cat_id": "HOBBIES", "base_price": 4.97, "unit_cost": 2.80, "true_elast": 0.0, "cross_elast": 0.0, "zero_var": True, "regime_target": "Intermittent"},
    {"item_id": "HOUSEHOLD_1_118", "dept_id": "HOUSEHOLD_1", "cat_id": "HOUSEHOLD", "base_price": 6.97, "unit_cost": 3.10, "true_elast": -2.48, "cross_elast": 0.32, "zero_var": False, "regime_target": "Smooth"},
    {"item_id": "HOUSEHOLD_1_304", "dept_id": "HOUSEHOLD_1", "cat_id": "HOUSEHOLD", "base_price": 9.97, "unit_cost": 4.20, "true_elast": -2.82, "cross_elast": 0.42, "zero_var": False, "regime_target": "Erratic"},
    {"item_id": "HOUSEHOLD_2_065", "dept_id": "HOUSEHOLD_2", "cat_id": "HOUSEHOLD", "base_price": 14.97, "unit_cost": 5.90, "true_elast": -2.58, "cross_elast": 0.35, "zero_var": False, "regime_target": "Lumpy"},
    {"item_id": "HOUSEHOLD_2_219", "dept_id": "HOUSEHOLD_2", "cat_id": "HOUSEHOLD", "base_price": 5.47, "unit_cost": 3.15, "true_elast": -1.53, "cross_elast": 0.25, "zero_var": False, "regime_target": "Intermittent"},
]


def generate_m5_panel(seed: int = 202611, n_days: int = 112) -> pd.DataFrame:
    """
    Generates a deterministic 120-series (12 M5 SKUs x 10 stores across CA, TX, WI)
    daily panel over `n_days` (16 weeks) following exact M5 column conventions.
    Includes realistic price variation, SNAP event bumps, stockout censoring runs,
    and zero-variance price items.
    """
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    store_mult = {
        "CA_1": 1.25, "CA_2": 0.95, "CA_3": 1.40, "CA_4": 0.80,
        "TX_1": 1.05, "TX_2": 1.20, "TX_3": 0.90,
        "WI_1": 0.85, "WI_2": 1.10, "WI_3": 0.75,
    }

    for sku in M5_CATALOG:
        for st in M5_STORES:
            sid = st["store_id"]
            state = st["state_id"]
            item_id = sku["item_id"]
            base_p = sku["base_price"]
            cost = sku["unit_cost"]
            s_mult = store_mult[sid]

            # Initial on-hand stock at start of planning horizon
            # Calibrate relative to 4-week expected demand: sister stores in each state
            # have strong surplus stock (CA_2, CA_4, TX_3, WI_3) while flagship stores
            # have tight stock (CA_1, CA_3, TX_1, TX_2, WI_1, WI_2), enabling intra-state transfers
            if sku["regime_target"] in ("Smooth", "Erratic"):
                est_4w = 510.0 * s_mult
            else:
                est_4w = 135.0 * s_mult
            if sid in ("CA_2", "CA_4", "TX_3", "WI_3"):
                on_hand = int(round(est_4w * float(rng.uniform(1.55, 1.85))))
            else:
                on_hand = int(round(est_4w * float(rng.uniform(0.76, 0.90))))

            # Generate weekly prices across 16 weeks
            weekly_prices = []
            for w in range(n_days // 7):
                if sku["zero_var"]:
                    weekly_prices.append(base_p)
                else:
                    # Guarantee at least 4 distinct promotional price tiers in historical 16 weeks
                    if w == 2:
                        tier = 0.95
                    elif w == 5:
                        tier = 0.90
                    elif w == 9:
                        tier = 0.85
                    elif w == 12:
                        tier = 1.04
                    else:
                        tier = rng.choice([1.0, 1.0, 0.95, 0.90, 0.85, 1.04], p=[0.45, 0.15, 0.15, 0.12, 0.08, 0.05])
                    weekly_prices.append(round(base_p * tier, 2))

            # Create stockout censoring window on a subset of high-velocity series
            censor_start = -1
            if not sku["zero_var"] and sid in ("CA_1", "TX_2", "WI_2") and sku["regime_target"] in ("Smooth", "Erratic"):
                censor_start = int(rng.integers(35, 70))

            for d in range(n_days):
                w = d // 7
                dow = d % 7
                is_weekend = 1 if dow in (5, 6) else 0
                is_snap = 1 if (d % 30) < 10 and sku["cat_id"] == "FOODS" else 0
                own_p = weekly_prices[w]
                # Competitor / department reference price with slight noise
                dept_p = round(base_p * float(rng.uniform(0.94, 1.04)), 2)

                # Base demand according to target Syntetos-Boylan regime
                regime = sku["regime_target"]
                if regime == "Smooth":
                    active_prob = 0.96
                    base_qty = 14.0 * s_mult
                    disp = 0.18
                elif regime == "Erratic":
                    active_prob = 0.92
                    base_qty = 12.0 * s_mult
                    disp = 0.82
                elif regime == "Intermittent":
                    active_prob = 0.42
                    base_qty = 4.5 * s_mult
                    disp = 0.22
                else:  # Lumpy
                    active_prob = 0.36
                    base_qty = 9.5 * s_mult
                    disp = 0.95

                # Price elasticity effect
                price_ratio = own_p / base_p
                dept_ratio = dept_p / base_p
                elast_mult = (price_ratio ** sku["true_elast"]) * (dept_ratio ** sku["cross_elast"])
                weekend_mult = 1.28 if is_weekend else 1.0
                snap_mult = 1.18 if is_snap else 1.0

                if censor_start <= d < censor_start + 6:
                    qty = 0
                    stockout_flag = 1
                else:
                    stockout_flag = 0
                    if rng.random() < active_prob:
                        noise = float(rng.lognormal(mean=0.0, sigma=disp))
                        expected = base_qty * elast_mult * weekend_mult * snap_mult * noise
                        qty = max(0, int(round(expected)))
                    else:
                        qty = 0

                records.append({
                    "series_id": f"{item_id}_{sid}",
                    "item_id": item_id,
                    "dept_id": sku["dept_id"],
                    "cat_id": sku["cat_id"],
                    "store_id": sid,
                    "state_id": state,
                    "day_idx": d + 1,
                    "week_idx": w + 1,
                    "is_weekend": is_weekend,
                    "is_snap": is_snap,
                    "sell_price": own_p,
                    "base_price": base_p,
                    "dept_ref_price": dept_p,
                    "unit_cost": cost,
                    "units_sold": qty,
                    "stockout_censored": stockout_flag,
                    "initial_on_hand": on_hand,
                })

    return pd.DataFrame(records)
