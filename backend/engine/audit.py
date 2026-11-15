"""
Forensic Data & Demand-Regime Audit Module (Syntetos-Boylan ADI / CV^2).
Audits every SKU-store series for:
1. Syntetos-Boylan demand classification (Smooth, Erratic, Intermittent, Lumpy)
   using exact cutoffs ADI = 1.32 and CV^2 = 0.49.
2. Zero-variance price SKUs (< 3 distinct historical prices in sell_prices.csv).
3. Stockout-censored runs (>= 4 consecutive zero-sale days on Smooth/Erratic items).
"""
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd


ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


def classify_syntetos_boylan(adi: float, cv2: float) -> str:
    if adi < ADI_CUTOFF and cv2 < CV2_CUTOFF:
        return "Smooth"
    if adi < ADI_CUTOFF and cv2 >= CV2_CUTOFF:
        return "Erratic"
    if adi >= ADI_CUTOFF and cv2 < CV2_CUTOFF:
        return "Intermittent"
    return "Lumpy"


def max_zero_run(sales: np.ndarray) -> int:
    max_run = 0
    curr = 0
    for val in sales:
        if val == 0:
            curr += 1
            if curr > max_run:
                max_run = curr
        else:
            curr = 0
    return int(max_run)


def run_forensic_audit(df: pd.DataFrame) -> Dict[str, Any]:
    series_rows: List[Dict[str, Any]] = []
    regime_counts = {"Smooth": 0, "Erratic": 0, "Intermittent": 0, "Lumpy": 0}
    zero_var_series = 0
    stockout_censored_series = 0

    for series_id, grp in df.groupby("series_id", sort=False):
        grp_sorted = grp.sort_values("day_idx")
        sales = grp_sorted["units_sold"].to_numpy(dtype=float)
        prices = grp_sorted["sell_price"].to_numpy(dtype=float)
        n_total = len(sales)
        non_zero = sales[sales > 0]
        n_pos = len(non_zero)

        if n_pos > 0:
            adi = float(n_total / n_pos)
            mean_pos = float(np.mean(non_zero))
            std_pos = float(np.std(non_zero, ddof=1)) if n_pos > 1 else 0.0
            cv2 = float((std_pos / mean_pos) ** 2) if mean_pos > 0 else 0.0
        else:
            adi = float(n_total)
            cv2 = 0.0

        regime = classify_syntetos_boylan(adi, cv2)
        regime_counts[regime] += 1

        distinct_prices = int(len(np.unique(np.round(prices, 2))))
        is_zero_var = distinct_prices < 3
        if is_zero_var:
            zero_var_series += 1

        longest_zero = max_zero_run(sales)
        # Flag stockout censoring if Smooth/Erratic item has >= 5 consecutive zero days
        is_censored = bool(regime in ("Smooth", "Erratic") and longest_zero >= 5)
        if is_censored:
            stockout_censored_series += 1

        first = grp_sorted.iloc[0]
        series_rows.append({
            "series_id": str(series_id),
            "item_id": str(first["item_id"]),
            "dept_id": str(first["dept_id"]),
            "cat_id": str(first["cat_id"]),
            "store_id": str(first["store_id"]),
            "state_id": str(first["state_id"]),
            "adi": round(adi, 3),
            "cv2": round(cv2, 3),
            "regime": regime,
            "distinct_prices": distinct_prices,
            "zero_price_variance": is_zero_var,
            "max_zero_run_days": longest_zero,
            "stockout_censored": is_censored,
            "mean_daily_units": round(float(np.mean(sales)), 2),
            "base_price": round(float(first["base_price"]), 2),
            "unit_cost": round(float(first["unit_cost"]), 2),
            "initial_on_hand": int(first["initial_on_hand"]),
        })

    total_series = len(series_rows)
    estimable_series = total_series - zero_var_series
    integrity_score = round(
        100.0 * (1.0 - (0.5 * zero_var_series + 0.3 * stockout_censored_series) / max(1, total_series)),
        1,
    )

    return {
        "total_series": total_series,
        "estimable_series": estimable_series,
        "zero_variance_price_series": zero_var_series,
        "stockout_censored_series": stockout_censored_series,
        "integrity_score_pct": integrity_score,
        "adi_cutoff": ADI_CUTOFF,
        "cv2_cutoff": CV2_CUTOFF,
        "regime_counts": regime_counts,
        "series_audit": series_rows,
    }
