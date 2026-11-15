# MarginPulse — Dynamic Markdown & Multi-Store Inventory Allocator

Flagship retail operations research & econometric decision platform built on the **Walmart M5 Forecasting** schema (`CA`, `TX`, `WI` across `FOODS`, `HOBBIES`, `HOUSEHOLD`).

## Architecture & Mathematical Formulation

1. **Stage 1 — Forensic Demand-Regime & Leakage Audit (`backend/engine/audit.py`)**
   - Computes **Average Inter-Demand Interval ($	ext{ADI}$)** and **Squared Coefficient of Variation ($	ext{CV}^2$)** per SKU-store series, classifying into Syntetos-Boylan quadrants ($	ext{ADI} = 1.32, 	ext{CV}^2 = 0.49$).
   - Detects **Zero-Variance Price SKUs** (`< 3` distinct prices in `sell_prices.csv`) and locks them out of empirical elasticity fabrication.
   - Flags **Stockout-Censored Runs** ($\ge 5$ consecutive zero-sale days on Smooth/Erratic items) so censored zeroes do not bias log-log demand regressions.

2. **Stage 2 — Panel Price Elasticity & Cross-SKU Cannibalization (`backend/engine/elasticity.py`)**
   - Estimates ridge-regularized log-log panel regression with store fixed effects and 100-iteration weekly block bootstrap for empirical 95% confidence intervals:
     $$\log(Q_{i,s,t} + 1) = \alpha_s + \epsilon_{ii} \log(P_{i,s,t}) + \epsilon_{ij} \log(\bar{P}_{-i,d,t}) + \gamma_1 \text{Weekend}_t + \gamma_2 \text{SNAP}_{s,t} + u_{i,s,t}$$
   - Evaluates out-of-sample on a 28-day temporal holdout using M5 revenue-weighted **WRMSSE**, **MAE**, and **95% prediction interval coverage**.

3. **Stage 3 — Multi-Period MILP Markdown & Transfer Optimizer (`backend/engine/optimizer.py`)**
   - Solved via `scipy.optimize.milp` (HiGHS branch-and-cut solver):
     - Binary markdown tier selection $y_{s,m} \in \{0,1\}$ for $m \in \{0\%, 10\%, 20\%, 30\%\}$.
     - Integer intra-state store-to-store transfers $t^{\text{in}}_s, t^{\text{out}}_s \in \mathbb{Z}_{\ge 0}$ conserving $\sum_{s \in (i, \text{state})} (t^{\text{in}}_s - t^{\text{out}}_s) = 0$.
     - Exact inventory flow balance, markdown budget cap, and aggregate service-level floor constraint.

## Running Locally

```bash
pytest -v
uvicorn backend.main:app --reload --port 8000
```
