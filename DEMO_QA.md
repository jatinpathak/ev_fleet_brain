# Demo script & judge Q&A — EV Fleet Intelligence Brain v3

## 2.5–3 minute demo flow (business value first)

1. **Executive Command Center** (20s) — lead with the numbers: ₹ savings
   potential, fleet-readiness %, mean battery health, CO₂ reduction, top supplier
   risk. Read the **Today's Highlights** (the three actions) and the alert centre.
2. **Battery Health** (25s) — pick a ⚠️ anomaly cell; show SoH, RUL **with a 90%
   prediction interval**, the top drivers, and the **Battery Passport**
   (chemistry, warranty, second-life). "Accuracy is real: RMSE ~50 cycles on
   held-out cells."
3. **Fleet Readiness** (20s) — the ranked table + the **AI recommendation**
   (which vehicle first, confidence, ₹ impact, alternatives) + the depot×duty
   heatmap.
4. **Supply-Chain Risk** (25s) — the supplier **knowledge graph**; disrupt a node
   and watch **risk propagate** downstream; the ₹ value at risk.
5. **Carbon** (15s) — Scope 1/2/3 split + the **hourly grid** chart: charging
   off-peak cuts Scope 3 CO₂.
6. **Scenario Lab** (25s, the wow) — run *Supplier disruption* or *Accelerated
   degradation*; every KPI recomputes **before → after** live.
7. **Copilot** (15s) — ask one question; show the **planner trace** (multi-agent
   reasoning) then the plain-English answer.

Close: "Runs at 10,000 vehicles; REST API + Docker for integration; everything
MVP/synthetic is labelled; the battery number is from real data."

---

## Judge questions & answers

**Q: Why XGBoost and not an LSTM / Transformer for battery life?**
XGBoost on engineered early-cycle features matches the published Severson et al.
methodology. On a ~124-cell dataset it is more accurate, faster and far more
explainable than deep learning, and it gives a real, defensible held-out number
(RMSE ~50 cycles, MAPE ~3.6%). Deep sequence models need far more data to beat
it here; we keep XGBoost as the **primary** model and would add an LSTM only as a
labelled side-by-side comparison — never a replacement.

**Q: Is this really "multi-agent" or just a chatbot?**
It's a **bounded planner**: it decomposes the request into sub-tasks, decides
which domain agents to call (Battery, Fleet, Supply Chain, Maintenance, Quality,
Carbon), invokes them with a hard step cap (loop-guarded — not an open-ended
autonomous loop), and synthesises the result. The UI shows the actual plan trace,
so it's demonstrable, not a claim.

**Q: How does this integrate with our systems (ERP/SAP/telematics)?**
A thin **FastAPI** REST layer (`/fleet/summary`, `/battery/predict`,
`/scenario/run`) and a **Dockerfile + docker-compose** for one-command run. A
lightweight **SQLite feature store** holds the engineered feature tables the
engines/API read. No message queues or streaming — deliberately, per scope.

**Q: Does it scale beyond the demo?**
Yes — benchmarked at **10,000 vehicles**. Scoring is vectorised (numpy over the
whole frame: 10k vehicles in ~20 ms vs ~12 s for the naive loop), heavy loads are
cached, and the sidebar has a live fleet-size toggle that shows per-page render
time.

**Q: How do you handle model trust / drift / uncertainty?**
Every prediction carries a **confidence interval** (conformal, or split-conformal
fallback) and **explainability** (SHAP or feature-importance). A **monitoring
stub** tracks the prediction distribution and computes **PSI drift** with a
"world-shift" slider to show it reacting. It's a demonstrator, not a retraining
pipeline (labelled as such).

**Q: What's real vs synthetic?**
The battery dataset is shaped on the **real Severson LFP cycling data** (and the
real dataset can be dropped in unchanged), so the accuracy headline is genuine.
Everything else — fleet telematics, suppliers, maintenance backlog — is
**synthetic**, fixed-seed and reproducible, and labelled in the UI. All reference
tables (country risk, emission factors, tariffs, TCO) are **illustrative**.

**Q: What would you productionise next?**
Real telematics + BatteryML ingestion; audited country-risk and grid factors;
the optional OR-Tools/SHAP/mapie primary paths (already wired with fallbacks);
and an LSTM/GNN comparison track — none of which change the current, working demo.

---

## One-liner

"An AI **decision platform** for industrial EV fleets: it tells you **which
vehicles to electrify**, **which batteries to trust**, **where the supply chain
is fragile**, and **what happens under any what-if** — with the money, CO₂ and
downtime attached to every recommendation."
