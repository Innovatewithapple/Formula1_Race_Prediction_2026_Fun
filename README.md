# 🏎️ Formula 1 Podium Predictor (XGBoost Regressor)

An advanced, leak-proof machine learning model built to forecast F1 results for the **Paddock ML Challenge**. 

Let's be completely honest: data models, telemetry streams, and hyperparameter grids are fantastic for finding performance patterns, but **the drivers are the real heroes.** The legends pushing these cars to the absolute absolute limit at 200+ MPH are what make this sport pure magic. 

This pipeline simply takes their incredible on-track speed, profiles their Saturday Qualifying gaps, studies their heroic defensive Sprint race-craft over multiple seasons (2023-2025), and attempts to guess who will stand on the steps holding the trophies on Sunday afternoon!

This model shifts away from basic classification to leverage a highly robust **XGBoost Regression architecture**, predicting the exact continuous numeric finishing positions of the 20-car grid rather than isolated categorical buckets. This structurally eliminates duplicate placement bugs and ensures a stable, rule-abiding 1-to-20 leaderboard every single race weekend.

## 📊 The Feature Engineering Strategy
The core competitive edge of this model rests on a massive multi-season dataset spanning **2023, 2024, and 2025 seasonal history** (comprising over 1,400 individual driver telemetry snapshots) combined with custom race-craft dimensions:
* **`quali_position`** & **`final_quali_gap`**: Saturday single-lap qualifying pace benchmarks.
* **`sprint_position_delta`**: Our custom engineered feature mapping a driver's raw position change during the Saturday Sprint race (`sprint_quali_position - sprint_position`). This provides a vital signal regarding a car's heavy-fuel tyre degradation and long-run race craft.
* **`team_name` (One-Hot Encoded)**: Dynamically weights baseline manufacturer performance shifting profiles across regulatory eras.

## 🛡️ Crash-Proof Defensive Ingestion
Built explicitly for headless platform parsing, `predict.py` executes strict structural safeguards:
* **Feature Schema Reconciliation**: Automatically reindexes incoming live shapes against `model.feature_names_in_`. Any missing columns are padded with `0` and extra columns are cleanly discarded—preventing errors if new teams or driver lineups enter.
* **Absent Sprint Handling**: On standard non-sprint race weekends, the data factory captures empty elements and applies strict numeric placeholder flags (`-99` and `99.0`), aligning perfectly with the tree split boundaries established during training.
* **OpenF1 String-List Unpacking**: Dynamically handles raw time floats as well as list strings (e.g. `[0.125, 0.0, None]`), cleanly isolating the furthest valid gap record reached.

## 🚀 Repository Structure
* `predict.py`: Automated leak-proof inference entry point script.
* `f1_regressor_multi_season_2.65.pkl`: Fine-tuned multi-season champion model binary.
* `paddock.yaml`: Configuration instructions mapping the execution command.
* `requirements.txt`: Python software package dependencies tree.

---
*Developed as a highly optimized, cross-validated forecasting asset for the 2026 Grid Campaign.*
