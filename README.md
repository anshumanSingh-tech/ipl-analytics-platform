---
title: IPL Analytics Platform
emoji: 🏏
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# IPL Analytics Platform

**Live Demo:** https://ipl-analytics-platform.onrender.com

End-to-end cricket analytics platform covering 19 IPL seasons (2008-2026), built with Python, Plotly Dash, and Machine Learning. Continuously updated with live Cricsheet data.

---

## Overview

This project was built as a portfolio project to demonstrate end-to-end data science skills, from raw data ingestion to a deployed interactive dashboard.

The platform covers 1,200+ IPL matches and 287,000+ ball-by-ball deliveries, with an automated pipeline to fetch new season data from Cricsheet.org.

---

## Features

### Team Stats
- Season-by-season win trends for all IPL franchises
- Win percentage rankings with venue and toss analysis
- Runs per over breakdown by match phase (powerplay / middle / death)

### Player Explorer
- Career batting and bowling profiles for every IPL player
- Strike rate vs average scatter with player type quadrants
- Phase-wise performance (powerplay, middle, death overs)
- Full sortable leaderboard

### Win Probability Predictor
- Pre-match win probability powered by XGBoost and Random Forest
- Trained on 29 cricket-intelligent features across 19 seasons
- Head-to-head record and venue-specific matchup history
- Calibrated probability output

### Auction Value Simulator
- Real-time auction price prediction using 20 player performance metrics
- Player tier classification (Icon / Premium / Standard / Emerging)
- Similar player comparison table
- Top 25 most valuable players leaderboard

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data ingestion | Python, Requests, Cricsheet JSON API |
| Data cleaning | Pandas, NumPy |
| Feature engineering | Custom rolling window features (29 features) |
| ML models | XGBoost, Random Forest, scikit-learn |
| Explainability | SHAP |
| Dashboard | Plotly Dash, Dash Bootstrap Components |
| Deployment | Hugging Face Spaces, Render, Docker, Waitress |
| Version control | Git, Git LFS, GitHub |

---

## Project Structure

ipl-analytics/
- data/raw/ - Raw Kaggle and Cricsheet data
- data/processed/ - Cleaned CSVs, model files
- notebooks/ - 01 setup, 02 cleaning, 03 EDA, 04 ML models
- src/ - clean.py, features.py, models.py, update_dataset.py
- dashboard/ - app.py, layout.py, callbacks.py
- app.py - Hugging Face entry point
- Dockerfile - Container build for HF Spaces
- waitress_server.py - Production server for Render
- render.yaml - Render deployment config
- requirements.txt

---

## Data Pipeline

Kaggle IPL Dataset (2008-2024) plus Cricsheet JSON API (2025-2026, auto-updated) feeds into src/clean.py for standardised CSVs, then src/features.py for 29 engineered features, then notebooks/04_ml_models.ipynb for trained models, then the dashboard for a live Plotly Dash app, deployed on Hugging Face Spaces and Render.

---

## Key Insights from Analysis

- Mumbai Indians and Chennai Super Kings show the most consistent win rates across all 19 seasons
- Death overs (16-20) average the highest runs per over, confirming death bowling as the scarcest and most valuable T20 skill
- Toss winners choose to field about 65 percent of the time in modern IPL, reflecting the statistical advantage of chasing with dew
- Win probability model achieves approximately 0.53 CV AUC, consistent with sports analytics literature showing T20 outcomes have high inherent randomness not captured by pre-match statistics alone

---

## How to Run Locally

git clone https://github.com/anshumanSingh-tech/ipl-analytics-platform.git
cd ipl-analytics-platform
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python waitress_server.py

### Update with latest IPL data

python src/update_dataset.py
python src/update_dataset.py 2026 --force

---

## Dataset

- Source: Kaggle IPL Complete Dataset (2008-2024) plus Cricsheet.org (2025-2026)
- Matches: 1,200+ across 19 seasons
- Deliveries: 287,000+ ball-by-ball records
- Auto-updated: Cricsheet JSON pipeline fetches new matches automatically

---

## Author

Anshuman Singh
BCA Graduate, AI/ML Specialization
GitHub: https://github.com/anshumanSingh-tech
