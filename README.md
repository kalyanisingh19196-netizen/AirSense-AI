# AirSense AI — Fresh Project

A new Streamlit application foundation with a premium dark environmental-intelligence interface.

## Important data notice
The starter dashboard uses clearly labelled simulated values only to demonstrate the interface.
It does not claim to display live CPCB data or trained predictions. Later stages will add source-labelled
air-quality and weather ingestion, CPCB AQI calculation where inputs support it, model training,
explainability, anomaly detection and validated forecasts.

## Setup on macOS
```bash
cd ~/Desktop/AirSense_AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Folder structure
- `app.py`: new application and navigation
- `src/`: backend modules (to be added in future stages)
- `assets/`: local styling/assets
- `data/raw/`: downloaded source snapshots
- `data/processed/`: cleaned datasets
- `models/`: trained models
- `reports/`: evaluation outputs

## Suggested next build stages
1. Connect reliable air-quality and weather data sources with source timestamps.
2. Validate and clean datasets; implement CPCB AQI methodology only with adequate inputs.
3. Build time-aware features and chronological evaluation.
4. Train and compare forecasting models.
5. Add SHAP explanations, anomaly detection and hotspot analysis.
6. Connect health information as general exposure guidance, not diagnosis.
7. Deploy and document the finished app.
