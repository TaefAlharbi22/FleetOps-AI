# FleetOps AI V2

A polished Streamlit prototype for an AI Operations competition project.

## Main experience
- AI Operations Agent with chat
- Project-to-equipment matching
- Operations Command Center
- Executive report generation
- Transparent scoring logic

## Data
The app automatically searches common project/Colab locations for:
`logistics_predictive_maintenanceV2.csv`

If it cannot find the file, it asks for a CSV upload.

## Gemini
Add the following to Streamlit Secrets:
```toml
GEMINI_API_KEY = "your-key"
```

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
