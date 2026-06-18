import pandas as pd
from backend.forecasting import ForecastingEngine

# Load data
df = pd.read_csv("data/processed/events_clean.csv")
df = df[df["duration_to_close_min"].notna() & df["severity_tier"].notna()]

engine = ForecastingEngine()
acc = engine.sev_model.score(engine.preprocessor.transform(df), df["severity_tier"])
print(f"Severity Model Accuracy: {acc*100:.2f}%")
