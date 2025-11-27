# src/agent/tools.py
import pandas as pd
import joblib
import os
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(BASE, "data_out", "synthetic_reinsurance.csv")
MODEL_PATH = os.path.join(BASE, "models", "severity_model.pkl")

def load_df(limit=None):
    df = pd.read_csv(DATA_PATH, parse_dates=['inception_date','loss_date','reported_date'])
    if limit:
        return df.head(limit)
    return df

def query_data(filters: dict = None, limit=200):
    """
    filters e.g. {'line_of_business': 'Property', 'loss_year': 2020}
    """
    df = load_df()
    if filters:
        dfq = df.copy()
        for k,v in filters.items():
            if k in dfq.columns:
                dfq = dfq[dfq[k]==v]
        return dfq.head(limit).to_dict(orient='records')
    return df.head(limit).to_dict(orient='records')

def load_model():
    assert os.path.exists(MODEL_PATH), "Model not found. Train model first."
    obj = joblib.load(MODEL_PATH)
    return obj['pipeline']

def run_model_on_records(records):
    """
    records: list[dict] - rows with keys matching X columns
    returns prediction list
    """
    df = pd.DataFrame(records)
    # create same features used by training
    df['loss_year'] = pd.to_datetime(df['loss_date']).dt.year
    df['report_lag_days'] = (pd.to_datetime(df['reported_date']) - pd.to_datetime(df['loss_date'])).dt.days.fillna(0)
    X = df[['line_of_business','exposure','catastrophe_flag','loss_year','report_lag_days','currency']]
    model = load_model()
    preds = model.predict(X)
    return preds.tolist()

def simulate_scenario(shock_pct=0.2, subset_filters=None, sample_size=100):
    """
    Apply shock to exposure/severity and run predictions for sample.
    """
    df = load_df()
    if subset_filters:
        for k,v in subset_filters.items():
            if k in df.columns:
                df = df[df[k]==v]
    if sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    original = df.copy()
    df['exposure'] = df['exposure'] * (1 + shock_pct)
    records = df.to_dict(orient='records')
    preds = run_model_on_records(records)
    return {
        'sample_size': len(df),
        'shock_pct': shock_pct,
        'predictions_mean': float(np.mean(preds)),
        'predictions_median': float(np.median(preds)),
        'predictions_summary': {
            'min': float(np.min(preds)),
            'max': float(np.max(preds))
        }
    }
