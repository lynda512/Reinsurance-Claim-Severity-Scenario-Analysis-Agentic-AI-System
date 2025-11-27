# This will genereta a synthtic dataset
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data_out"), exist_ok=True)

def generate_synthetic_reinsurance_data(n=20000, seed=42):
    np.random.seed(seed)
    policy_id = np.arange(1, n+1)
    inception = [datetime(2015,1,1) + timedelta(days=int(x)) for x in np.random.exponential(scale=365*3, size=n)]
    loss_date = [d + timedelta(days=int(np.random.exponential(scale=180))) for d in inception]
    lob = np.random.choice(['Property', 'Casualty', 'Marine', 'Aviation', 'Motor'], size=n, p=[0.4,0.3,0.15,0.05,0.1])
    exposure = np.round(np.random.lognormal(mean=10, sigma=1.2, size=n))  # proxy exposure amount
    severity_base = np.random.lognormal(mean=8, sigma=1.5, size=n)
    catastrophe_flag = np.random.binomial(1, p=np.where(lob=='Property', 0.08, 0.02))
    cat_multiplier = np.where(catastrophe_flag==1, np.random.uniform(5, 50, size=n), 1.0)
    inflation_index = 1.0 + (loss_date - np.array([datetime(2015,1,1)]*n)).astype('timedelta64[D]').astype(int) / 365 * 0.02
    # severity influenced by exposure, lob, catastrophes, and some randomness
    lob_factor = pd.Series(lob).map({'Property':1.2, 'Casualty':0.9, 'Marine':1.0, 'Aviation':1.5, 'Motor':0.8}).values
    severity = severity_base * exposure**0.3 * lob_factor * cat_multiplier * inflation_index
    # apply some missingness and noise
    mask_missing = np.random.rand(n) < 0.01
    severity[mask_missing] = np.nan
    reported_delay_days = np.random.poisson(lam=10, size=n)
    reported_date = [ld + timedelta(days=int(d)) for ld,d in zip(loss_date, reported_delay_days)]
    currency = np.random.choice(['USD','EUR','GBP'], size=n, p=[0.6,0.3,0.1])
    df = pd.DataFrame({
        'policy_id': policy_id,
        'inception_date': inception,
        'loss_date': loss_date,
        'reported_date': reported_date,
        'line_of_business': lob,
        'exposure': exposure,
        'catastrophe_flag': catastrophe_flag,
        'severity': severity,
        'currency': currency
    })
    # Introduce a few duplicates and outliers
    dup_idx = np.random.choice(df.index, size=int(n*0.002), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True).reset_index(drop=True)
    outlier_idx = np.random.choice(df.index, size=int(n*0.001), replace=False)
    df.loc[outlier_idx, 'severity'] *= 100  # extreme outliers
    return df

if __name__ == "__main__":
    df = generate_synthetic_reinsurance_data(n=25000)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data_out", "synthetic_reinsurance.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic data to {out_path}")
