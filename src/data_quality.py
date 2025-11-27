
# src/data_quality.py
import pandas as pd
import numpy as np
import json
import os

def load_data(path):
    df = pd.read_csv(path, parse_dates=['inception_date','loss_date','reported_date'])
    return df

def data_quality_report(df):
    report = {}
    report['n_rows'] = int(len(df))
    report['n_cols'] = int(len(df.columns))
    report['missing_perc'] = df.isnull().mean().to_dict()
    report['duplicate_rows'] = int(df.duplicated().sum())
    # outlier detection for severity using IQR
    if 'severity' in df.columns:
        severity = df['severity'].dropna()
        q1 = float(severity.quantile(0.25))
        q3 = float(severity.quantile(0.75))
        iqr = q3 - q1
        upper = q3 + 3*iqr
        lower = q1 - 3*iqr
        report['severity_outliers_count'] = int(((severity>upper) | (severity<lower)).sum())
        report['severity_q1_q3'] = {'q1': q1, 'q3': q3}
    # basic distributions
    report['lob_counts'] = df['line_of_business'].value_counts().to_dict() if 'line_of_business' in df.columns else {}
    return report

def save_report(report, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=int)
    print(f"Saved data quality report to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data_out/synthetic_reinsurance.csv")
    parser.add_argument("--out", default="../data_out/data_quality_report.json")
    args = parser.parse_args()
    df = load_data(args.data)
    report = data_quality_report(df)
    save_report(report, args.out)
