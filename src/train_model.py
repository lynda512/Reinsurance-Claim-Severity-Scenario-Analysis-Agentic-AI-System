# src/train_model.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path, parse_dates=['inception_date','loss_date','reported_date'])

def prepare_features(df):
    df = df.copy()
    # Feature engineering
    df['loss_year'] = df['loss_date'].dt.year
    df['report_lag_days'] = (df['reported_date'] - df['loss_date']).dt.days.fillna(0)
    # target
    df = df[df['severity'].notna()]
    y = df['severity'].values
    X = df[['line_of_business','exposure','catastrophe_flag','loss_year','report_lag_days','currency']]
    return X, y

def build_pipeline(cat_cols, num_cols):
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median'))
    ])
    pre = ColumnTransformer([
        ('cat', cat_pipe, cat_cols),
        ('num', num_pipe, num_cols)
    ], remainder='drop')
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    pipeline = Pipeline([
        ('pre', pre),
        ('model', model)
    ])
    return pipeline

def train_and_save(data_path, model_path="../models/severity_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    df = load_data(data_path)
    X, y = prepare_features(df)
    cat_cols = ['line_of_business','currency']
    num_cols = ['exposure','catastrophe_flag','loss_year','report_lag_days']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = build_pipeline(cat_cols, num_cols)
    pipeline.fit(X_train, y_train)
    # evaluate quickly
    from sklearn.metrics import mean_absolute_error
    yhat = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, yhat)
    joblib.dump({'pipeline': pipeline}, model_path)
    print(f"Saved model to {model_path} | MAE on test: {mae:.2f}")
    return model_path, mae

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data_out/synthetic_reinsurance.csv")
    parser.add_argument("--model_out", default="../models/severity_model.pkl")
    args = parser.parse_args()
    train_and_save(args.data, args.model_out)

