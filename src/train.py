
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"

TARGET = "selling_price"

NUMBER_COLS = ["vehicle_age", "km_driven", "mileage", "engine", "max_power",
               "seats"]
ONEHOT_COLS = ["seller_type", "fuel_type", "transmission_type"]
TARGET_ENCODE_COLS = ["brand", "model"]


def load_processed(path=PROCESSED_PATH):
    return pd.read_csv(path)


def split_data(df, test_size=0.2, random_state=42):
    """Same random_state = same split every time."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_split(X_train, X_test, y_train, y_test):

    train = X_train.copy()
    train[TARGET] = y_train
    test = X_test.copy()
    test[TARGET] = y_test
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)


def build_pipeline(model, scale=False, number_cols=NUMBER_COLS):

    number_step = StandardScaler() if scale else "passthrough"
    cv_generator = KFold(n_splits=5, shuffle=True, random_state=42)
    prep = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
        ("target_enc", TargetEncoder(cv=cv_generator), TARGET_ENCODE_COLS),
        ("numbers", number_step, number_cols),
    ])

    pipe = Pipeline([("prep", prep), ("model", model)])

    # train on log(price), predictions come back in rupees automatically
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                      inverse_func=np.expm1)


def get_models():
    return {
        "Linear Regression": build_pipeline(LinearRegression(), scale=True),
        "Random Forest": build_pipeline(
            RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        "XGBoost": build_pipeline(
            XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                         random_state=42)),
    }


def compare_models(models, X_train, y_train):
    """5-fold cross-validation. Returns one row per model."""
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"r2": "r2",
               "mae": "neg_mean_absolute_error",
               "mape": "neg_mean_absolute_percentage_error"}

    rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X_train, y_train, cv=folds, scoring=scoring)
        rows.append({
            "model": name,
            "R2": scores["test_r2"].mean(),
            "MAE (Rs)": -scores["test_mae"].mean(),
            "avg % error": -scores["test_mape"].mean() * 100,
        })
    return pd.DataFrame(rows).sort_values("avg % error").reset_index(drop=True)


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    return {
        "R2": r2_score(y, pred),
        "MAE (Rs)": mean_absolute_error(y, pred),
        "avg % error": mean_absolute_percentage_error(y, pred) * 100,
    }


def main():
    df = load_processed()
    X_train, X_test, y_train, y_test = split_data(df)
    save_split(X_train, X_test, y_train, y_test)
    print(f"train: {len(X_train)} rows, test: {len(X_test)} rows (saved in data/)\n")

    # compare models using ONLY the train data
    models = get_models()
    results = compare_models(models, X_train, y_train)
    print("Cross-validation results (lower % error is better):")
    print(results.round(2).to_string(index=False), "\n")

    # best = lowest average % error
    best_name = results.loc[0, "model"]
    best_model = models[best_name]
    print("Best model:", best_name)

    # one final check on the test data
    best_model.fit(X_train, y_train)
    test_scores = evaluate(best_model, X_test, y_test)
    print("Test scores:", {k: round(v, 2) for k, v in test_scores.items()})

    # train again on ALL data and save it for predict.py
    best_model.fit(df.drop(columns=[TARGET]), df[TARGET])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()