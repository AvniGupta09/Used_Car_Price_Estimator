

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"

_pipeline = None   # the model is loaded once and then reused


def load_model():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline

_pipeline = None   # the model is loaded once and then reused


def load_model():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_price(brand, model, vehicle_age, km_driven, seller_type,
                  fuel_type, transmission_type, mileage, engine,
                  max_power, seats):
    """Returns the estimated selling price in rupees."""

    car = pd.DataFrame([{
        "brand": brand,
        "model": model,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }])

    # same text cleaning as data_cleaning.py
    for col in ["brand", "model", "seller_type", "fuel_type", "transmission_type"]:
        car[col] = car[col].str.strip().str.lower()


    pipeline = load_model()
    return float(pipeline.predict(car)[0])


if __name__ == "__main__":
    price = predict_price(
        brand="Maruti", model="Swift", vehicle_age=5, km_driven=50000,
        seller_type="Individual", fuel_type="Petrol",
        transmission_type="Manual", mileage=21.0, engine=1197,
        max_power=83.0, seats=5,
    )
    print(f"Estimated price: Rs {price:,.0f}")