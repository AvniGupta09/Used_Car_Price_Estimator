"""
app.py
------
A small website: enter or choose the car details, get the estimated price.

Run from the project root:
    python app.py
Then open http://127.0.0.1:5000

Everything else happens automatically. If they are missing, app.py first
creates data/processed.csv and trains the model (artifacts/model.pkl).
The first start takes about a minute, later starts are instant.

To rebuild everything after changing the data or the code:
    python app.py --retrain
"""

import math
import os
import re
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from src import features, train
from src.predict import predict_price

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PATH = PROJECT_ROOT / "data" / "raw.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.pkl"

app = Flask(__name__)


# ----------------------------------------------------------------------
# Get everything ready before the website starts
# ----------------------------------------------------------------------
def prepare(force=False):
    """Create data/processed.csv and artifacts/model.pkl if they are missing."""
    if not RAW_PATH.exists():
        raise SystemExit(f"Could not find {RAW_PATH}. Put raw.csv in the data folder first.")

    if force or not PROCESSED_PATH.exists():
        print("Step 1 of 2: cleaning the data and creating features...")
        features.main()

    if force or not MODEL_PATH.exists():
        print("Step 2 of 2: training the models (the first time takes about a minute)...")
        train.main()


# Flask's debug mode starts this file twice. Only the first start may force a rebuild.
is_first_start = os.environ.get("WERKZEUG_RUN_MAIN") != "true"
prepare(force="--retrain" in sys.argv and is_first_start)

# ----------------------------------------------------------------------
# Dropdown values come from the data the model was trained on
# ----------------------------------------------------------------------
data = pd.read_csv(PROCESSED_PATH)

MODELS_BY_BRAND = {
    brand: sorted(group["model"].unique())
    for brand, group in data.groupby("brand")
}
FUELS = sorted(data["fuel_type"].unique())
SELLERS = sorted(data["seller_type"].unique())
TRANSMISSIONS = sorted(data["transmission_type"].unique())
SEATS = sorted(int(x) for x in data["seats"].unique())

# typical specs of every brand + model, used to pre-fill the form
DEFAULTS = {}
for (brand, model), group in data.groupby(["brand", "model"]):
    DEFAULTS[f"{brand}|{model}"] = {
        "fuel_type": group["fuel_type"].mode()[0],
        "transmission_type": group["transmission_type"].mode()[0],
        "seats": int(group["seats"].median()),
        "engine": int(group["engine"].median()),
        "max_power": round(float(group["max_power"].median()), 1),
        "mileage": round(float(group["mileage"].median()), 1),
    }

# name: (label, minimum, maximum, whole number?)  - a little wider than the data
NUMBER_FIELDS = {
    "vehicle_age": ("Vehicle age (years)", 0, 30, True),
    "km_driven":   ("Kilometres driven", 0, 1_000_000, True),
    "mileage":     ("Mileage (km/l)", 1, 50, False),
    "engine":      ("Engine (cc)", 500, 8000, True),
    "max_power":   ("Max power (bhp)", 20, 700, False),
}

MAX_KM_PER_YEAR = 150_000   # same rule that data_cleaning.py uses
TYPICAL_ERROR = 0.25       # the model is off by about 13% on average (test data)
UPPER_WORDS = {"bmw", "mg", "amg", "cng", "lpg"}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def pretty(text):
    """'land rover' -> 'Land Rover', 'bmw' -> 'BMW', 'xuv500' -> 'XUV500'."""
    def fix(match):
        word = match.group(0)
        if word in UPPER_WORDS or any(ch.isdigit() for ch in word):
            return word.upper()
        return word.capitalize()
    return re.sub(r"[A-Za-z0-9]+", fix, str(text))


def options(values):
    """[(value, label), ...] for a dropdown."""
    return [(v, pretty(v)) for v in values]


def format_inr(number):
    """4568160 -> '45,68,160' (Indian digit grouping)."""
    digits = str(int(round(number)))
    if len(digits) <= 3:
        return digits
    last3, rest = digits[-3:], digits[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return ",".join(groups + [last3])


def in_words(number):
    """456816 -> '4.57 lakh', 25000000 -> '2.50 crore'."""
    if number >= 1e7:
        return f"{number / 1e7:.2f} crore"
    return f"{number / 1e5:.2f} lakh"


def short_inr(number):
    """Short label for the chart axis: 450000 -> '4.5L', 25000000 -> '2.5Cr'."""
    if number == 0:
        return "0"
    if number >= 1e7:
        return f"{round(number / 1e7, 2):g}Cr"
    return f"{round(number / 1e5, 2):g}L"


# ----------------------------------------------------------------------
# Reading and checking the form
# ----------------------------------------------------------------------
def read_form(form):
    """Returns (values, error_message, warning_message)."""
    values = {}

    brand = form.get("brand", "").strip().lower()
    model = form.get("model", "").strip().lower()
    if not brand:
        return None, "Please enter or choose a brand.", None
    if not model:
        return None, "Please enter or choose a model.", None
    values["brand"], values["model"] = brand, model

    for name, allowed in [("fuel_type", FUELS), ("seller_type", SELLERS),
                          ("transmission_type", TRANSMISSIONS)]:
        if form.get(name) not in allowed:
            return None, "Please choose a value for every dropdown.", None
        values[name] = form.get(name)

    try:
        seats = int(form.get("seats", ""))
    except ValueError:
        return None, "Please choose the number of seats.", None
    if seats not in SEATS:
        return None, "Please choose the number of seats.", None
    values["seats"] = seats

    for name, (label, low, high, whole) in NUMBER_FIELDS.items():
        try:
            number = float(form.get(name, ""))
        except ValueError:
            return None, f"{label}: please enter a number.", None
        if not low <= number <= high:
            return None, f"{label}: please enter a value between {low:,} and {high:,}.", None
        values[name] = int(number) if whole else number

    km_per_year = values["km_driven"] / max(values["vehicle_age"], 1)
    if km_per_year > MAX_KM_PER_YEAR:
        return None, ("That is more than 150,000 km per year. "
                      "Please check the vehicle age and kilometres driven."), None

    # brands and models the model has never seen still work, but are less reliable
    warning = None
    if brand not in MODELS_BY_BRAND:
        warning = "This brand was not in the training data, so the estimate is less reliable."
    elif model not in MODELS_BY_BRAND[brand]:
        warning = "This model was not in the training data, so the estimate is less reliable."

    return values, None, warning


# ----------------------------------------------------------------------
# The price curve (drawn as an SVG in the page)
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 480, 270
LEFT, RIGHT, TOP, BOTTOM = 56, 20, 28, 44


def nice_axis(value):
    """Tidy tick step and number of steps for a chart axis, e.g. 1280000 -> (500000, 3)."""
    raw = value / 4
    size = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * size
        if step >= raw:
            break
    return step, math.ceil(value / step)


def make_chart(curve, age, price):
    """curve = [(age, price), ...]. Returns everything the SVG needs."""
    last_age = curve[-1][0]
    step, n_steps = nice_axis(max(max(p for _, p in curve), price) * 1.02)
    y_max = step * n_steps
    plot_w = WIDTH - LEFT - RIGHT
    plot_h = HEIGHT - TOP - BOTTOM
    base_y = TOP + plot_h

    def x_of(a):
        return LEFT + a / last_age * plot_w

    def y_of(p):
        return TOP + (1 - max(p, 0) / y_max) * plot_h

    points = [(x_of(a), y_of(p)) for a, p in curve]
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = f"{line} L{points[-1][0]:.1f},{base_y} L{points[0][0]:.1f},{base_y} Z"

    marker_x, marker_y = x_of(age), y_of(price)
    return {
        "width": WIDTH, "height": HEIGHT, "left": LEFT, "right": WIDTH - RIGHT,
        "base_y": base_y,
        "line": line, "area": area,
        "x_ticks": [(round(x_of(a), 1), a) for a in range(0, last_age + 1, 5)],
        "y_ticks": [(round(y_of(step * i), 1), short_inr(step * i)) for i in range(n_steps + 1)],
        "marker_x": round(marker_x, 1), "marker_y": round(marker_y, 1),
        "label_x": round(marker_x - 12 if marker_x > WIDTH * 0.7 else marker_x + 12, 1),
        "label_y": round(max(marker_y - 12, TOP + 8), 1),
        "label_anchor": "end" if marker_x > WIDTH * 0.7 else "start",
        "label": short_inr(price),
    }


def build_result(values, price, warning):
    age = values["vehicle_age"]

    # the same car at ages 0, 1, 2 ... with the same km driven per year
    km_per_year = values["km_driven"] / max(age, 1)
    curve = []
    for a in range(max(15, age) + 1):
        same_car = {**values, "vehicle_age": a, "km_driven": km_per_year * a}
        curve.append((a, predict_price(**same_car)))

    return {
        "car": f"{pretty(values['brand'])} {pretty(values['model'])}",
        "price": format_inr(price),
        "words": in_words(price),
        "low": format_inr(round(price * (1 - TYPICAL_ERROR), -3)),
        "high": format_inr(round(price * (1 + TYPICAL_ERROR), -3)),
        "warning": warning,
        "chart": make_chart(curve, age, price),
    }


# ----------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    form = {}
    result = None
    error = None

    if request.method == "POST":
        form = request.form
        values, error, warning = read_form(form)

        if error is None:
            try:
                price = predict_price(**values)
                result = build_result(values, price, warning)
            except FileNotFoundError:
                error = "The model file was not found. Run  python -m src.train  first."

    page_data = {
        "models": {b: [pretty(m) for m in models] for b, models in MODELS_BY_BRAND.items()},
        "defaults": DEFAULTS,
    }

    return render_template(
        "index.html",
        form=form,
        result=result,
        error=error,
        brands=[pretty(b) for b in sorted(MODELS_BY_BRAND)],
        fuels=options(FUELS),
        sellers=options(SELLERS),
        transmissions=options(TRANSMISSIONS),
        seats=options(SEATS),
        number_fields=NUMBER_FIELDS,
        page_data=page_data,
    )


if __name__ == "__main__":
    app.run(debug=True)