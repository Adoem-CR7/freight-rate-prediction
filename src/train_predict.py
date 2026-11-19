from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"

MAIN_OUTPUT = ROOT_DIR / "validation_predictions.csv"
DECEMBER_OUTPUT = DATA_DIR / "december_predictions.csv"

def prepare_features(df, december=False):

    df = df.copy()

    # Date features
    df["date"] = pd.to_datetime(df["date"])
    df["month_num"] = df["date"].dt.month

    # Invalid shipment weights
    df.loc[df["weight"] <= 0, "weight"] = np.nan

    # December model uses weekday information
    if december:
        df["day_of_week"] = df["date"].dt.dayofweek

    return df

MAIN_FEATURES = [
    "distance",
    "weight",
    "market_index",
    "month_num",
    "equipment",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon"
]

DECEMBER_FEATURES = [
    "distance",
    "weight",
    "month_num",
    "day_of_week",
    "equipment",
    "pickup",
    "delivery"
]

CATEGORICAL_FEATURES = [
    "equipment",
    "pickup",
    "delivery"
]

def validate_input_data(df, features, require_target=False):

    # Check that all required columns are present
    required_columns = set(features + ["date", "distance"])

    if require_target:
        required_columns.add("posted_rate")

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )
    
    # Check load IDs when available
    if "load_id" in df.columns:

        if df["load_id"].isna().any():
            raise ValueError("Dataset contains missing load IDs")

        if df["load_id"].duplicated().any():
            raise ValueError("Dataset contains duplicate load IDs")

    # Check shipment distances
    if not np.isfinite(df["distance"]).all():
        raise ValueError("Dataset contains invalid distances")

    if (df["distance"] <= 0).any():
        raise ValueError("Dataset contains non-positive distances")

    # Check target values when training
    if require_target:

        if not np.isfinite(df["posted_rate"]).all():
            raise ValueError("Training data contains invalid target values")

        if (df["posted_rate"] <= 0).any():
            raise ValueError("Training data contains non-positive target values")

def create_model():

    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

def train_model(train_df, features):

    X_train = train_df[features].copy()

    # Target: rate per mile
    y_train = (
        train_df["posted_rate"] /
        train_df["distance"]
    )

    # Distance-weighted MAE
    sample_weights = train_df["distance"]

    # Categorical preprocessing
    categories = {}

    for col in CATEGORICAL_FEATURES:

        X_train[col] = X_train[col].astype("category")

        categories[col] = (
            X_train[col].cat.categories
        )

    # Train model
    model = create_model()

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        categorical_feature=CATEGORICAL_FEATURES
    )

    return model, categories

def predict_rates(model, df, features, categories):

    X = df[features].copy()

    # Apply training categories
    for col in CATEGORICAL_FEATURES:

        X[col] = pd.Categorical(
            X[col],
            categories=categories[col]
        )

    # Predict rate per mile
    predicted_rpm = model.predict(X)

    # Convert to total freight rate
    predicted_rate = (
        predicted_rpm *
        df["distance"].to_numpy()
    )

    # Check for invalid predictions
    if not np.isfinite(predicted_rate).all():
        raise ValueError(
            "Model produced NaN or infinite predictions"
        )

    # Ensure positive freight rates
    predicted_rate = np.maximum(
        predicted_rate,
        0.01
    )

    return predicted_rate

def generate_main_predictions(train_df):

    print("Training main prediction model...")

    validation = pd.read_csv(VALIDATION_PATH)

    # Validate the input before feature engineering
    validate_input_data(
        validation,
        features=[
            feature for feature in MAIN_FEATURES
            if feature != "month_num"
        ],
        require_target=False
    )

    validation = prepare_features(
        validation,
        december=False
    )

    model, categories = train_model(
        train_df,
        MAIN_FEATURES
    )

    predictions = predict_rates(
        model,
        validation,
        MAIN_FEATURES,
        categories
    )

    # Load official template
    template = pd.read_csv(TEMPLATE_PATH)

    prediction_lookup = pd.DataFrame({
        "load_id": validation["load_id"],
        "predicted_rate": predictions
    })

    # Preserve the template's load_id order
    output = (
        template[["load_id"]]
        .merge(
            prediction_lookup,
            on="load_id",
            how="left",
            validate="one_to_one"
        )
    )

    assert len(output) == 12000
    assert output["predicted_rate"].notna().all()
    assert np.isfinite(output["predicted_rate"]).all()
    assert (output["predicted_rate"] > 0).all()

    output.to_csv(
        MAIN_OUTPUT,
        index=False
    )

    print(f"Saved: {MAIN_OUTPUT}")

    return model

def generate_december_predictions(train_df):

    print("Training December prediction model...")

    december = pd.read_csv(DECEMBER_PATH)

    # Validate original December inputs
    validate_input_data(
        december,
        features=[
            feature for feature in DECEMBER_FEATURES
            if feature not in ["month_num", "day_of_week"]
        ],
        require_target=False
    )

    december = prepare_features(
        december,
        december=True
    )

    model, categories = train_model(
        train_df,
        DECEMBER_FEATURES
    )

    predictions = predict_rates(
        model,
        december,
        DECEMBER_FEATURES,
        categories
    )

    # Preserve the original December structure
    december["predicted_rate"] = predictions

    output_columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate"
    ]

    output = december[output_columns].copy()

    # Validate December output
    if len(output) != 31:
        raise ValueError(
            f"Expected 31 December predictions, got {len(output)}"
        )

    if not np.isfinite(output["predicted_rate"]).all():
        raise ValueError("December predictions contain invalid values")

    if (output["predicted_rate"] <= 0).any():
        raise ValueError("December predictions must be positive")

    # Save dates in the original format
    output["date"] = (
        output["date"].dt.strftime("%Y-%m-%d")
    )

    output.to_csv(
        DECEMBER_OUTPUT,
        index=False
    )

    print(f"Saved: {DECEMBER_OUTPUT}")

    return model

def main():

    print("Loading training data...")

    train = pd.read_csv(TRAIN_PATH)

    # Check training data before processing
    validate_input_data(
        train,
        features=[
            feature for feature in MAIN_FEATURES
            if feature != "month_num"
        ],
        require_target=True
    )

    # Main model
    main_train = prepare_features(
        train,
        december=False
    )

    generate_main_predictions(main_train)

    # December model
    december_train = prepare_features(
        train,
        december=True
    )

    generate_december_predictions(december_train)

    print("\nAll prediction files generated successfully!")


if __name__ == "__main__":
    main()