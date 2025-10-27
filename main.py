from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI(title="Location-wise Prediction API", version="2.0")

# ================== Load Data ================== #
CSV_PATH = "probs_loc.csv"

try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading CSV file: {str(e)}")

# Normalize column names for consistency
df.columns = [c.strip() for c in df.columns]

# Extract available locations (all columns except known identifiers)
non_location_cols = {"feature_type", "hour", "dow", "month"}
available_locations = [col for col in df.columns if col not in non_location_cols]

# Define prediction categories
FEATURE_TYPES = df["feature_type"].unique().tolist() if "feature_type" in df.columns else [
    "Camera Fault", "Helmet Not Detected", "Zone Intrusion"
]


# ================== Helper Function ================== #
def generate_predictions(location: str,
                         start_datetime: Optional[str] = None,
                         end_datetime: Optional[str] = None):
    """
    Generate location-wise predictions between given datetimes or for future hours.
    """

    if location not in available_locations:
        raise ValueError(f"Invalid location '{location}'. Available: {available_locations}")

    now = datetime.now()

    # --- Handle time range ---
    if not start_datetime and not end_datetime:
        start_datetime = now + timedelta(hours=1)
        end_datetime = start_datetime + timedelta(hours=3)
    else:
        if isinstance(start_datetime, str):
            start_datetime = datetime.fromisoformat(start_datetime)
        if isinstance(end_datetime, str):
            end_datetime = datetime.fromisoformat(end_datetime)

    predictions_list = []
    current_time = start_datetime

    # --- Loop through hours between start & end ---
    while current_time <= end_datetime:
        hour = current_time.hour
        dow = current_time.weekday()  # 0=Monday
        month = current_time.month

        prediction_data = {}

        for feature in FEATURE_TYPES:
            # Filter exact match rows
            match = df[
                (df["feature_type"] == feature) &
                (df["hour"] == hour) &
                (df["dow"] == dow) &
                (df["month"] == month)
            ]

            # Use exact value if match found, else use average of that feature
            if not match.empty and location in match.columns:
                value = float(match[location].iloc[0])
            else:
                avg_value = df.loc[df["feature_type"] == feature, location].mean() if location in df.columns else 0.0
                value = float(avg_value) if pd.notna(avg_value) else 0.0

            prediction_data[feature] = round(value, 6)

        predictions_list.append({
            "datetime": current_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "location": location,
            "predictions": prediction_data,
            "timestamp": datetime.now().isoformat()
        })

        current_time += timedelta(hours=1)

    result = {
        "location": location,
        "total_predictions": len(predictions_list),
        "start_datetime": start_datetime.strftime("%Y-%m-%dT%H:%M:%S"),
        "end_datetime": end_datetime.strftime("%Y-%m-%dT%H:%M:%S"),
        "predictions": predictions_list
    }

    return result


# ================== API Endpoints ================== #
@app.get("/predict")
def get_prediction(
    location: str = Query(..., description="Location/column name from the CSV file"),
    start_datetime: Optional[str] = Query(None, description="Start datetime in ISO format (optional)"),
    end_datetime: Optional[str] = Query(None, description="End datetime in ISO format (optional)")
):
    """
    Get predictions for a specific location and optional datetime range.
    If no range is given → predicts for next 4 hours.
    """
    try:
        result = generate_predictions(location, start_datetime, end_datetime)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {
        "message": "Welcome to the Location-wise Prediction API 🚀",
        "usage_example": (
            "/predict?location=Finishing Stand&start_datetime=2025-10-23T20:00:00"
            "&end_datetime=2025-10-23T23:00:00"
        ),
        "available_locations": available_locations
    }
