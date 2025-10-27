import pandas as pd
from datetime import datetime

def load_files():
    # Load probs_loc.csv and get column names as valid features
    loc_df = pd.read_csv('probs_loc.csv')
    print(f"Loaded probs_loc.csv: {loc_df.shape}")  # Debug shape
    # Extract column names (excluding base columns) as valid features
    valid_features = [col for col in loc_df.columns if col not in ['feature_type', 'hour', 'dow', 'month']]
    if not valid_features:
        print("Warning: No valid features found in probs_loc.csv")
    print(f"Valid features: {valid_features}")  # Debug valid features
    return {'loc': loc_df, 'valid_features': valid_features}

def predict(input_data, loaded_data):
    loc_df = loaded_data['loc']
    valid_features = loaded_data['valid_features']
    feature_input = input_data.get('feature', 'default').strip().lower()  # Use 'feature' input
    hour = input_data.get('hour', 0)
    dow = input_data.get('dow', 0)
    month = input_data.get('month', 0)
    
    # Debug input and available data
    print(f"Input: feature={feature_input}, hour={hour}, dow={dow}, month={month}")
    print(f"Sample data head: {loc_df.head().to_dict()}")  # Debug first few rows
    
    # Check if input matches any valid feature (case-insensitive)
    if any(feat.lower() == feature_input for feat in valid_features) and feature_input in loc_df.columns:
        # Find the row matching hour, dow, month for the given feature
        mask = (loc_df['hour'] == hour) & (loc_df['dow'] == dow) & (loc_df['month'] == month)
        if mask.any():
            row = loc_df[mask].iloc[0]
            value = row.get(feature_input, None)
            if pd.notna(value) and isinstance(value, (int, float)):
                predictions = {
                    "Camera Fault": float(value),
                    "Helmet Not Detected": float(value),
                    "Zone Intrusion": float(value)
                }
                status = "match_found"
            else:
                print(f"No valid value at match for {feature_input}: {value}")
                predictions = {"Camera Fault": 0.01, "Helmet Not Detected": 0.01, "Zone Intrusion": 0.01}
                status = "no_valid_value_at_match"
        else:
            print(f"No exact match for hour={hour}, dow={dow}, month={month}")
            # Use the first non-null value from the column
            value = loc_df[feature_input].dropna().iloc[0] if not loc_df[feature_input].isna().all() else None
            if pd.notna(value) and isinstance(value, (int, float)):
                predictions = {
                    "Camera Fault": float(value),
                    "Helmet Not Detected": float(value),
                    "Zone Intrusion": float(value)
                }
                status = "partial_match_using_first_row"
            else:
                print(f"No valid data in {feature_input} column")
                predictions = {"Camera Fault": 0.01, "Helmet Not Detected": 0.01, "Zone Intrusion": 0.01}
                status = "no_data_found_using_defaults"
    else:
        print(f"Invalid feature: {feature_input}")
        predictions = {"Camera Fault": 0.01, "Helmet Not Detected": 0.01, "Zone Intrusion": 0.01}
        status = "no_match_found_using_defaults"
    
    return {
        "feature": feature_input,  # Use 'feature' instead of 'location'
        "hour": hour,
        "dow": dow,
        "month": month,
        "predictions": predictions,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }