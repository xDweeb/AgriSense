import pandas as pd
import json
from datetime import datetime, timedelta

# === Step 1. Read and Sort the Data ===
# Change the file name as needed.
df = pd.read_csv("irrigation_data.csv", parse_dates=["date"])

# Sort by date to ensure proper chronological order
df.sort_values("date", inplace=True)
df.reset_index(drop=True, inplace=True)

# === Step 2. Define the Column Lists ===
# Columns for past 365 days input
past_cols = [
    "humidity", 
    "rainfall", 
    "temperature", 
    "soil_moist", 
    "soil_temp"
]

# Columns for future 7 days input
future_cols = [
    "humidity", 
    "rainfall", 
    "temperature"
]

# Output includes irrigation_volume, irrigation_duration, and irrigation_countdown.

# === Step 3. Process the Data to Create Samples ===
processed_samples = []

# We need a full past window (365 days) and a full future window (7 days).
for i in range(365, len(df) - 7):
    current_row = df.iloc[i]
    
    # Only process days when an irrigation event occurred.
    if pd.isna(current_row["irrigation_time"]) or current_row["irrigation_volume"] == 0:
        continue

    # --- Extract Past 365 Days Data ---
    past_window = df.iloc[i-365:i][past_cols]
    past_data = past_window.values.tolist()
    
    # --- Extract Future 7 Days Data ---
    future_window = df.iloc[i+1:i+8][future_cols]
    future_data = future_window.values.tolist()
    
    # --- Prepare the Output Values ---
    irrigation_volume = current_row["irrigation_volume"]
    irrigation_duration = current_row["irrigation_duration"]
    
    # === Compute Irrigation Countdown Using timedelta ===
    next_irrigation_rows = df[(df.index > i) & (df["irrigation_volume"] > 0)]

    if not next_irrigation_rows.empty:
        next_irrigation_idx = next_irrigation_rows.index.min()
        
        try:
            # Convert current and next irrigation times to full datetime objects
            current_datetime = datetime.combine(current_row["date"], 
                                                datetime.strptime(current_row["irrigation_time"], "%H:%M").time())

            next_irrigation_datetime = datetime.combine(df.loc[next_irrigation_idx, "date"], 
                                                        datetime.strptime(df.loc[next_irrigation_idx, "irrigation_time"], "%H:%M").time())

            # Compute time difference in hours
            irrigation_countdown = (next_irrigation_datetime - current_datetime).total_seconds() / 3600  
        
        except Exception as e:
            continue  # Skip if there's an error parsing time
    else:
        irrigation_countdown = None  # No future irrigation event found
    
    # --- Build the Sample Dictionary ---
    sample = {
    "date": pd.to_datetime(current_row["date"]).strftime("%Y-%m-%d"),
    "past_data": past_data,      # 365 days × 5 features
    "future_data": future_data,  # 7 days × 3 features
    "output": {
        "irrigation_volume": irrigation_volume,
        "irrigation_duration": irrigation_duration,
        "irrigation_countdown": irrigation_countdown
        }
    }
    
    # Append the sample to our list
    processed_samples.append(sample)

# === Step 4. Save the Processed Samples to a JSON File ===
with open("processed_irrigation_data.json", "w") as f:
    json.dump(processed_samples, f, indent=2)

print(f"Processed {len(processed_samples)} samples and saved to 'processed_irrigation_data.json'")
