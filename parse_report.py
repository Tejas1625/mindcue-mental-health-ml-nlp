import pandas as pd
import json
import os
import traceback

# Define file paths
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT_IN_PATH = os.path.join(APP_ROOT, "reports", "emissions.csv")
DATA_OUT_PATH = os.path.join(APP_ROOT, "static", "emissions_data.json")


def parse_and_save():
    """
    Reads the latest CodeCarbon emissions report and saves key metrics to a JSON file.
    """
    if not os.path.exists(REPORT_IN_PATH):
        print(f"Emission report not found at {REPORT_IN_PATH}. Run train.py first.")
        # Create a dummy file so the web app doesn't crash
        data = {"error": "No report found. Train model to generate."}
    else:
        try:
            # Read the CSV and get the last row (the summary of the latest run)
            df = pd.read_csv(REPORT_IN_PATH)
            latest_run = df.iloc[-1]

            # Extract key metrics using .get() for safety and standard column names
            # The 'emissions' column is in kg, so we convert to grams for readability.
            emissions_val = latest_run.get('emissions', 0) * 1000

            data = {
                "duration_seconds": f"{latest_run.get('duration', 0):.2f} s",
                "emissions_g": f"{emissions_val:.4f} g",
                "energy_consumed_kwh": f"{latest_run.get('energy_consumed', 0):.6f} kWh",
                "cpu_model": latest_run.get('cpu_model', 'N/A'),
                "country_name": latest_run.get('country_name', 'N/A'),
                "timestamp": latest_run.get('timestamp', 'N/A')
            }
            print("Successfully parsed emissions report.")
        except Exception:
            print(f"Error parsing emissions report. Please check the file format.")
            print(traceback.format_exc())
            data = {"error": "Could not parse report. The format may have changed."}

    # Save the data to a JSON file in the static folder
    os.makedirs(os.path.dirname(DATA_OUT_PATH), exist_ok=True)
    with open(DATA_OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Emissions data saved to {DATA_OUT_PATH}")


if __name__ == "__main__":
    parse_and_save()

