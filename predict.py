"""
FORMULA 1 PODIUM PREDICTOR INFERENCE ENGINE
Developed for the Paddock ML Challenge.

This script acts as the automated entry point for live race forecasts.
It ingests a raw Qualifying data CSV right after the Saturday lock,
defensively cleans missing or modified features, aligns column configurations,
and prints the finalized 1-2-3 leaderboard.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import ast

def extract_furthest_gap(row, gap_column_name, default_flag=99.0):
    """
    Defensively unpacks OpenF1 single-lap gap layouts.
    Handles raw numeric floats as well as list strings like '[0.125, 0.0, None]'.
    """
    if gap_column_name not in row.index or pd.isna(row[gap_column_name]):
        return default_flag
    val = row[gap_column_name]
    try:
        if isinstance(val, (int, float)):
            return float(val)
        # Safely convert OpenF1 string representations of lists back to Python lists
        gap_list = ast.literal_eval(str(val))
        valid_gaps = [float(x) for x in gap_list if x is not None]
        if len(valid_gaps) > 0:
            return valid_gaps[-1]  # Grab the furthest active lap gap session reached
    except:
        pass
    return default_flag

def main():
    # Verify that the automated platform engine passed the target live data path
    if len(sys.argv) < 2:
        print("❌ Error: No live qualification input CSV file path provided.")
        sys.exit(1)
        
    live_csv_path = sys.argv[1]
    
    try:
        df_incoming = pd.read_csv(live_csv_path)
    except Exception as e:
        print(f"❌ Critical Error reading the data file: {e}")
        sys.exit(1)
        
    df_live = df_incoming.copy()
    
    # =====================================================================
    # DEFENSIVE SAFEGUARD 1: POSITION COLUMN RE-MAPPING
    # =====================================================================
    # If the platform passes the grid order as 'position' instead of 'quali_position'
    if 'position' in df_live.columns and 'quali_position' not in df_live.columns:
        df_live['quali_position'] = df_live['position']
        
    if 'quali_position' not in df_live.columns:
        # Fallback default if completely absent to prevent script crashes
        df_live['quali_position'] = 20
        
    # =====================================================================
    # DEFENSIVE SAFEGUARD 2: QUALIFYING GAPS COLLAPSED
    # =====================================================================
    # Maps 'gap_to_leader' into our clean, single-value float baseline
    if 'gap_to_leader' in df_live.columns:
        df_live['final_quali_gap'] = df_live.apply(lambda r: extract_furthest_gap(r, 'gap_to_leader'), axis=1)
    elif 'quali_Q3_gap' in df_live.columns:
        df_live['final_quali_gap'] = df_live.apply(
            lambda r: extract_furthest_gap(r, 'quali_Q3_gap') if not pd.isna(r['quali_Q3_gap'])
            else (extract_furthest_gap(r, 'quali_Q2_gap') if not pd.isna(r['quali_Q2_gap'])
            else extract_furthest_gap(r, 'quali_Q1_gap')), axis=1
        )
    
    if 'final_quali_gap' not in df_live.columns:
        df_live['final_quali_gap'] = 99.0
    df_live['final_quali_gap'] = df_live['final_quali_gap'].fillna(99.0)

    # =====================================================================
    # DEFENSIVE SAFEGUARD 3: ABSENT SPRINT FEATURE PADDING
    # =====================================================================
    # If it is a normal standard weekend, or if the platform completely drops 
    # sprint inputs, we automatically pad the columns with our verified training flags!
    df_live['sprint_quali_position'] = df_live.get('sprint_quali_position', pd.Series(-99, index=df_live.index)).fillna(-99)
    df_live['sprint_position'] = df_live.get('sprint_position', pd.Series(-99, index=df_live.index)).fillna(-99)
    
    if 'final_sprint_quali_gap' not in df_live.columns:
        if 'sprint_gap_to_leader' in df_live.columns:
            df_live['final_sprint_quali_gap'] = df_live.apply(lambda r: extract_furthest_gap(r, 'sprint_gap_to_leader'), axis=1)
        else:
            df_live['final_sprint_quali_gap'] = 99.0
            
    df_live['final_sprint_quali_gap'] = df_live['final_sprint_quali_gap'].fillna(99.0)
    
    # Automatically compute our engineered race-craft feature delta indicator
    df_live['sprint_position_delta'] = df_live['sprint_quali_position'] - df_live['sprint_position']

    # =====================================================================
    # DEFENSIVE SAFEGUARD 4: LOAD PRE-TRAINED MODEL ENGINE
    # =====================================================================
    # Explicitly looking for your saved 2.65 model binary file inside the repo folder
    model_path = 'f1_regressor_multi_season_2.65.pkl'
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Critical Error loading the model binary ({model_path}): {e}")
        sys.exit(1)

    # =====================================================================
    # DEFENSIVE SAFEGUARD 5: ONE-HOT TEAM CATEGORICAL LAYOUT ALIGNMENT
    # =====================================================================
    features = [
        'quali_position', 'final_quali_gap', 'sprint_quali_position',
        'final_sprint_quali_gap', 'sprint_position', 'sprint_position_delta'
    ]
    X_numeric = df_live[features].copy()
    
    if 'team_name' in df_live.columns:
        X_categorical = pd.get_dummies(df_live['team_name'], drop_first=True)
        X_live = pd.concat([X_numeric, X_categorical], axis=1)
    else:
        X_live = X_numeric
        
    # --- THE ABSOLUTE ANCHOR SHIELD ---
    # Dynamically forces the live matrix to perfectly match the trained model footprint.
    # Missing columns are padded with 0, and extra columns are cleanly discarded.
    X_live = X_live.reindex(columns=model.feature_names_in_, fill_value=0)

    # =====================================================================
    # STEP 6: EXECUTE INFERENCE & PRINT FINAL LOG OUTLOOK
    # =====================================================================
    df_live['predicted_position_score'] = model.predict(X_live)
    sorted_podium = df_live.sort_values(by='predicted_position_score').reset_index(drop=True)
    
    print("=== PODIUM PICK SUBMISSION ===")
    print(f"P1: {int(sorted_podium.loc[0, 'driver_number'])}")
    print(f"P2: {int(sorted_podium.loc[1, 'driver_number'])}")
    print(f"P3: {int(sorted_podium.loc[2, 'driver_number'])}")

if __name__ == "__main__":
    main()
