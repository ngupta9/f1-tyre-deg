import os
import fastf1
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Import modules
from tire_model import F1TireModel
from data_collection import collect_data
from plotting import plot_feature_importance, plot_degradation_curves, plot_fuel_load_effect, plot_race_stint_simulation

# Create cache directory
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable Fast-F1 cache
fastf1.Cache.enable_cache(cache_dir)

def preprocess_data(df):
    """
    Basic preprocessing for the model
    """
    print(f"Data before preprocessing: {len(df)} laps")
    
    # Remove basic outliers (very conservative)
    def remove_outliers(group):
        mean_time = group['lap_time'].mean()
        std_time = group['lap_time'].std()
        # Keep data within 3 standard deviations (very conservative)
        mask = np.abs(group['lap_time'] - mean_time) <= 3 * std_time
        return group[mask]
    
    # Group by compound and tire age for outlier removal
    df = df.groupby(['compound', 'tire_age']).apply(remove_outliers).reset_index(drop=True)
    
    print(f"Data after outlier removal: {len(df)} laps")
    print()  # Add spacing
    
    # Create feature matrix
    feature_cols = ['track_temp', 'fuel_load', 'car_tier', 'compound', 'tire_age']
    X = df[feature_cols].copy()
    y = df['lap_time'].copy()
    
    return X, y, df

def main():
    # Initialize model
    model = F1TireModel()
    
    # Data collection settings
    circuit = "Spa"
    years = [2022, 2023]
    
    # Collect data
    print(f"Collecting data for {circuit} ({years})...")
    df = collect_data(years, circuit)
    
    if len(df) > 0:
        # Preprocess data
        X, y, processed_df = preprocess_data(df)
        
        # Train model
        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = model.train_model(X, y)
        
        # Calculate mid-range temperature from actual data
        temp_min = processed_df['track_temp'].min()
        temp_max = processed_df['track_temp'].max()
        mid_temp = int((temp_min + temp_max) / 2)
        print(f"Using mid-range temperature: {mid_temp}°C (range: {temp_min:.1f}°C to {temp_max:.1f}°C)")
        
        # Generate plots
        print()
        print("Generating plots...")
        plot_feature_importance(model.model, model.feature_names, circuit_name=circuit)
        plot_degradation_curves(model, model.compound_encoder, circuit_name=circuit, track_temp=mid_temp)
        plot_fuel_load_effect(model, model.compound_encoder, circuit_name=circuit, track_temp=mid_temp)
        plot_race_stint_simulation(model, model.compound_encoder, circuit_name=circuit, track_temp=mid_temp)
        
        # Keep plots open
        input("Press Enter to close all plots and exit...")
        plt.close('all')

    else:
        print("❌ No data collected. Check internet connection.")

if __name__ == "__main__":
    main()