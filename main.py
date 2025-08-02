import os
import fastf1
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from model import F1TireDegradationModel
from data_collection import collect_data, preprocess_data
from plotting import (plot_feature_importance, plot_race_stint_simulation, 
                     plot_degradation_curves, plot_fuel_load_effect)

# Create cache directory if it doesn't exist
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable Fast-F1 cache for faster data loading
fastf1.Cache.enable_cache(cache_dir)

def main():
    # Initialize model
    model = F1TireDegradationModel()
    
    # Define circuit and years once
    circuit = "Silverstone"  # Example circuit
    years = [2022, 2023]  # Recent years
    
    # Collect data (this might take a while the first time)
    print("Starting data collection...")
    df = collect_data(years, circuit)
    
    # Store circuit name in model for plotting functions
    model.circuit_name = circuit
    
    if len(df) > 0:
        # Preprocess data
        print("\nPreprocessing data...")
        X, y, processed_df = preprocess_data(df, model.compound_encoder)
        
        # Calculate mid-range temperature from actual data
        temp_min = processed_df['track_temp'].min()
        temp_max = processed_df['track_temp'].max()
        mid_temp = int((temp_min + temp_max) / 2)
        print(f"\nUsing mid-range temperature: {mid_temp}°C (range: {temp_min:.1f}°C to {temp_max:.1f}°C)")
        
        # Train model
        print("\nTraining XGBoost model...")
        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = model.train_model(X, y)
        
        # Show feature importance
        print("\nShowing feature importance...")
        plot_feature_importance(model.model, model.feature_names, circuit_name=circuit)
        
        # Show degradation curves for all tiers
        print("\nShowing degradation curves...")
        plot_degradation_curves(model.model, model.compound_encoder, circuit_name=circuit, 
                               track_temp=mid_temp, fuel_load=50)
        
        # Show fuel load effect comparison
        print("\nShowing fuel load effects...")
        plot_fuel_load_effect(model.model, model.compound_encoder, circuit_name=circuit,
                             track_temp=mid_temp, car_tier=1)  # Tier 2 (midfield) team
        
        # Show realistic race stint simulation
        print("\nShowing race stint simulation...")
        plot_race_stint_simulation(model.model, model.compound_encoder, circuit, 
                                 track_temp=mid_temp, car_tier=1, stint_length=25)

        # Example predictions for different tiers
        print("\nExample predictions:")
        available_compounds = list(model.compound_encoder.classes_)
        
        if 'SOFT' in available_compounds:
            print(f"Soft tire, 10 laps, {mid_temp}°C, 70% fuel:")
            print(f"  Tier 1 team: {model.predict_lap_time('SOFT', 10, mid_temp, 70, 0):.2f}s")
            print(f"  Tier 2 team: {model.predict_lap_time('SOFT', 10, mid_temp, 70, 1):.2f}s")
            print(f"  Tier 3 team: {model.predict_lap_time('SOFT', 10, mid_temp, 70, 2):.2f}s")
        
        if 'MEDIUM' in available_compounds:
            print(f"Medium tire, 15 laps, {mid_temp+2}°C, 40% fuel:")
            print(f"  Tier 1 team: {model.predict_lap_time('MEDIUM', 15, mid_temp+2, 40, 0):.2f}s")
            print(f"  Tier 2 team: {model.predict_lap_time('MEDIUM', 15, mid_temp+2, 40, 1):.2f}s")
            print(f"  Tier 3 team: {model.predict_lap_time('MEDIUM', 15, mid_temp+2, 40, 2):.2f}s")
        
        if 'HARD' in available_compounds:
            print(f"Hard tire, 20 laps, {mid_temp-1}°C, 30% fuel:")
            print(f"  Tier 1 team: {model.predict_lap_time('HARD', 20, mid_temp-1, 30, 0):.2f}s")
            print(f"  Tier 2 team: {model.predict_lap_time('HARD', 20, mid_temp-1, 30, 1):.2f}s")
            print(f"  Tier 3 team: {model.predict_lap_time('HARD', 20, mid_temp-1, 30, 2):.2f}s")
        
        print(f"\nAll plots have been saved to the 'plots/{circuit.replace(' ', '_')}' directory!")

    else:
        print("No data collected. Check your internet connection and try again.")

if __name__ == "__main__":
    main()