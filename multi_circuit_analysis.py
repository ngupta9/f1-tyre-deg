"""
Multi-Circuit Analysis Script
Run analysis and save plots for multiple F1 circuits automatically
"""

import os
import fastf1
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from tire_model import F1TireModel
from data_collection import collect_data, preprocess_data
from plotting import (plot_feature_importance, plot_race_stint_simulation, plot_degradation_curves, 
                      plot_fuel_load_effect)

# Create cache directory if it doesn't exist
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Enable Fast-F1 cache for faster data loading
fastf1.Cache.enable_cache(cache_dir)

def analyze_circuit(circuit, years):
    """
    Analyze a single circuit and save all plots
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {circuit}")
    print(f"{'='*60}")
    
    # Initialize model
    model = F1TireModel()
    
    # Collect data
    print("Starting data collection...")
    df = collect_data(years, circuit)
    
    # Store circuit name in model
    model.circuit_name = circuit
    
    if len(df) > 0:
        # Preprocess data
        print("\nPreprocessing data...")
        X, y, processed_df = preprocess_data(df)
        
        # Calculate mid-range temperature from actual data
        temp_min = processed_df['track_temp'].min()
        temp_max = processed_df['track_temp'].max()
        mid_temp = int((temp_min + temp_max) / 2)
        print(f"Using mid-range temperature: {mid_temp}°C (range: {temp_min:.1f}°C to {temp_max:.1f}°C)")
        
        # Train model
        print("Training XGBoost model...")
        X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = model.train_model(X, y)
        
        # Generate all plots (automatically saved)
        print("Generating and saving plots...")
        plot_feature_importance(model.model, model.feature_names, circuit_name=circuit)
        plot_degradation_curves(model, model.compound_encoder, circuit_name=circuit, track_temp=mid_temp)
        plot_fuel_load_effect(model, model.compound_encoder, circuit_name=circuit, track_temp=mid_temp)
        plot_race_stint_simulation(model, model.compound_encoder, circuit, track_temp=mid_temp)

        print(f"✅ Analysis complete for {circuit}!")
        return True
    else:
        print(f"❌ No data collected for {circuit}")
        return False

def main():
    """
    Run analysis for multiple circuits
    """
    # Define circuits and years
    circuits = [
        "Silverstone",
        "Monaco", 
        "Spa",
        "Monza",
        "Brazil"
    ]
    
    years = [2022, 2023]
    
    print("F1 Multi-Circuit Tire Degradation Analysis")
    print(f"Circuits to analyze: {', '.join(circuits)}")
    print(f"Years: {years}")
    
    successful_analyses = []
    failed_analyses = []
    
    # Analyze each circuit
    for circuit in circuits:
        try:
            success = analyze_circuit(circuit, years)
            if success:
                successful_analyses.append(circuit)
            else:
                failed_analyses.append(circuit)
        except Exception as e:
            print(f"❌ Error analyzing {circuit}: {e}")
            failed_analyses.append(circuit)
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful_analyses)} circuits")
    for circuit in successful_analyses:
        print(f"   - {circuit}")
    
    if failed_analyses:
        print(f"\n❌ Failed: {len(failed_analyses)} circuits")
        for circuit in failed_analyses:
            print(f"   - {circuit}")
    
    print(f"\nAll plots saved to individual circuit folders in the 'plots/' directory!")

if __name__ == "__main__":
    main()