import matplotlib.pyplot as plt
import numpy as np
import os
from data_collection import get_circuit_info

def create_plots_directory(circuit_name):
    """
    Create directory structure for saving plots
    """
    # Clean circuit name for folder (remove special characters)
    clean_circuit_name = circuit_name.replace(" ", "_").replace("ã", "a").replace("ç", "c")
    plots_dir = f"plots/{clean_circuit_name}"
    
    # Create directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    return plots_dir

def plot_feature_importance(model, feature_names, circuit_name=None, save_plots=True):
    """
    Plot feature importance
    """
    if model is None:
        print("Model not trained yet!")
        return
        
    importance = model.feature_importances_
    
    plt.figure(figsize=(12, 8))
    indices = np.argsort(importance)[::-1]
    
    # Only show top 8 features for readability
    top_n = min(8, len(importance))
    plt.bar(range(top_n), importance[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45)
    plt.title('Feature Importance in Tire Degradation Model')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots and circuit_name:
        plots_dir = create_plots_directory(circuit_name)
        plt.savefig(f"{plots_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {plots_dir}/feature_importance.png")
    
    plt.show()

def plot_race_stint_simulation(model, compound_encoder, circuit_name, track_temp=30, car_tier=1, stint_length=25, save_plots=True):
    """
    Plot realistic race stint showing both tire degradation and fuel burn-off effects
    """
    if model is None:
        print("Model not trained yet!")
        return
        
    if circuit_name is None:
        print("No circuit data available. Run collect_data() first.")
        return
        
    # Use only the compounds that were actually in the training data
    available_compounds = list(compound_encoder.classes_)
    
    # Define F1-style colors
    compound_colors = {
        'SOFT': 'red',
        'MEDIUM': 'gold', 
        'HARD': 'lightgray'
    }
    
    # Get circuit-specific fuel information
    race_distance, fuel_per_lap = get_circuit_info(circuit_name)
    
    # Calculate realistic fuel burn for this circuit during stint
    fuel_burned_in_stint = stint_length * fuel_per_lap  # e.g., 25 laps × 3.3 kg/lap = 82.5kg
    fuel_burned_pct = (fuel_burned_in_stint / 110) * 100  # Convert to percentage of 110kg tank
    
    # Set realistic fuel levels for mid-race stint
    fuel_start_pct = min(70, 50 + fuel_burned_pct)  # Start with enough fuel + buffer
    fuel_end_pct = max(5, fuel_start_pct - fuel_burned_pct)  # End with minimum 5%
    
    print(f"Stint simulation for {circuit_name}: {fuel_start_pct:.1f}% → {fuel_end_pct:.1f}% fuel ({fuel_burned_pct:.1f}% burned)")
    
    # Create subplot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    for compound in available_compounds:
        tire_ages = []
        fuel_loads = []
        lap_times = []
        
        for stint_lap in range(1, stint_length + 1):
            # Tire age increases linearly
            tire_age = stint_lap
            
            # Fuel decreases linearly during stint (circuit-specific burn rate)
            fuel_load = fuel_start_pct - ((fuel_start_pct - fuel_end_pct) * (stint_lap - 1) / (stint_length - 1))
            
            # Create features (simplified)
            compound_encoded = compound_encoder.transform([compound])[0]
            
            X_pred = np.array([[
                compound_encoded, tire_age, track_temp, fuel_load, car_tier
            ]])
            
            lap_time = model.predict(X_pred)[0]
            
            tire_ages.append(tire_age)
            fuel_loads.append(fuel_load)
            lap_times.append(lap_time)
        
        color = compound_colors.get(compound, 'blue')
        ax.plot(tire_ages, lap_times, label=f'{compound} Compound', 
               linewidth=3, color=color, marker='o', markersize=4)
    
    # Add fuel load as secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(tire_ages, fuel_loads, '--', color='gray', alpha=0.7, linewidth=2, label='Fuel Load')
    ax2.set_ylabel('Fuel Load (%)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Customize plot
    ax.set_xlabel('Tire Age (Laps into Stint)')
    ax.set_ylabel('Predicted Lap Time (seconds)')
    ax.set_title(f'Race Stint Simulation: Tire Degradation + Fuel Burn-off\n(Circuit: {circuit_name}, Track Temp: {track_temp}°C, Car Tier: {car_tier+1}, Stint Length: {stint_length} laps)')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key points
    ax.annotate('Stint Start\n(Heavy Fuel)', xy=(2, max(lap_times[:3])), xytext=(4, max(lap_times) + 0.1),
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.7), fontsize=10)
    ax.annotate('Stint End\n(Light Fuel)', xy=(stint_length-1, min(lap_times[-3:])), xytext=(stint_length-5, min(lap_times) - 0.1),
               arrowprops=dict(arrowstyle='->', color='black', alpha=0.7), fontsize=10)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots and circuit_name:
        plots_dir = create_plots_directory(circuit_name)
        plt.savefig(f"{plots_dir}/race_stint_simulation.png", dpi=300, bbox_inches='tight')
        print(f"Saved stint simulation plot to {plots_dir}/race_stint_simulation.png")
    
    plt.show()

def plot_degradation_curves(model, compound_encoder, circuit_name=None, track_temp=30, fuel_load=50, save_plots=True):
    """
    Plot tire degradation curves for each compound and car tier
    """
    if model is None:
        print("Model not trained yet!")
        return
        
    tire_ages = np.arange(1, 31)  # 1 to 30 laps (realistic race stint length)
    
    # Use only the compounds that were actually in the training data
    available_compounds = list(compound_encoder.classes_)
    print(f"Available compounds in data: {available_compounds}")
    
    # Define F1-style colors
    compound_colors = {
        'SOFT': 'red',
        'MEDIUM': 'gold', 
        'HARD': 'lightgray'
    }
    
    # Create subplots for each car tier with shared y-axis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    tier_names = ['Tier 1 (Top Teams)', 'Tier 2 (Midfield)', 'Tier 3 (Bottom Teams)']
    
    for tier in range(3):
        ax = axes[tier]
        
        for compound in available_compounds:
            lap_times = []
            for age in tire_ages:
                # Create features with specific car tier (simplified)
                compound_encoded = compound_encoder.transform([compound])[0]
                
                X_pred = np.array([[
                    compound_encoded, age, track_temp, fuel_load, tier
                ]])
                
                lap_time = model.predict(X_pred)[0]
                lap_times.append(lap_time)
            
            color = compound_colors.get(compound, 'blue')
            ax.plot(tire_ages, lap_times, label=f'{compound} Compound', 
                   linewidth=2, color=color)
        
        ax.set_xlabel('Tire Age (Laps)')
        ax.set_ylabel('Predicted Lap Time (seconds)')
        ax.set_title(f'{tier_names[tier]}\n(Track Temp: {track_temp}°C, Fuel Load: {fuel_load}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots and circuit_name:
        plots_dir = create_plots_directory(circuit_name)
        plt.savefig(f"{plots_dir}/tire_degradation_curves.png", dpi=300, bbox_inches='tight')
        print(f"Saved degradation curves plot to {plots_dir}/tire_degradation_curves.png")
    
    plt.show()

def plot_fuel_load_effect(model, compound_encoder, circuit_name=None, track_temp=30, car_tier=1, save_plots=True):
    """
    Plot the effect of fuel load on lap times for each tire compound
    """
    if model is None:
        print("Model not trained yet!")
        return
        
    tire_age = 10  # Fixed tire age for comparison
    fuel_loads = [20, 50, 80]  # Light, Medium, Heavy fuel
    fuel_labels = ['Light Fuel (20%)', 'Medium Fuel (50%)', 'Heavy Fuel (80%)']
    
    # Use only the compounds that were actually in the training data
    available_compounds = list(compound_encoder.classes_)
    
    # Define F1-style colors
    compound_colors = {
        'SOFT': 'red',
        'MEDIUM': 'gold', 
        'HARD': 'lightgray'
    }
    
    # Create subplot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # For each compound, show bars for different fuel loads
    x_positions = np.arange(len(available_compounds))
    bar_width = 0.25
    
    for i, fuel_load in enumerate(fuel_loads):
        lap_times = []
        
        for compound in available_compounds:
            # Create features (simplified)
            compound_encoded = compound_encoder.transform([compound])[0]
            
            X_pred = np.array([[
                compound_encoded, tire_age, track_temp, fuel_load, car_tier
            ]])
            
            lap_time = model.predict(X_pred)[0]
            lap_times.append(lap_time)
        
        # Plot bars for this fuel load
        bars = ax.bar(x_positions + i * bar_width, lap_times, 
                     bar_width, label=fuel_labels[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, lap_time in zip(bars, lap_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{lap_time:.2f}s', ha='center', va='bottom', fontsize=10)
    
    # Customize plot
    ax.set_xlabel('Tire Compound')
    ax.set_ylabel('Predicted Lap Time (seconds)')
    ax.set_title(f'Fuel Load Effect on Lap Times\n(Tire Age: {tire_age} laps, Track Temp: {track_temp}°C, Car Tier: {car_tier+1})')
    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels(available_compounds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots and circuit_name:
        plots_dir = create_plots_directory(circuit_name)
        plt.savefig(f"{plots_dir}/fuel_load_effect.png", dpi=300, bbox_inches='tight')
        print(f"Saved fuel load effect plot to {plots_dir}/fuel_load_effect.png")
    
    plt.show()