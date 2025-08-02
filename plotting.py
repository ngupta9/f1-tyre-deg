import matplotlib.pyplot as plt
import numpy as np
import os
from data_collection import get_circuit_info

def create_plots_directory(circuit_name):
    """
    Create directory structure for saving plots
    """
    # Circuit name for folder (remove special characters)
    circuit_name = circuit_name.replace(" ", "_").replace("ã", "a").replace("ç", "c")
    plots_dir = f"plots/{circuit_name}"
    
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
        
    importance = model.feature_importances_ * 100  # Convert to percentages
    
    plt.figure(figsize=(12, 8))
    indices = np.argsort(importance)[::-1]
    
    # Only show top 8 features for readability
    top_n = min(8, len(importance))
    plt.bar(range(top_n), importance[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45)
    plt.title('Feature Importance in Tire Degradation Model')
    plt.ylabel('Importance Score (%)')
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots and circuit_name:
        plots_dir = create_plots_directory(circuit_name)
        plt.savefig(f"{plots_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {plots_dir}/feature_importance.png")
    
def plot_fuel_load_effect(model_instance, compound_encoder, circuit_name=None, track_temp=30, car_tier=1, save_plots=True):
    """
    Plot the effect of fuel load on lap times for each tire compound
    """
    if model_instance is None or model_instance.model is None:
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
            # Use the model's predict_lap_time method
            lap_time = model_instance.predict_lap_time(compound, tire_age, track_temp, fuel_load, car_tier)
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
    
def plot_race_stint_simulation(model_instance, compound_encoder, circuit_name, track_temp=30, car_tier=1, stint_length=25, save_plots=True):
    """
    Plot realistic race stint showing both tire degradation and fuel burn-off effects
    """
    if model_instance is None or model_instance.model is None:
        print("Model not trained yet!")
        return
        
    if circuit_name is None:
        print("No circuit data available.")
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
            
            # Fuel decreases linearly during stint
            fuel_load = fuel_start_pct - ((fuel_start_pct - fuel_end_pct) * (stint_lap - 1) / (stint_length - 1))
            
            # Use model's predict_lap_time method
            lap_time = model_instance.predict_lap_time(compound, tire_age, track_temp, fuel_load, car_tier)
            
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
    
    # Add annotations for key points (positioned within graph)
    if len(lap_times) > 5:
        y_range = max(lap_times) - min(lap_times)
        y_mid = min(lap_times) + y_range * 0.7  # 70% up from bottom
        
        ax.annotate('Stint Start\n(Heavy Fuel)', xy=(2, lap_times[1]), xytext=(4, y_mid),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.7), fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.annotate('Stint End\n(Light Fuel)', xy=(stint_length-1, lap_times[-2]), xytext=(stint_length-5, y_mid - y_range*0.2),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.7), fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots and circuit_name:
        plots_dir = create_plots_directory(circuit_name)
        plt.savefig(f"{plots_dir}/race_stint_simulation.png", dpi=300, bbox_inches='tight')
        print(f"Saved stint simulation plot to {plots_dir}/race_stint_simulation.png")
    
    plt.show(block=False)  # Non-blocking show

def plot_degradation_curves(model_instance, compound_encoder, circuit_name=None, track_temp=30, save_plots=True):
    """
    Plot tire degradation curves for each compound across car tiers and fuel loads (3x3 grid)
    """
    if model_instance is None or model_instance.model is None:
        print("Model not trained yet!")
        return
        
    tire_ages = np.arange(1, 51)  # 1 to 50 laps to capture HARD tire cliff effects
    
    # Use only the compounds that were actually in the training data
    available_compounds = list(compound_encoder.classes_)
    print(f"Available compounds in data: {available_compounds}")
    
    # Define F1-style colors
    compound_colors = {
        'SOFT': 'red',
        'MEDIUM': 'gold', 
        'HARD': 'lightgray'
    }
    
    # Shading colors (lighter versions)
    shading_colors = {
        'SOFT': 'red',
        'MEDIUM': 'orange', 
        'HARD': 'lightblue'
    }
    
    # Define fuel loads and car tiers
    fuel_loads = [80, 50, 20]  # Heavy, Medium, Light fuel
    fuel_labels = ['80% Fuel (Heavy)', '50% Fuel (Medium)', '20% Fuel (Light)']
    tier_names = ['Tier 1 (Top Teams)', 'Tier 2 (Midfield)', 'Tier 3 (Bottom Teams)']
    
    # Create 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(24, 18), sharey=True)
    
    for fuel_idx, (fuel_load, fuel_label) in enumerate(zip(fuel_loads, fuel_labels)):
        for tier in range(3):
            ax = axes[fuel_idx, tier]
            
            # Calculate lap times for all compounds at all ages
            compound_data = {}
            for compound in available_compounds:
                lap_times = []
                for age in tire_ages:
                    lap_time = model_instance.predict_lap_time(compound, age, track_temp, fuel_load, tier)
                    lap_times.append(lap_time)
                compound_data[compound] = lap_times
            
            # Find fastest compound at each age for shading
            fastest_compound = []
            for i, age in enumerate(tire_ages):
                times_at_age = {comp: times[i] for comp, times in compound_data.items()}
                fastest = min(times_at_age, key=times_at_age.get)
                fastest_compound.append(fastest)
            
            # Add shading for fastest compound sections
            current_fastest = fastest_compound[0]
            section_start = tire_ages[0]
            
            for i, age in enumerate(tire_ages[1:], 1):
                if fastest_compound[i] != current_fastest or i == len(tire_ages) - 1:
                    # End of current section, add shading
                    section_end = tire_ages[i] if i == len(tire_ages) - 1 else tire_ages[i-1]
                    color = shading_colors.get(current_fastest, 'gray')
                    ax.axvspan(section_start, section_end, alpha=0.2, color=color, zorder=0)
                    
                    # Start new section
                    current_fastest = fastest_compound[i]
                    section_start = tire_ages[i-1] if i < len(tire_ages) else tire_ages[i]
            
            # Plot the curves
            for compound in available_compounds:
                color = compound_colors.get(compound, 'blue')
                ax.plot(tire_ages, compound_data[compound], label=f'{compound}', 
                       linewidth=2, color=color)
            
            ax.set_xlabel('Tire Age (Laps)')
            if tier == 0:  # Only leftmost column gets y-label
                ax.set_ylabel('Predicted Lap Time (seconds)')
            
            # Title shows both fuel and tier info
            ax.set_title(f'{tier_names[tier]}\n{fuel_label}\n(Track Temp: {track_temp}°C)')
            
            # Only show legend on the top-left subplot
            if fuel_idx == 0 and tier == 0:
                ax.legend()
            
            ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Tire Degradation Across Fuel Loads and Car Tiers', fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plots and circuit_name:
        plots_dir = create_plots_directory(circuit_name)
        plt.savefig(f"{plots_dir}/tire_degradation_curves.png", dpi=300, bbox_inches='tight')
        print(f"Saved degradation curves plot to {plots_dir}/tire_degradation_curves.png")
    
    plt.show(block=False)  # Non-blocking show