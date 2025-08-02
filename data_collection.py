import fastf1
import pandas as pd
import numpy as np

def get_circuit_info(circuit_name):
    """
    Get circuit-specific information derived from real F1 data
    """
    circuit_data = {
        # Circuit: (race_distance_laps, fuel_per_lap_kg)
        # Data derived from actual F1 race analysis
        "Silverstone": (52, 3.3),
        "Monaco": (71, 3.1),
        "Spa": (66, 3.3),
        "Monza": (49, 3.1),
        "Brazil": (71, 2.9),
        "Austria": (71, 2.2),
        "Hungary": (70, 3.3),
        "Netherlands": (72, 3.0),
        "Singapore": (60, 3.6),
        "Japan": (40, 3.5),
        "Abu Dhabi": (58, 3.2),
        "Bahrain": (57, 3.4),
        "Australia": (55, 3.2),
    }
    
    # Default values if circuit not found
    default_laps = 60
    default_fuel_per_lap = 2.8
    
    return circuit_data.get(circuit_name, (default_laps, default_fuel_per_lap))

def get_car_tier(driver, year):
    """
    Assign car tier based on constructor championship standings
    """
    # Driver to team mapping for 2022-2023
    driver_to_team = {
        # 2022 teams
        'VER': 'Red Bull', 'PER': 'Red Bull',
        'LEC': 'Ferrari', 'SAI': 'Ferrari',
        'HAM': 'Mercedes', 'RUS': 'Mercedes',
        'ALO': 'Alpine', 'OCO': 'Alpine',
        'NOR': 'McLaren', 'RIC': 'McLaren', 'PIA': 'McLaren',
        'BOT': 'Alfa Romeo', 'ZHO': 'Alfa Romeo',
        'MAG': 'Haas', 'MSC': 'Haas', 'HUL': 'Haas',
        'GAS': 'AlphaTauri', 'TSU': 'AlphaTauri', 'DEV': 'AlphaTauri', 'LAW': 'AlphaTauri',
        'VET': 'Aston Martin', 'STR': 'Aston Martin', 'ALB': 'Williams', 'LAT': 'Williams',
        # 2023 updates
        'SAR': 'Williams'  # Sargeant
    }
    
    # Constructor championship standings by year
    team_tiers = {
        2022: {
            'Tier 1': ['Red Bull', 'Ferrari', 'Mercedes'],
            'Tier 2': ['Alpine', 'McLaren', 'Alfa Romeo', 'Haas'],
            'Tier 3': ['AlphaTauri', 'Aston Martin', 'Williams']
        },
        2023: {
            'Tier 1': ['Red Bull', 'Mercedes', 'Ferrari'],
            'Tier 2': ['Aston Martin', 'McLaren', 'Alpine', 'AlphaTauri'],
            'Tier 3': ['Williams', 'Alfa Romeo', 'Haas']
        }
    }
    
    # Get team for driver
    team = driver_to_team.get(driver, 'Unknown')
    
    # Get tier for team and year
    year_tiers = team_tiers.get(year, team_tiers[2022])  # Default to 2022 if year not found
    
    for tier_name, teams in year_tiers.items():
        if team in teams:
            if tier_name == 'Tier 1':
                return 0
            elif tier_name == 'Tier 2':
                return 1
            else:  # Tier 3
                return 2
    
    # Default to Tier 2 (midfield) if unknown
    return 1

def collect_data(years, circuit_name="São Paulo"):
    """
    Collect F1 data for specified years and circuit
    """
    all_data = []
    
    # Get circuit-specific information
    race_distance, fuel_per_lap = get_circuit_info(circuit_name)
    starting_fuel = 110  # kg - standard F1 fuel capacity
    
    print(f"Circuit info for {circuit_name}:")
    print(f"  Race distance: {race_distance} laps") 
    print(f"  Fuel consumption: {fuel_per_lap} kg/lap")
    
    for year in years:
        print(f"\nCollecting data for {year} {circuit_name}...")
        
        try:
            # Load race session only - practice data is too noisy
            try:
                session = fastf1.get_session(year, circuit_name, 'R')
            except Exception:
                # Try alternative circuit names
                alt_names = {
                    "São Paulo": ["Brazil", "Brazilian Grand Prix", "Interlagos"],
                    "Monaco": ["Monaco Grand Prix"],
                    "Silverstone": ["British Grand Prix", "Great Britain"],
                    "Spa": ["Belgian Grand Prix", "Belgium"],
                    "Monza": ["Italian Grand Prix", "Italy"]
                }
                
                session = None
                if circuit_name in alt_names:
                    for alt_name in alt_names[circuit_name]:
                        try:
                            session = fastf1.get_session(year, alt_name, 'R')
                            print(f"  Found using name: {alt_name}")
                            break
                        except:
                            continue
                
                if session is None:
                    print(f"  Could not find circuit '{circuit_name}' for {year}")
                    continue
            
            print(f"  Loading session data...")
            session.load()
            print(f"  Session loaded successfully!")
            
            # Get weather data
            weather = session.weather_data
            print(f"  Weather data: {len(weather)} records")
            
            # Get lap data for all drivers
            laps = session.laps
            print(f"  Total laps: {len(laps)}")
            
            # Filter valid laps (no pit stops, no safety cars, etc.)
            valid_laps = laps.pick_wo_box().pick_accurate()
            print(f"  Valid laps after filtering: {len(valid_laps)}")
            
            processed_count = 0
            
            for _, lap in valid_laps.iterrows():
                try:
                    # Get tire compound
                    compound = lap['Compound']
                    if pd.isna(compound):
                        continue
                    
                    # Normalize compound names
                    compound = str(compound).upper()
                    if compound not in ['SOFT', 'MEDIUM', 'HARD']:
                        # Try to map common alternatives
                        compound_map = {
                            'S': 'SOFT', 'M': 'MEDIUM', 'H': 'HARD',
                            'SOFTS': 'SOFT', 'MEDIUMS': 'MEDIUM', 'HARDS': 'HARD'
                        }
                        compound = compound_map.get(compound, compound)
                        if compound not in ['SOFT', 'MEDIUM', 'HARD']:
                            continue
                        
                    # Get tire age (number of laps on this tire)
                    tire_age = lap['TyreLife']
                    if pd.isna(tire_age) or tire_age < 1:
                        continue
                        
                    # Get lap time in seconds - simple outlier filtering
                    lap_time = lap['LapTime']
                    if pd.isna(lap_time) or lap_time.total_seconds() > 200:
                        continue
                    lap_time_seconds = lap_time.total_seconds()
                        
                    # Get track temperature (interpolate from weather data)
                    if len(weather) > 0:
                        lap_time_stamp = lap['Time']
                        closest_weather = weather.iloc[(weather['Time'] - lap_time_stamp).abs().argsort()[:1]]
                        track_temp = closest_weather['TrackTemp'].iloc[0]
                    else:
                        track_temp = 25  # Default temperature if no weather data
                    
                    # Dynamic fuel calculation based on circuit
                    # Calculate fuel remaining based on race progress
                    race_progress = lap['LapNumber'] / race_distance
                    fuel_used = race_progress * starting_fuel
                    fuel_remaining = max(5, starting_fuel - fuel_used)  # Minimum 5kg
                    
                    # Convert to percentage for consistency (5kg = 0%, 110kg = 100%)
                    fuel_load = ((fuel_remaining - 5) / 105) * 100
                    
                    all_data.append({
                        'year': year,
                        'compound': compound,
                        'tire_age': tire_age,
                        'track_temp': track_temp,
                        'fuel_load': fuel_load,
                        'lap_time': lap_time_seconds,
                        'lap_number': lap['LapNumber'],
                        'driver': lap['Driver'],
                        'car_tier': get_car_tier(lap['Driver'], year)
                    })
                    processed_count += 1
                    
                except Exception as e:
                    continue
            
            print(f"  Successfully processed {processed_count} laps from {year}")
                    
        except Exception as e:
            print(f"  Error loading {year} data: {e}")
            continue
            
    df = pd.DataFrame(all_data)
    print(f"\nCollected {len(df)} valid laps total")
    if len(df) > 0:
        print(f"Compounds found: {df['compound'].unique()}")
        print(f"Years: {df['year'].unique()}")
        # Show car tier distribution
        tier_counts = df['car_tier'].value_counts().sort_index()
        tier_names = {0: 'Tier 1 (Top)', 1: 'Tier 2 (Mid)', 2: 'Tier 3 (Bottom)'}
        print("Car tier distribution:")
        for tier, count in tier_counts.items():
            print(f"  {tier_names.get(tier, f'Tier {tier}')}: {count} laps")
        
        # Investigate data ranges to understand feature importance issues
        print("\nData ranges (to understand feature importance):")
        print(f"Track temperature: {df['track_temp'].min():.1f}°C to {df['track_temp'].max():.1f}°C (range: {df['track_temp'].max() - df['track_temp'].min():.1f}°C)")
        print(f"Fuel load: {df['fuel_load'].min():.1f}% to {df['fuel_load'].max():.1f}% (range: {df['fuel_load'].max() - df['fuel_load'].min():.1f}%)")
        print(f"Tire age: {df['tire_age'].min()} to {df['tire_age'].max()} laps (range: {df['tire_age'].max() - df['tire_age'].min()} laps)")
        print(f"Lap times: {df['lap_time'].min():.2f}s to {df['lap_time'].max():.2f}s (range: {df['lap_time'].max() - df['lap_time'].min():.2f}s)")
        
        # Check correlations
        print(f"\nCorrelations with lap time:")
        print(f"Track temp: {df['track_temp'].corr(df['lap_time']):.3f}")
        print(f"Fuel load: {df['fuel_load'].corr(df['lap_time']):.3f}")
        print(f"Tire age: {df['tire_age'].corr(df['lap_time']):.3f}")
        
        # Check data distribution by car tier and compound
        print(f"\nData distribution by car tier and compound:")
        compound_tier_dist = df.groupby(['car_tier', 'compound']).size().unstack(fill_value=0)
        tier_names = {0: 'Tier 1', 1: 'Tier 2', 2: 'Tier 3'}
        compound_tier_dist.index = [tier_names.get(i, f'Tier {i}') for i in compound_tier_dist.index]
        print(compound_tier_dist)
        
    return df

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