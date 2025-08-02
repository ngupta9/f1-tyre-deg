import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

class F1TireModel:
    def __init__(self):
        self.model = None
        self.compound_encoder = LabelEncoder()
        self.feature_names = ['track_temp', 'fuel_load', 'car_tier', 'compound_encoded', 'tire_age_squared']
        
    def add_synthetic_cliff_data(self, X, y):
        """
        Add focused synthetic data to ensure cliff effects are learned
        """
        synthetic_data = []
        
        # Extract actual data ranges for realistic synthetic points
        temp_min = X['track_temp'].min()
        temp_max = X['track_temp'].max()
        fuel_min = X['fuel_load'].min()
        fuel_max = X['fuel_load'].max()
        
        # Use actual data ranges for synthetic points
        temp_values = [
            temp_min + 0.2 * (temp_max - temp_min),  # 20th percentile
            temp_min + 0.5 * (temp_max - temp_min),  # 50th percentile (median)
            temp_min + 0.8 * (temp_max - temp_min)   # 80th percentile
        ]
        
        fuel_values = [
            fuel_min + 0.2 * (fuel_max - fuel_min),  # Light fuel
            fuel_min + 0.5 * (fuel_max - fuel_min),  # Medium fuel
            fuel_min + 0.8 * (fuel_max - fuel_min)   # Heavy fuel
        ]
        
        # Get baseline times for each compound (only for compounds that exist in the data)
        available_compounds = list(set(X['compound'].unique()))  # Remove any duplicates
        compound_baselines = {}
        
        for compound in available_compounds:
            # Find data for this compound with low tire age
            compound_mask = (X['compound_encoded'] == self.compound_encoder.transform([compound])[0])
            fresh_mask = (X['tire_age'] <= 10)
            compound_fresh = X[compound_mask & fresh_mask]
            
            if len(compound_fresh) > 0:
                baseline_indices = compound_fresh.index
                compound_baselines[compound] = y[baseline_indices].median()
            else:
                # Fallback baseline
                compound_baselines[compound] = y.median()
        
        # Define cliff characteristics for each compound (only use available compounds)
        all_cliff_specs = {
            'SOFT': {
                'cliff_start': 18,        # Cliff starts at lap 18
                'cliff_severity': 0.08,   # Moderate cliff (not too extreme)
                'max_penalty': 4.0        # Maximum penalty
            },
            'MEDIUM': {
                'cliff_start': 28,        # Later cliff
                'cliff_severity': 0.05,   # Gentler cliff
                'max_penalty': 2.0        # Lower maximum penalty
            },
            'HARD': {
                'cliff_start': 40,        # Very late cliff
                'cliff_severity': 0.03,   # Very gentle cliff
                'max_penalty': 1.0        # Minimal penalty
            }
        }
        
        # Only use specs for compounds that exist in the data
        cliff_specs = {compound: all_cliff_specs[compound] 
                      for compound in available_compounds 
                      if compound in all_cliff_specs}
        
        # Add synthetic cliff points
        synthetic_breakdown = {}
        
        for compound, specs in cliff_specs.items():
            compound_encoded = self.compound_encoder.transform([compound])[0]
            base_time = compound_baselines.get(compound, y.median())  # Use real data baseline directly
            
            compound_synthetic_count = 0
            ages_added = []
            
            # Add cliff effect points only where real data is sparse
            for age in range(specs['cliff_start'], specs['cliff_start'] + 20, 2):
                
                # Check if sufficient real data exists for this compound and age
                compound_mask = (X['compound_encoded'] == compound_encoded)
                age_mask = (X['tire_age'] >= age - 1) & (X['tire_age'] <= age + 1)  # ±1 lap tolerance
                existing_data_count = len(X[compound_mask & age_mask])
                
                # Only add synthetic data if real data is sparse (less than 5 samples)
                if existing_data_count < 5:
                    laps_past_cliff = age - specs['cliff_start']
                    cliff_penalty = min(
                        specs['cliff_severity'] * (laps_past_cliff ** 1.3),
                        specs['max_penalty']
                    )
                    degraded_time = base_time + cliff_penalty
                    
                    ages_added.append(age)
                    
                    # Add points using actual data ranges
                    for temp in temp_values:
                        for fuel in fuel_values:
                            for tier in [0, 1, 2]:
                                synthetic_data.append({
                                    'track_temp': temp,
                                    'fuel_load': fuel,
                                    'car_tier': tier,
                                    'compound_encoded': compound_encoded,
                                    'tire_age': age,
                                    'tire_age_squared': age ** 2,
                                    'lap_time': degraded_time + np.random.normal(0, 0.06)
                                })
                                compound_synthetic_count += 1
            
            # Store breakdown info (ensure no duplicates)
            if compound_synthetic_count > 0 and compound not in synthetic_breakdown:
                synthetic_breakdown[compound] = {
                    'count': compound_synthetic_count,
                    'ages': ages_added,
                    'age_range': f"{min(ages_added)}-{max(ages_added)}" if ages_added else "None"
                }
        
        if synthetic_data:
            synthetic_df = pd.DataFrame(synthetic_data)
            
            # Combine with original data
            X_synthetic = synthetic_df.drop('lap_time', axis=1)
            y_synthetic = synthetic_df['lap_time']
            
            X_combined = pd.concat([X, X_synthetic], ignore_index=True)
            y_combined = pd.concat([y, y_synthetic], ignore_index=True)
            
            # Print detailed breakdown
            print(f"Added {len(synthetic_data)} synthetic degradation points:")
            for compound, info in synthetic_breakdown.items():
                print(f"  {compound}: {info['count']} points (ages {info['age_range']})")
            print()
            
            return X_combined, y_combined
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        """
        Train the  5-feature model
        """
        # Create tire_age_squared feature
        X_processed = X.copy()
        X_processed['tire_age_squared'] = X['tire_age'] ** 2
        
        # Encode compounds
        X_processed['compound_encoded'] = self.compound_encoder.fit_transform(X_processed['compound'])
        
        # Add synthetic cliff data
        X_augmented, y_augmented = self.add_synthetic_cliff_data(X_processed, y)
        
        # Select the 5 features
        feature_cols = ['track_temp', 'fuel_load', 'car_tier', 'compound_encoded', 'tire_age_squared']
        X_final = X_augmented[feature_cols]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_augmented, test_size=test_size, random_state=42, shuffle=True
        )
        
        # XGBoost parameters optimized for tire physics
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,           # Sufficient depth for interactions
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'early_stopping_rounds': 50,
            # Monotonic constraints: temp(no), fuel(yes), car_tier(no), compound(no), age²(yes)
            'monotone_constraints': (0, 1, 0, 0, 1),
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        # Train model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print("Model Performance:")
        print(f"Training MAE: {train_mae:.3f} seconds")
        print(f"Test MAE: {test_mae:.3f} seconds")
        print(f"Training R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        
        # Show feature importance for model
        feature_importance = self.model.feature_importances_
        importance_dict = dict(zip(feature_cols, feature_importance))
        print()
        print("Feature Importance:")
        for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance*100:.1f}%")
        print()
        
        return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
    
    def predict_lap_time(self, compound, tire_age, track_temp, fuel_load, car_tier=1):
        """
        Predict lap time for given conditions
        """
        if self.model is None:
            return None
            
        try:
            compound_encoded = self.compound_encoder.transform([compound.upper()])[0]
        except ValueError:
            return None
        
        # Create feature array
        tire_age_squared = tire_age ** 2
        X_pred = np.array([[track_temp, fuel_load, car_tier, compound_encoded, tire_age_squared]])
        
        prediction = self.model.predict(X_pred)[0]
        return prediction