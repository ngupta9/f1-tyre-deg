import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

class F1TireDegradationModel:
    def __init__(self):
        self.model = None
        self.compound_encoder = LabelEncoder()
        self.feature_names = ['Compound', 'Tire Age', 'Track Temp', 'Fuel Load', 'Car Tier']
        self.circuit_name = None  # Store current circuit name
        
    def train_model(self, X, y, test_size=0.2):
        """
        Train XGBoost model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Set up XGBoost parameters with monotonic constraints
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'early_stopping_rounds': 50,
            # Monotonic constraints: (compound, tire_age, temp, fuel, car_tier)
            # 0 = no constraint, 1 = increasing only, -1 = decreasing only
            'monotone_constraints': (0, 1, 0, 1, 0)
        }
        
        # Train model with early stopping built into params
        self.model = xgb.XGBRegressor(**params)
        
        # Fit with evaluation set
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
        
        print(f"Training MAE: {train_mae:.3f} seconds")
        print(f"Test MAE: {test_mae:.3f} seconds")
        print(f"Training R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        
        return X_train, X_test, y_train, y_test, y_pred_train, y_pred_test
    
    def predict_lap_time(self, compound, tire_age, track_temp, fuel_load, car_tier=1):
        """
        Predict lap time for given conditions
        car_tier: 0=Tier 1 (top), 1=Tier 2 (mid), 2=Tier 3 (bottom)
        """
        if self.model is None:
            print("Model not trained yet!")
            return None
            
        # Use the same encoding as training data
        try:
            compound_encoded = self.compound_encoder.transform([compound.upper()])[0]
        except ValueError:
            print(f"Compound '{compound}' not found. Available: {list(self.compound_encoder.classes_)}")
            return None
        
        # Create feature array in same order as training (simplified)
        X_pred = np.array([[
            compound_encoded, tire_age, track_temp, fuel_load, car_tier
        ]])
        
        prediction = self.model.predict(X_pred)[0]
        return prediction