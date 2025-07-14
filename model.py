import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class SalesPredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def preprocess_data(self, df):
        # Aggregate sales by product and day
        agg_df = df.groupby(['Day_Count', 'Day', 'ProductName', 'weather', 
                            'Inflation_Percentage', 'Unemployment_Percentage', 
                            'Promo_applied']).agg({
            'Amt': 'sum',  # Total sales across stores
            'Product_Calorie': 'first'
        }).reset_index()
        
        # Feature engineering
        agg_df['weekend'] = agg_df['Day'].isin(['Saturday', 'Sunday']).astype(int)
        agg_df['promo_flag'] = (agg_df['Promo_applied'].notna()).astype(int)
        
        # Encode categorical variables
        categorical_cols = ['Day', 'ProductName', 'weather']
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                agg_df[f'{col}_encoded'] = self.encoders[col].fit_transform(agg_df[col])
            else:
                agg_df[f'{col}_encoded'] = self.encoders[col].transform(agg_df[col])
        
        return agg_df
    
    def train(self, df):
        processed_df = self.preprocess_data(df)
        
        # Define features
        self.feature_columns = [
            'Day_encoded', 'ProductName_encoded', 'weather_encoded',
            'Product_Calorie', 'Inflation_Percentage', 'Unemployment_Percentage',
            'weekend', 'promo_flag'
        ]
        
        X = processed_df[self.feature_columns]
        y = processed_df['Amt']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        print(f"Train R²: {r2_score(y_train, train_pred):.3f}")
        print(f"Test R²: {r2_score(y_test, test_pred):.3f}")
        print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.2f}")
        
        return self.model
    
    def predict(self, day, product, weather, calories, inflation, unemployment, promo):
        # Create input DataFrame
        input_data = pd.DataFrame({
            'Day': [day],
            'ProductName': [product],
            'weather': [weather],
            'Product_Calorie': [calories],
            'Inflation_Percentage': [inflation],
            'Unemployment_Percentage': [unemployment],
            'Promo_applied': [promo if promo else np.nan]
        })
        
        # Feature engineering
        input_data['weekend'] = input_data['Day'].isin(['Saturday', 'Sunday']).astype(int)
        input_data['promo_flag'] = input_data['Promo_applied'].notna().astype(int)
        
        # Encode categorical variables
        for col in ['Day', 'ProductName', 'weather']:
            input_data[f'{col}_encoded'] = self.encoders[col].transform(input_data[col])
        
        # Prepare features
        X = input_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        return max(0, prediction)  # Ensure non-negative prediction
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.encoders = model_data['encoders']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns'] 