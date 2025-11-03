import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_synthetic_data(self):
        """Generate synthetic data for testing without real datasets"""
        n_samples = 1000
        
        # Simulate features mentioned in the paper
        synthetic_data = {
            'hw_ratio': np.random.uniform(0.5, 3.0, n_samples),
            'spine_ratio': np.random.uniform(0.5, 4.0, n_samples),
            'neck_to_feet_dist': np.random.uniform(100, 200, n_samples),
            'hip_to_feet_dist': np.random.uniform(80, 180, n_samples),
            'head_acceleration': np.random.uniform(-50, 50, n_samples),
            'neck_acceleration': np.random.uniform(-50, 50, n_samples),
            'hip_acceleration': np.random.uniform(-50, 50, n_samples),
            'body_tilt_angle': np.random.uniform(0, 90, n_samples)
        }
        
        # Simulate class distribution similar to paper (highly imbalanced)
        labels = np.random.choice(
            [0, 1, 2], 
            n_samples, 
            p=[0.906, 0.026, 0.068]  # Normal, Fall, Lying percentages from paper
        )
        
        features_df = pd.DataFrame(synthetic_data)
        labels_df = pd.DataFrame({'label': labels})
        
        return features_df, labels_df
    
    def handle_missing_values(self, df):
        """Handle missing values as described in the paper"""
        # Add indicator columns for missing values
        for col in ['spine_ratio', 'deflection_angle']:  # Features that might have missing values
            missing_indicator = f'have_{col}'
            df[missing_indicator] = ~df[col].isna().astype(int)
            
        # Fill missing values with class-wise mean (as per paper)
        for label in self.config['data']['classes']:
            mask = df['label'] == label
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64] and col != 'label':
                    df.loc[mask, col] = df.loc[mask, col].fillna(df.loc[mask, col].mean())
        
        return df
    
    def remove_outliers(self, df, feature_names, lower_percentile=10, upper_percentile=90):
        """Remove outliers using percentile method as mentioned in paper"""
        for feature in feature_names:
            lower_bound = np.percentile(df[feature].dropna(), lower_percentile)
            upper_bound = np.percentile(df[feature].dropna(), upper_percentile)
            df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
        
        return df
    
    def prepare_data(self, features_df, labels_df):
        """Prepare data for training"""
        # Combine features and labels
        data = pd.concat([features_df, labels_df], axis=1)
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Remove outliers from ratio features (as per paper)
        data = self.remove_outliers(data, ['hw_ratio', 'spine_ratio'])
        
        # Split features and labels
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=self.config['data']['train_test_split'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler