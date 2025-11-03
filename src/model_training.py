import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek
import joblib
import os

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.sampling_methods = {}
        
    def initialize_models(self):
        """Initialize all models as per paper"""
        self.models = {
            'knn': KNeighborsClassifier(
                n_neighbors=self.config['models']['knn']['n_neighbors'],
                weights=self.config['models']['knn']['weights']
            ),
            'svm': SVC(
                C=self.config['models']['svm']['C'],
                kernel=self.config['models']['svm']['kernel'],
                probability=True
            ),
            'adaboost': AdaBoostClassifier(
                n_estimators=self.config['models']['adaboost']['n_estimators']
            ),
            'xgboost': XGBClassifier(
                n_estimators=self.config['models']['xgboost']['n_estimators'],
                max_depth=self.config['models']['xgboost']['max_depth']
            )
        }
        
    def initialize_sampling_methods(self):
        """Initialize sampling methods for imbalanced data"""
        self.sampling_methods = {
            'random_sampling': RandomOverSampler(random_state=42),
            'smote': SMOTE(random_state=42),
            'smote_tomek': SMOTETomek(random_state=42),
            'adasyn': ADASYN(random_state=42)
        }
    
    def apply_sampling(self, X_train, y_train, method_name):
        """Apply sampling method to training data"""
        if method_name in self.sampling_methods:
            sampler = self.sampling_methods[method_name]
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            return X_resampled, y_resampled
        else:
            return X_train, y_train
    
    def train_models(self, X_train, y_train, use_sampling=True):
        """Train all models with optional sampling"""
        self.initialize_models()
        self.initialize_sampling_methods()
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            if use_sampling:
                # Try different sampling methods (as per paper)
                best_score = 0
                best_model = None
                
                for sampling_method in self.sampling_methods.keys():
                    X_resampled, y_resampled = self.apply_sampling(X_train, y_train, sampling_method)
                    
                    # Train model
                    current_model = model.__class__(**model.get_params())
                    current_model.fit(X_resampled, y_resampled)
                    
                    # Simple cross-validation score (can be enhanced)
                    score = current_model.score(X_train, y_train)
                    
                    if score > best_score:
                        best_score = score
                        best_model = current_model
                
                trained_models[model_name] = best_model
            else:
                # Train without sampling
                model.fit(X_train, y_train)
                trained_models[model_name] = model
        
        self.models = trained_models
        return trained_models
    
    def save_models(self, directory='models'):
        """Save trained models"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for model_name, model in self.models.items():
            filename = os.path.join(directory, f'{model_name}.joblib')
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
    
    def load_models(self, directory='models'):
        """Load trained models"""
        self.models = {}
        for filename in os.listdir(directory):
            if filename.endswith('.joblib'):
                model_name = filename.replace('.joblib', '')
                model_path = os.path.join(directory, filename)
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} from {model_path}")