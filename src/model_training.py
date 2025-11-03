import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from classification_models import ClassificationModels
from src.imbalanced_handling import ImbalancedDataHandler

# model_training.py - Use the existing handler
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.classifier = ClassificationModels(config)
        self.data_handler = ImbalancedDataHandler(config)  # REUSE existing
    
    def train_with_sampling(self, X, y, test_size=0.2, method='smote'):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Use the existing ImbalancedDataHandler
        X_train_resampled, y_train_resampled = self.data_handler.apply_sampling(
            X_train, y_train, method
        )
        
        self.classifier.train_models(X_train_resampled, y_train_resampled)
        return X_test, y_test