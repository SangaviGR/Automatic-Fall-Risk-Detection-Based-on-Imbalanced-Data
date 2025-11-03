import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from classification_models import ClassificationModels

class ModelTrainer:
    """
    Complete training pipeline with imbalanced data handling
    As described in paper Section V: EXPERIMENTAL RESULTS
    """
    
    def __init__(self, config):
        self.config = config
        self.classifier = ClassificationModels(config)
        
    def train_with_sampling(self, X, y, test_size=0.2):
        """
        Train models with SMOTE sampling as per paper
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print("📊 Original class distribution:")
        print(f"  Training: {np.bincount(y_train)}")
        print(f"  Testing:  {np.bincount(y_test)}")
        
        # Apply SMOTE sampling (as mentioned in paper)
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print("📈 After SMOTE sampling:")
        print(f"  Training: {np.bincount(y_train_resampled)}")
        
        # Train models
        self.classifier.train_models(X_train_resampled, y_train_resampled)
        
        return X_test, y_test
    
    def train_without_sampling(self, X, y, test_size=0.2):
        """
        Train models without sampling (baseline)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        self.classifier.train_models(X_train, y_train)
        
        return X_test, y_test