import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

class ClassificationModels:
    """
    Implements all classification models mentioned in the paper:
    - K-Nearest Neighbors (KNN)
    - Support Vector Machines (SVM) 
    - Boosting Methods (AdaBoost, XGBoost)
    """
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_params = {}
        self.cv_results = {}
        
    def initialize_models(self):
        """Initialize all classification models with default parameters"""
        self.models = {
            'knn': KNeighborsClassifier(
                n_neighbors=self.config.get('knn', {}).get('n_neighbors', 5),
                weights=self.config.get('knn', {}).get('weights', 'uniform')
            ),
            'svm': SVC(
                C=self.config.get('svm', {}).get('C', 1.0),
                kernel=self.config.get('svm', {}).get('kernel', 'rbf'),
                probability=True,
                random_state=42
            ),
            'adaboost': AdaBoostClassifier(
                n_estimators=self.config.get('adaboost', {}).get('n_estimators', 50),
                random_state=42
            ),
            'xgboost': XGBClassifier(
                n_estimators=self.config.get('xgboost', {}).get('n_estimators', 100),
                max_depth=self.config.get('xgboost', {}).get('max_depth', 6),
                random_state=42,
                eval_metric='mlogloss'
            )
        }
    
    def train_models(self, X_train, y_train, optimize_hyperparams=False):
        """
        Train all classification models
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimize_hyperparams: Whether to perform hyperparameter optimization
        """
        self.initialize_models()
        
        if optimize_hyperparams:
            self._optimize_hyperparameters(X_train, y_train)
        
        print("🏋️ Training Classification Models...")
        print("=" * 50)
        
        for model_name, model in self.models.items():
            print(f"Training {model_name.upper()}...")
            
            # If we optimized hyperparameters, use the best ones
            if optimize_hyperparams and model_name in self.best_params:
                model.set_params(**self.best_params[model_name])
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            self.cv_results[model_name] = {
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'all_scores': cv_scores
            }
            
            print(f"  ✅ {model_name.upper()} trained successfully")
            print(f"  📊 Cross-validation accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        
        print("🎯 All models trained successfully!")
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """Perform hyperparameter optimization for each model"""
        print("\n🔍 Optimizing Hyperparameters...")
        
        # KNN hyperparameter optimization
        knn_param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        knn_grid = GridSearchCV(
            KNeighborsClassifier(), knn_param_grid, 
            cv=5, scoring='accuracy', n_jobs=-1
        )
        knn_grid.fit(X_train, y_train)
        self.best_params['knn'] = knn_grid.best_params_
        print(f"  KNN best params: {knn_grid.best_params_}")
        
        # SVM hyperparameter optimization
        svm_param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto']
        }
        svm_grid = GridSearchCV(
            SVC(probability=True, random_state=42), svm_param_grid,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        svm_grid.fit(X_train, y_train)
        self.best_params['svm'] = svm_grid.best_params_
        print(f"  SVM best params: {svm_grid.best_params_}")
        
        # AdaBoost hyperparameter optimization
        ada_param_grid = {
            'n_estimators': [25, 50, 100],
            'learning_rate': [0.1, 0.5, 1.0]
        }
        ada_grid = GridSearchCV(
            AdaBoostClassifier(random_state=42), ada_param_grid,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        ada_grid.fit(X_train, y_train)
        self.best_params['adaboost'] = ada_grid.best_params_
        print(f"  AdaBoost best params: {ada_grid.best_params_}")
        
        # XGBoost hyperparameter optimization
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        xgb_grid = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='mlogloss'), xgb_param_grid,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        xgb_grid.fit(X_train, y_train)
        self.best_params['xgboost'] = xgb_grid.best_params_
        print(f"  XGBoost best params: {xgb_grid.best_params_}")
    
    def predict(self, X_test):
        """Make predictions with all trained models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            predictions[model_name] = {
                'class_predictions': model.predict(X_test),
                'probabilities': model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            }
        
        return predictions
    
    def evaluate_models(self, X_test, y_test, class_names=None):
        """
        Evaluate all models on test data
        
        Args:
            X_test: Test features
            y_test: True test labels
            class_names: Names of classes for reporting
            
        Returns:
            results: Dictionary with evaluation metrics for each model
        """
        if class_names is None:
            class_names = [f'Class_{i}' for i in np.unique(y_test)]
        
        results = {}
        predictions = self.predict(X_test)
        
        print("\n📊 Model Evaluation Results")
        print("=" * 60)
        
        for model_name, pred_dict in predictions.items():
            y_pred = pred_dict['class_predictions']
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            
            results[model_name] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': y_pred
            }
            
            print(f"\n{model_name.upper()} Results:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Cross-val Mean: {self.cv_results[model_name]['mean_score']:.3f}")
            
            # Print per-class metrics
            for class_name in class_names:
                if class_name in report:
                    precision = report[class_name]['precision']
                    recall = report[class_name]['recall']
                    f1 = report[class_name]['f1-score']
                    print(f"  {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return results
    
    def plot_comparison(self, results):
        """Plot comparison of all models"""
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        cv_scores = [self.cv_results[model]['mean_score'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        x_pos = np.arange(len(models))
        ax1.bar(x_pos - 0.2, accuracies, 0.4, label='Test Accuracy', alpha=0.8)
        ax1.bar(x_pos + 0.2, cv_scores, 0.4, label='CV Accuracy', alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.upper() for m in models])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confusion matrices
        for i, model_name in enumerate(models):
            cm = results[model_name]['confusion_matrix']
            plt.subplot(2, len(models), len(models) + i + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name.upper()} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self):
        """Get feature importance for tree-based models"""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_dict[model_name] = np.mean(np.abs(model.coef_), axis=0)
        
        return importance_dict
    
    def plot_feature_importance(self, feature_names=None):
        """Plot feature importance for models that support it"""
        importance_dict = self.get_feature_importance()
        
        if not importance_dict:
            print("No feature importance available for these models")
            return
        
        n_models = len(importance_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance) in enumerate(importance_dict.items()):
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importance))]
            
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sorted_importance = importance[indices]
            
            axes[idx].barh(range(len(sorted_importance)), sorted_importance)
            axes[idx].set_yticks(range(len(sorted_importance)))
            axes[idx].set_yticklabels(sorted_features)
            axes[idx].set_title(f'{model_name.upper()} Feature Importance')
            axes[idx].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()

# Test the implementation
def test_classification_models():
    """Test the classification models with synthetic data"""
    print("🧪 Testing Classification Models")
    print("=" * 60)
    
    # Configuration (as mentioned in paper)
    config = {
        'knn': {'n_neighbors': 5, 'weights': 'uniform'},
        'svm': {'C': 1.0, 'kernel': 'rbf'},
        'adaboost': {'n_estimators': 50},
        'xgboost': {'n_estimators': 100, 'max_depth': 6}
    }
    
    # Initialize classifier
    classifier = ClassificationModels(config)
    
    # Generate synthetic fall detection data (3 classes as per paper)
    print("\n📊 Generating synthetic fall detection data...")
    X, y = generate_fall_detection_data()
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print("Classes: 0=Normal, 1=Fall, 2=Lying")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    classifier.train_models(X_train, y_train, optimize_hyperparams=False)
    
    # Evaluate models
    results = classifier.evaluate_models(X_test, y_test, class_names=['Normal', 'Fall', 'Lying'])
    
    # Plot comparisons
    classifier.plot_comparison(results)
    
    # Show feature importance (for tree-based models)
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    classifier.plot_feature_importance(feature_names)
    
    print("\n" + "=" * 60)
    print("🎉 All classification model tests completed!")

def generate_fall_detection_data(n_samples=1000):
    """
    Generate synthetic fall detection data with 3 classes:
    - Class 0: Normal (walking, standing)
    - Class 1: Fall (falling motion) 
    - Class 2: Lying (on ground)
    """
    n_normal = int(n_samples * 0.7)  # 70% normal
    n_fall = int(n_samples * 0.15)   # 15% fall
    n_lying = n_samples - n_normal - n_fall  # 15% lying
    
    # Feature descriptions based on paper:
    # Features 0-1: Ratio features (HW ratio, Spine ratio)
    # Features 2-3: Distance features  
    # Features 4-6: Acceleration features
    # Features 7-12: Deflection angles
    # Feature 13: Body tilt angle
    
    # Normal class - upright, stable features
    X_normal = np.column_stack([
        np.random.uniform(1.5, 3.0, n_normal),    # HW ratio (high - upright)
        np.random.uniform(1.5, 2.5, n_normal),    # Spine ratio (high)
        np.random.uniform(150, 200, n_normal),    # Neck-to-feet distance (high)
        np.random.uniform(120, 180, n_normal),    # Hip-to-feet distance (high)
        np.random.normal(0, 5, n_normal),         # Head acceleration (stable)
        np.random.normal(0, 5, n_normal),         # Neck acceleration (stable)
        np.random.normal(0, 5, n_normal),         # Hip acceleration (stable)
        np.random.uniform(0, 20, n_normal),       # Spine deflection (small)
        np.random.uniform(80, 100, n_normal),     # Waist deflection (horizontal)
        np.random.uniform(80, 100, n_normal),     # Right thigh deflection
        np.random.uniform(80, 100, n_normal),     # Left thigh deflection  
        np.random.uniform(80, 100, n_normal),     # Right calf deflection
        np.random.uniform(80, 100, n_normal),     # Left calf deflection
        np.random.uniform(0, 30, n_normal)        # Body tilt (small)
    ])
    
    # Fall class - changing, unstable features
    X_fall = np.column_stack([
        np.random.uniform(0.5, 1.5, n_fall),      # HW ratio (low - horizontal)
        np.random.uniform(0.5, 1.5, n_fall),      # Spine ratio (low)
        np.random.uniform(50, 120, n_fall),       # Neck-to-feet distance (low)
        np.random.uniform(30, 100, n_fall),       # Hip-to-feet distance (low)
        np.random.normal(-20, 10, n_fall),        # Head acceleration (falling)
        np.random.normal(-20, 10, n_fall),        # Neck acceleration (falling)
        np.random.normal(-20, 10, n_fall),        # Hip acceleration (falling)
        np.random.uniform(30, 90, n_fall),        # Spine deflection (varying)
        np.random.uniform(30, 90, n_fall),        # Waist deflection (varying)
        np.random.uniform(30, 90, n_fall),        # Right thigh deflection
        np.random.uniform(30, 90, n_fall),        # Left thigh deflection
        np.random.uniform(30, 90, n_fall),        # Right calf deflection
        np.random.uniform(30, 90, n_fall),        # Left calf deflection
        np.random.uniform(30, 90, n_fall)         # Body tilt (large)
    ])
    
    # Lying class - horizontal, stable on ground
    X_lying = np.column_stack([
        np.random.uniform(0.3, 0.8, n_lying),     # HW ratio (very low)
        np.random.uniform(0.3, 0.8, n_lying),     # Spine ratio (very low)
        np.random.uniform(10, 50, n_lying),       # Neck-to-feet distance (very low)
        np.random.uniform(5, 30, n_lying),        # Hip-to-feet distance (very low)
        np.random.normal(0, 2, n_lying),          # Head acceleration (stable)
        np.random.normal(0, 2, n_lying),          # Neck acceleration (stable)
        np.random.normal(0, 2, n_lying),          # Hip acceleration (stable)
        np.random.uniform(80, 100, n_lying),      # Spine deflection (horizontal)
        np.random.uniform(80, 100, n_lying),      # Waist deflection (horizontal)
        np.random.uniform(80, 100, n_lying),      # Right thigh deflection
        np.random.uniform(80, 100, n_lying),      # Left thigh deflection
        np.random.uniform(80, 100, n_lying),      # Right calf deflection
        np.random.uniform(80, 100, n_lying),      # Left calf deflection
        np.random.uniform(80, 100, n_lying)       # Body tilt (horizontal)
    ])
    
    # Combine all data
    X = np.vstack([X_normal, X_fall, X_lying])
    y = np.array([0]*n_normal + [1]*n_fall + [2]*n_lying)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]

if __name__ == "__main__":
    test_classification_models()