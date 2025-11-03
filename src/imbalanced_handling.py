import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

class ImbalancedDataHandler:
    """
    Implements all imbalanced data handling methods mentioned in the paper:
    - Sampling methods (Oversampling and Undersampling)
    - Anomaly detection methods
    """
    
    def __init__(self, config):
        self.config = config
        self.sampling_methods = {}
        self.anomaly_methods = {}
        
    def initialize_sampling_methods(self):
        """Initialize all sampling methods as per paper"""
        self.sampling_methods = {
            # Random sampling (mentioned as one kind of random sampling)
            'random_sampling': RandomOverSampler(random_state=42),
            
            # Synthetic sampling methods (three kinds as per paper)
            'smote': SMOTE(random_state=42),
            'smote_tomek': SMOTETomek(random_state=42),
            'adasyn': ADASYN(random_state=42)
        }
        
    def initialize_anomaly_methods(self):
        """Initialize all anomaly detection methods as per paper"""
        self.anomaly_methods = {
            'isolation_forest': IsolationForest(
                contamination='auto',  # Auto-detect outliers
                random_state=42,
                n_estimators=100
            ),
            'one_class_svm': OneClassSVM(
                kernel='rbf',  # Radial Basis Function as mentioned in paper
                gamma='scale',
                nu=0.1  # Controls the number of outliers
            ),
            'elliptic_envelope': EllipticEnvelope(
                contamination=0.1,  # Assume 10% outliers
                random_state=42
            )
        }
    
    def apply_sampling(self, X, y, method_name):
        """
        Apply sampling method to balance the dataset
        
        Args:
            X: Feature matrix
            y: Target labels
            method_name: Name of sampling method to apply
            
        Returns:
            X_resampled, y_resampled: Resampled data
        """
        self.initialize_sampling_methods()
        
        if method_name not in self.sampling_methods:
            raise ValueError(f"Unknown sampling method: {method_name}")
        
        sampler = self.sampling_methods[method_name]
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        print(f"Applied {method_name}:")
        print(f"  Original shape: {X.shape}, Classes: {np.bincount(y)}")
        print(f"  Resampled shape: {X_resampled.shape}, Classes: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def apply_anomaly_detection(self, X, y, method_name, normal_class=0):
        """
        Apply anomaly detection for imbalanced data
        
        Args:
            X: Feature matrix
            y: Target labels  
            method_name: Name of anomaly detection method
            normal_class: Which class to consider as 'normal' (majority class)
            
        Returns:
            y_pred: Binary predictions (0=normal, 1=anomaly)
            model: Trained anomaly detection model
        """
        self.initialize_anomaly_methods()
        
        if method_name not in self.anomaly_methods:
            raise ValueError(f"Unknown anomaly method: {method_name}")
        
        # Separate normal and anomaly data
        normal_mask = (y == normal_class)
        X_normal = X[normal_mask]
        
        # Train on normal data only
        model = self.anomaly_methods[method_name]
        
        if method_name == 'one_class_svm':
            # OneClassSVM only uses normal data for training
            model.fit(X_normal)
            # Predict on all data
            y_pred = model.predict(X)
            # Convert from {-1, 1} to {0, 1}
            y_pred = (y_pred == -1).astype(int)
        else:
            # Isolation Forest and Elliptic Envelope use all data
            model.fit(X)
            y_pred = model.predict(X)
            # Convert from {-1, 1} to {0, 1}
            y_pred = (y_pred == -1).astype(int)
        
        return y_pred, model
    
    def compare_sampling_methods(self, X_train, y_train, X_test, y_test, classifier):
        """
        Compare different sampling methods and return performance metrics
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            classifier: Base classifier to use for evaluation
            
        Returns:
            results: Dictionary with performance metrics for each method
        """
        self.initialize_sampling_methods()
        
        results = {}
        
        # Test without sampling (baseline)
        classifier.fit(X_train, y_train)
        baseline_score = classifier.score(X_test, y_test)
        results['no_sampling'] = {
            'accuracy': baseline_score,
            'train_shape': X_train.shape,
            'class_distribution': np.bincount(y_train)
        }
        
        # Test each sampling method
        for method_name, sampler in self.sampling_methods.items():
            try:
                # Apply sampling to training data only
                X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                
                # Train classifier on resampled data
                classifier.fit(X_resampled, y_resampled)
                
                # Evaluate on original test data
                accuracy = classifier.score(X_test, y_test)
                
                results[method_name] = {
                    'accuracy': accuracy,
                    'train_shape': X_resampled.shape,
                    'class_distribution': np.bincount(y_resampled)
                }
                
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                results[method_name] = {'error': str(e)}
        
        return results
    
    def compare_anomaly_methods(self, X, y, normal_class=0):
        """
        Compare different anomaly detection methods
        
        Args:
            X: Feature matrix
            y: True labels
            normal_class: Which class is considered normal
            
        Returns:
            results: Dictionary with performance metrics for each method
        """
        self.initialize_anomaly_methods()
        
        results = {}
        
        # Convert to binary classification for anomaly detection
        y_binary = (y != normal_class).astype(int)
        
        for method_name in self.anomaly_methods.keys():
            try:
                y_pred, model = self.apply_anomaly_detection(X, y, method_name, normal_class)
                
                # Calculate metrics
                accuracy = np.mean(y_pred == y_binary)
                precision = self._calculate_precision(y_binary, y_pred)
                recall = self._calculate_recall(y_binary, y_pred)
                f1 = self._calculate_f1(y_binary, y_pred)
                
                results[method_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'predictions': y_pred
                }
                
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                results[method_name] = {'error': str(e)}
        
        return results
    
    def _calculate_precision(self, y_true, y_pred):
        """Calculate precision score"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    
    def _calculate_recall(self, y_true, y_pred):
        """Calculate recall score"""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def _calculate_f1(self, y_true, y_pred):
        """Calculate F1 score"""
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    def generate_synthetic_imbalanced_data(self, n_samples=1000, imbalance_ratio=0.1):
        """
        Generate synthetic imbalanced data for testing
        
        Args:
            n_samples: Total number of samples
            imbalance_ratio: Ratio of minority class samples
            
        Returns:
            X, y: Synthetic features and labels
        """
        # Generate features from normal distributions
        n_minority = int(n_samples * imbalance_ratio)
        n_majority = n_samples - n_minority
        
        # Majority class (class 0)
        X_majority = np.random.normal(0, 1, (n_majority, 5))
        y_majority = np.zeros(n_majority)
        
        # Minority class (class 1) - different distribution
        X_minority = np.random.normal(2, 1.5, (n_minority, 5))
        y_minority = np.ones(n_minority)
        
        # Combine
        X = np.vstack([X_majority, X_minority])
        y = np.hstack([y_majority, y_minority])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y

# Test the implementation
def test_imbalanced_handler():
    """Test the imbalanced data handler with synthetic data"""
    print("🧪 Testing Imbalanced Data Handling Methods")
    print("=" * 60)
    
    # Initialize handler
    config = {}
    handler = ImbalancedDataHandler(config)
    
    # Generate synthetic imbalanced data
    print("\n📊 Generating synthetic imbalanced data...")
    X, y = handler.generate_synthetic_imbalanced_data(n_samples=1000, imbalance_ratio=0.1)
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Imbalance ratio: {np.bincount(y)[1] / len(y):.3f}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Test sampling methods
    print("\n🔄 Testing Sampling Methods:")
    print("-" * 40)
    
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    
    sampling_results = handler.compare_sampling_methods(X_train, y_train, X_test, y_test, classifier)
    
    for method, result in sampling_results.items():
        if 'error' not in result:
            print(f"{method:15} | Accuracy: {result['accuracy']:.3f} | "
                  f"Train shape: {result['train_shape']} | "
                  f"Classes: {result['class_distribution']}")
    
    # Test anomaly detection methods
    print("\n🔍 Testing Anomaly Detection Methods:")
    print("-" * 40)
    
    anomaly_results = handler.compare_anomaly_methods(X_test, y_test, normal_class=0)
    
    for method, result in anomaly_results.items():
        if 'error' not in result:
            print(f"{method:20} | Accuracy: {result['accuracy']:.3f} | "
                  f"Precision: {result['precision']:.3f} | "
                  f"Recall: {result['recall']:.3f} | "
                  f"F1: {result['f1_score']:.3f}")
    
    # Test individual sampling application
    print("\n🎯 Testing Individual Sampling Application:")
    print("-" * 40)
    
    for method_name in handler.sampling_methods.keys():
        try:
            X_resampled, y_resampled = handler.apply_sampling(X_train, y_train, method_name)
            print(f"✅ {method_name} completed successfully")
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 All imbalanced data handling tests completed!")

if __name__ == "__main__":
    test_imbalanced_handler()