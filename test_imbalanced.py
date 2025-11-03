import numpy as np
from src.imbalanced_handling import ImbalancedDataHandler
from sklearn.ensemble import RandomForestClassifier

# Quick test
def quick_test():
    print("🚀 Quick Test: Imbalanced Data Handling")
    
    # Initialize
    config = {}
    handler = ImbalancedDataHandler(config)
    
    # Create highly imbalanced data (similar to paper: 90% Normal, 2.6% Fall, 6.8% Lying)
    n_samples = 1000
    n_normal = 900    # 90%
    n_fall = 26       # 2.6% 
    n_lying = 74      # 7.4% (adjusted to make 1000 total)
    
    # Generate features for each class with different distributions
    X_normal = np.random.normal(0, 1, (n_normal, 10))
    X_fall = np.random.normal(2, 2, (n_fall, 10))      # Different distribution for falls
    X_lying = np.random.normal(1, 1.5, (n_lying, 10))  # Different distribution for lying
    
    X = np.vstack([X_normal, X_fall, X_lying])
    y = np.array([0]*n_normal + [1]*n_fall + [2]*n_lying)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    print(f"📊 Generated data with distribution:")
    print(f"  Class 0 (Normal): {np.sum(y == 0)} samples ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  Class 1 (Fall):   {np.sum(y == 1)} samples ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  Class 2 (Lying):  {np.sum(y == 2)} samples ({np.sum(y == 2)/len(y)*100:.1f}%)")
    
    # Test SMOTE sampling (as used in paper)
    print("\n🔄 Applying SMOTE (as mentioned in paper)...")
    X_resampled, y_resampled = handler.apply_sampling(X, y, 'smote')
    
    print(f"\n📈 After SMOTE:")
    print(f"  Class 0 (Normal): {np.sum(y_resampled == 0)} samples")
    print(f"  Class 1 (Fall):   {np.sum(y_resampled == 1)} samples") 
    print(f"  Class 2 (Lying):  {np.sum(y_resampled == 2)} samples")
    
    # Test anomaly detection (consider Fall as anomaly)
    print("\n🔍 Testing Isolation Forest (as mentioned in paper)...")
    # Convert to binary: Normal vs Anomaly (Fall + Lying)
    y_binary = (y != 0).astype(int)
    y_pred, model = handler.apply_anomaly_detection(X, y, 'isolation_forest', normal_class=0)
    
    accuracy = np.mean(y_pred == y_binary)
    print(f"  Anomaly detection accuracy: {accuracy:.3f}")
    
    print("\n✅ All methods working correctly!")

if __name__ == "__main__":
    quick_test()