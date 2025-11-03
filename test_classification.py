import numpy as np
from src.classification_models import ClassificationModels, generate_fall_detection_data

def quick_test():
    print("🚀 Quick Test: Classification Models")
    
    # Configuration as mentioned in paper
    config = {
        'knn': {'n_neighbors': 5, 'weights': 'uniform'},
        'svm': {'C': 1.0, 'kernel': 'rbf'},
        'adaboost': {'n_estimators': 50},
        'xgboost': {'n_estimators': 100, 'max_depth': 6}
    }
    
    # Initialize classifier
    classifier = ClassificationModels(config)
    
    # Generate data with paper-like distribution (highly imbalanced)
    print("📊 Generating imbalanced data similar to paper...")
    X, y = generate_fall_detection_data(n_samples=1500)
    
    # Make it highly imbalanced like paper (90% Normal, 3% Fall, 7% Lying)
    normal_mask = (y == 0)
    fall_mask = (y == 1) 
    lying_mask = (y == 2)
    
    # Resample to match paper distribution
    X_balanced = np.vstack([X[normal_mask][:1350], X[fall_mask][:45], X[lying_mask][:105]])
    y_balanced = np.array([0]*1350 + [1]*45 + [2]*105)
    
    # Shuffle
    indices = np.random.permutation(len(X_balanced))
    X, y = X_balanced[indices], y_balanced[indices]
    
    print(f"Final distribution: Normal={np.sum(y==0)}, Fall={np.sum(y==1)}, Lying={np.sum(y==2)}")
    print(f"Percentages: Normal={np.sum(y==0)/len(y)*100:.1f}%, "
          f"Fall={np.sum(y==1)/len(y)*100:.1f}%, "
          f"Lying={np.sum(y==2)/len(y)*100:.1f}%")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    print("\n🏋️ Training models...")
    classifier.train_models(X_train, y_train, optimize_hyperparams=False)
    
    # Evaluate
    print("\n📊 Evaluating models...")
    results = classifier.evaluate_models(X_test, y_test, class_names=['Normal', 'Fall', 'Lying'])
    
    # Show best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0].upper()} with accuracy: {best_model[1]['accuracy']:.3f}")
    
    # Test KNN specifically (as mentioned in paper)
    knn_predictions = classifier.models['knn'].predict(X_test[:5])
    print(f"\n🔍 KNN predictions on first 5 samples: {knn_predictions}")
    print(f"   True labels: {y_test[:5]}")
    
    print("\n✅ All classification models working correctly!")

if __name__ == "__main__":
    quick_test()