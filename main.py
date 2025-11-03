from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

def main():
    # Load configuration
    config = {
        'knn': {'n_neighbors': 5, 'weights': 'uniform'},
        'svm': {'C': 1.0, 'kernel': 'rbf'},
        'adaboost': {'n_estimators': 50},
        'xgboost': {'n_estimators': 100, 'max_depth': 6}
    }
    
    # Initialize components
    preprocessor = DataPreprocessor(config)
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(config)
    
    # Load and preprocess data
    X, y = preprocessor.load_and_preprocess_data()
    
    # Train with SMOTE (as per paper)
    X_test, y_test = trainer.train_with_sampling(X, y)
    
    # Evaluate models
    results = evaluator.evaluate_all_models(
        trainer.classifier, X_test, y_test
    )
    
    # Generate reports
    evaluator.generate_paper_comparison(results)

if __name__ == "__main__":
    main()