from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
import yaml

def demo_with_synthetic_data():
    """Demo the system with synthetic data"""
    print("Fall Detection System Demo")
    print("=" * 50)
    
    # Load config
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Generate synthetic data
    print("1. Generating synthetic data...")
    preprocessor = DataPreprocessor(config)
    features, labels = preprocessor.load_synthetic_data()
    print(f"Generated {len(features)} samples")
    print(f"Class distribution: {labels['label'].value_counts().to_dict()}")
    
    # Prepare data
    print("2. Preparing data...")
    X_train, X_test, y_train, y_test, scaler = preprocessor.prepare_data(features, labels)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Train models
    print("3. Training models...")
    trainer = ModelTrainer(config)
    trained_models = trainer.train_models(X_train, y_train, use_sampling=True)
    print(f"Trained {len(trained_models)} models")
    
    # Evaluate models
    print("4. Evaluating models...")
    evaluator = ModelEvaluator(config)
    evaluation_results = []
    
    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name}...")
        result = evaluator.evaluate_model(model, X_test, y_test, model_name)
        evaluation_results.append(result)
    
    # Generate report
    print("5. Generating final report...")
    report = evaluator.generate_report(evaluation_results)
    
    print("\nDemo completed successfully!")
    return report

if __name__ == "__main__":
    demo_with_synthetic_data()