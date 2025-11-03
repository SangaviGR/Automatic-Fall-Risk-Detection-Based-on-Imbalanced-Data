import argparse
import yaml
from src.data_preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor
from src.pose_estimation import PoseEstimator
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description='Fall Detection System')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'extract_features'], required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode == 'extract_features':
        # Extract features from videos (requires datasets)
        pose_estimator = PoseEstimator(config)
        feature_extractor = FeatureExtractor(config)
        
        # This part requires datasets - placeholder for your partner
        print("Feature extraction requires datasets. Please implement this part when datasets are available.")
        
    elif args.mode == 'train':
        # Train models with extracted features
        trainer = ModelTrainer(config)
        trainer.train_models()
        
    elif args.mode == 'test':
        # Evaluate models
        evaluator = ModelEvaluator(config)
        evaluator.evaluate_models()

if __name__ == "__main__":
    main()