import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        
    def calculate_metrics(self, y_true, y_pred, average='macro'):
        """Calculate evaluation metrics as per paper"""
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        
        # Overall metrics
        overall_metrics = self.calculate_metrics(y_test, y_pred)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.config['data']['classes']):
            class_metrics = self.calculate_metrics(y_test, y_pred, average=None)
            per_class_metrics[class_name] = {
                'precision': class_metrics['precision'][i],
                'recall': class_metrics['recall'][i],
                'f1_score': class_metrics['f1_score'][i]
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'model_name': model_name,
            'overall_metrics': overall_metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def plot_confusion_matrix(self, cm, model_name, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, evaluation_results):
        """Generate comprehensive evaluation report"""
        report = {}
        
        for result in evaluation_results:
            model_name = result['model_name']
            report[model_name] = {
                'Overall': result['overall_metrics'],
                'Per Class': result['per_class_metrics']
            }
            
            # Print results in table format similar to paper
            print(f"\n{model_name} Performance:")
            print("Class\t\tPrecision\tRecall\t\tF1-Score")
            print("-" * 50)
            
            for class_name in self.config['data']['classes']:
                metrics = result['per_class_metrics'][class_name]
                print(f"{class_name}\t\t{metrics['precision']:.2f}\t\t{metrics['recall']:.2f}\t\t{metrics['f1_score']:.2f}")
            
            print(f"\nOverall F1-Score: {result['overall_metrics']['f1_score']:.2f}")
            
            # Plot confusion matrix
            self.plot_confusion_matrix(
                result['confusion_matrix'], 
                model_name, 
                self.config['data']['classes']
            )
        
        return report