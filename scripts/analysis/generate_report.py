"""Generate comprehensive analysis report for FER experiment."""

import os
import argparse
import pandas as pd
import yaml
import logging
from pathlib import Path
from datetime import datetime
from src.visualization.metrics import MetricsVisualizer
from src.visualization.plots import DatasetVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_report(config_path):
    """Generate analysis report.
    
    Args:
        config_path (str): Path to configuration file
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(config['paths']['reports_dir'], f'report_{timestamp}')
    plots_dir = os.path.join(report_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize visualizers
    metrics_viz = MetricsVisualizer(save_dir=plots_dir)
    dataset_viz = DatasetVisualizer(save_dir=plots_dir)
    
    # Load data
    training_history = pd.read_csv(config['paths']['training_history'])
    validation_results = pd.read_csv(config['paths']['validation_results'])
    test_results = pd.read_csv(config['paths']['test_results'])
    
    # Generate dataset analysis
    logger.info("Generating dataset analysis...")
    
    # Analyze training data distribution
    dataset_viz.plot_class_balance(
        data=pd.read_csv(config['paths']['train_metadata']),
        class_column='reaction_type',
        split_column='category'
    )
    
    # Plot data splits distribution
    dataset_viz.plot_data_splits(
        data=pd.concat([
            pd.read_csv(config['paths']['train_metadata']),
            pd.read_csv(config['paths']['val_metadata']),
            pd.read_csv(config['paths']['test_metadata'])
        ]),
        split_column='split',
        class_column='reaction_type'
    )
    
    # Analyze image statistics
    dataset_viz.plot_image_statistics(config['paths']['processed_data'])
    
    # Generate training analysis
    logger.info("Generating training analysis...")
    
    # Plot training history
    metrics_viz.plot_training_history(
        training_history,
        metrics=['loss', 'accuracy', 'precision', 'recall', 'f1_score']
    )
    
    # Generate evaluation analysis
    logger.info("Generating evaluation analysis...")
    
    # Plot confusion matrices
    metrics_viz.plot_confusion_matrix(
        test_results['true_label'],
        test_results['predicted_label'],
        classes=['not_interested', 'interested']
    )
    
    # Plot ROC curves
    metrics_viz.plot_roc_curves(
        test_results['true_label'],
        test_results[['prob_not_interested', 'prob_interested']].values,
        classes=['not_interested', 'interested']
    )
    
    # Plot precision-recall curves
    metrics_viz.plot_precision_recall_curves(
        test_results['true_label'],
        test_results[['prob_not_interested', 'prob_interested']].values,
        classes=['not_interested', 'interested']
    )
    
    # Generate summary statistics
    summary = {
        'dataset': {
            'total_samples': len(test_results),
            'class_distribution': test_results['true_label'].value_counts().to_dict(),
            'categories': test_results['category'].unique().tolist()
        },
        'performance': {
            'test_accuracy': test_results['accuracy'].mean(),
            'test_precision': test_results['precision'].mean(),
            'test_recall': test_results['recall'].mean(),
            'test_f1': test_results['f1_score'].mean()
        },
        'training': {
            'final_loss': training_history['loss'].iloc[-1],
            'best_val_accuracy': training_history['val_accuracy'].max(),
            'epochs_trained': len(training_history)
        }
    }
    
    # Save summary
    with open(os.path.join(report_dir, 'summary.yaml'), 'w') as f:
        yaml.dump(summary, f)
    
    logger.info(f"Report generated successfully in: {report_dir}")
    return report_dir

def main():
    """Main function to generate analysis report."""
    parser = argparse.ArgumentParser(description="Generate FER analysis report")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/analysis_config.yaml",
        help="Path to analysis configuration file"
    )
    args = parser.parse_args()
    
    try:
        report_dir = generate_report(args.config)
        print(f"\nReport generated successfully!")
        print(f"Report location: {report_dir}")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise

if __name__ == "__main__":
    main()