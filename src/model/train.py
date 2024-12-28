"""Main training script for the FER model."""

import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Local imports
from metrics import F1Score
from model_config import build_xception_model, get_callbacks
from data_processing.data_generator import setup_data_generators, get_class_weights
from src.visualization.metrics import MetricsVisualizer
from src.visualization.plots import DatasetVisualizer


def train_model(config):
    """Train the FER model with the specified configuration.
    
    Args:
        config (dict): Configuration dictionary containing training parameters
    """
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    # Setup paths
    train_dir = config['paths']['train_dir']
    val_dir = config['paths']['val_dir']
    checkpoint_path = config['paths']['checkpoint_path']
    csv_logger_path = config['paths']['csv_logger_path']
    
    # Setup data generators
    train_generator, validation_generator = setup_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        img_height=config['data']['img_height'],
        img_width=config['data']['img_width'],
        batch_size=config['training']['batch_size']
    )
    
    # Calculate class weights
    class_weights = get_class_weights(train_generator.classes)
    
    # Build model
    model = build_xception_model(
        input_shape=(config['data']['img_height'], 
                    config['data']['img_width'], 3),
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
        l2_reg=config['model']['l2_reg']
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config['training']['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', F1Score()]
    )
    
    # Get callbacks
    callbacks = get_callbacks(
        checkpoint_path=checkpoint_path,
        csv_logger_path=csv_logger_path,
        patience=config['training']['early_stopping_patience']
    )
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=config['training']['epochs'],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(checkpoint_path)
    
    # Initialize visualizers
    viz_dir = os.path.join(config['paths']['output_dir'], 'visualizations')
    metrics_viz = MetricsVisualizer(save_dir=viz_dir)
    dataset_viz = DatasetVisualizer(save_dir=viz_dir)
    
    # Convert history to dataframe
    history_df = pd.DataFrame(history.history)
    
    # Plot training metrics
    metrics_viz.plot_training_history(history_df)
    
    # Plot data distributions
    dataset_df = pd.DataFrame({
        'reaction_type': train_generator.classes,
        'category': [train_generator.filenames[i].split('/')[0] for i in range(len(train_generator.filenames))]
    })
    dataset_viz.plot_class_balance(dataset_df, 'reaction_type', 'category')
    
    # Plot confusion matrix and ROC curves if validation data is available
    if validation_generator is not None:
        predictions = model.predict(validation_generator)
        val_df = pd.DataFrame({
            'true_label': validation_generator.classes,
            'predicted_label': predictions.argmax(axis=1),
            'prob_interested': predictions[:, 1],
            'prob_not_interested': predictions[:, 0]
        })
        
        metrics_viz.plot_confusion_matrix(
            val_df['true_label'],
            val_df['predicted_label'],
            classes=['not_interested', 'interested']
        )
        
        metrics_viz.plot_roc_curves(
            val_df['true_label'],
            predictions,
            classes=['not_interested', 'interested']
        )
        
        metrics_viz.plot_precision_recall_curves(
            val_df['true_label'],
            predictions,
            classes=['not_interested', 'interested']
        )
    
    return history

if __name__ == "__main__":
    # Example configuration
    config = {
        'data': {
            'img_height': 224,
            'img_width': 224,
        },
        'model': {
            'num_classes': 2,
            'dropout_rate': 0.5,
            'l2_reg': 0.01
        },
        'training': {
            'batch_size': 16,
            'epochs': 300,
            'learning_rate': 0.001,
            'early_stopping_patience': 50
        },
        'paths': {
            'train_dir': 'data/processed/train',
            'val_dir': 'data/processed/val',
            'checkpoint_path': 'models/checkpoints/fer_model_weights.h5',
            'csv_logger_path': 'logs/training_log.csv'
        }
    }
    
    history = train_model(config)