"""Model configuration and building utilities."""

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_xception_model(input_shape=(224, 224, 3), num_classes=2, dropout_rate=0.5, l2_reg=0.01):
    """Build and return an Xception-based model for facial emotion recognition.
    
    Args:
        input_shape (tuple): Input shape of the images (height, width, channels)
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization
        l2_reg (float): L2 regularization factor
        
    Returns:
        keras.Model: Compiled Xception model
    """
    # Base model
    base_model = applications.Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )
    
    # Add custom layers
    x = Flatten(name='flatten')(base_model.output)
    x = Dense(512, activation='relu', name='fc1', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu', name='fc2', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    # Output layer
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def get_callbacks(checkpoint_path, csv_logger_path, patience=50):
    """Get model training callbacks.
    
    Args:
        checkpoint_path (str): Path to save model checkpoints
        csv_logger_path (str): Path to save training logs
        patience (int): Number of epochs to wait before early stopping
        
    Returns:
        list: List of Keras callbacks
    """
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            csv_logger_path,
            append=False,
            separator=';'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            verbose=1,
            min_lr=0.00001
        )
    ]
    return callbacks