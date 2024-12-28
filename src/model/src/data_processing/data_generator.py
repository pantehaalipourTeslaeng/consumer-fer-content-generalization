"""Data generator and augmentation utilities."""

import collections
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

def get_class_weights(y, smooth_factor=0):
    """Calculate class weights for imbalanced datasets.
    
    Args:
        y (array-like): Array of class labels
        smooth_factor (float): Smoothing factor for weight calculation
        
    Returns:
        dict: Dictionary of class weights
    """
    counter = collections.Counter(y)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}

def create_data_generators(img_height=224, img_width=224, batch_size=16):
    """Create training and validation data generators with augmentation.
    
    Args:
        img_height (int): Target image height
        img_width (int): Target image width
        batch_size (int): Batch size for training
        
    Returns:
        tuple: Training and validation data generators
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    # Validation data generator without augmentation
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    return train_datagen, val_datagen

def setup_data_generators(train_dir, val_dir, img_height=224, img_width=224, batch_size=16):
    """Set up data generators for training and validation directories.
    
    Args:
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        img_height (int): Target image height
        img_width (int): Target image width
        batch_size (int): Batch size for training
        
    Returns:
        tuple: Configured training and validation generators
    """
    train_datagen, val_datagen = create_data_generators(img_height, img_width, batch_size)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator