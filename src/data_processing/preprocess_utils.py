"""Utility functions for data preprocessing."""

import os
import cv2
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories(base_path, directories):
    """Create necessary directories if they don't exist.
    
    Args:
        base_path (str): Base path for creating directories
        directories (list): List of directory names to create
    """
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

def detect_face(image, face_cascade_path):
    """Detect faces in an image using OpenCV cascade classifier.
    
    Args:
        image (numpy.ndarray): Input image
        face_cascade_path (str): Path to cascade classifier XML file
        
    Returns:
        list: List of detected face rectangles (x, y, w, h)
    """
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def crop_and_align_face(image, face_rect, desired_size=(224, 224), padding=0.2):
    """Crop and align detected face with padding.
    
    Args:
        image (numpy.ndarray): Input image
        face_rect (tuple): Face detection rectangle (x, y, w, h)
        desired_size (tuple): Desired output size (width, height)
        padding (float): Padding factor around face
        
    Returns:
        numpy.ndarray: Processed face image
    """
    x, y, w, h = face_rect
    
    # Add padding
    padding_x = int(w * padding)
    padding_y = int(h * padding)
    
    # Calculate padded coordinates
    x1 = max(x - padding_x, 0)
    y1 = max(y - padding_y, 0)
    x2 = min(x + w + padding_x, image.shape[1])
    y2 = min(y + h + padding_y, image.shape[0])
    
    # Crop face region
    face = image[y1:y2, x1:x2]
    
    # Resize to desired size
    face = cv2.resize(face, desired_size)
    
    return face

def save_image(image, output_path):
    """Save image to disk with error handling.
    
    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Path to save the image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {str(e)}")
        return False

def validate_image(image_path, min_size=(64, 64)):
    """Validate image file and dimensions.
    
    Args:
        image_path (str): Path to image file
        min_size (tuple): Minimum allowed dimensions (width, height)
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width < min_size[0] or height < min_size[1]:
                logger.warning(f"Image {image_path} is smaller than minimum size {min_size}")
                return False
            return True
    except Exception as e:
        logger.error(f"Error validating image {image_path}: {str(e)}")
        return False