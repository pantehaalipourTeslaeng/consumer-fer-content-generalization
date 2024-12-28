"""Validate generated synthetic dataset."""

import os
import argparse
import yaml
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SyntheticDataValidator:
    """Validator for synthetic image dataset."""
    
    def __init__(self, config_path):
        """Initialize validator.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.metrics = defaultdict(list)
    
    def validate_dataset(self, metadata_path):
        """Validate the entire synthetic dataset.
        
        Args:
            metadata_path (str): Path to dataset metadata CSV
            
        Returns:
            dict: Validation metrics
        """
        # Load metadata
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata for {len(metadata_df)} images")
        
        # Validate each category
        for category in self.config['categories']:
            category_df = metadata_df[metadata_df['category'] == category]
            logger.info(f"Validating {len(category_df)} images for category: {category}")
            
            self.validate_category(category_df, category)
        
        # Compute and return overall metrics
        return self.compute_metrics()
    
    def validate_category(self, category_df, category_name):
        """Validate images for a specific category.
        
        Args:
            category_df (pandas.DataFrame): DataFrame containing category images
            category_name (str): Name of the category
        """
        for _, row in tqdm(category_df.iterrows(), total=len(category_df)):
            image_path = row['image_path']
            
            try:
                # Load and validate image
                image = Image.open(image_path)
                
                # Check image properties
                self.validate_image_properties(image, category_name)
                
            except Exception as e:
                logger.error(f"Error validating {image_path}: {str(e)}")
                self.metrics['failed_images'].append(image_path)
    
    def validate_image_properties(self, image, category):
        """Validate properties of a single image.
        
        Args:
            image (PIL.Image): Image to validate
            category (str): Category name
        """
        # Check resolution
        width, height = image.size
        self.metrics['resolutions'].append((width, height))
        
        # Check aspect ratio
        aspect_ratio = width / height
        self.metrics['aspect_ratios'].append(aspect_ratio)
        
        # Check color statistics
        if image.mode == 'RGB':
            image_array = np.array(image)
            for channel in range(3):
                channel_mean = image_array[:, :, channel].mean()
                channel_std = image_array[:, :, channel].std()
                self.metrics[f'channel_{channel}_mean'].append(channel_mean)
                self.metrics[f'channel_{channel}_std'].append(channel_std)
    
    def compute_metrics(self):
        """Compute summary metrics for the dataset.
        
        Returns:
            dict: Summary metrics
        """
        summary = {}
        
        # Resolution statistics
        resolutions = np.array(self.metrics['resolutions'])
        summary['min_resolution'] = resolutions.min(axis=0)
        summary['max_resolution'] = resolutions.max(axis=0)
        summary['mean_resolution'] = resolutions.mean(axis=0)
        
        # Aspect ratio statistics
        aspect_ratios = np.array(self.metrics['aspect_ratios'])
        summary['aspect_ratio_mean'] = aspect_ratios.mean()
        summary['aspect_ratio_std'] = aspect_ratios.std()
        
        # Color statistics
        for channel in range(3):
            means = np.array(self.metrics[f'channel_{channel}_mean'])
            stds = np.array(self.metrics[f'channel_{channel}_std'])
            summary[f'channel_{channel}_mean_avg'] = means.mean()
            summary[f'channel_{channel}_std_avg'] = stds.mean()
        
        # Failure statistics
        summary['total_failed'] = len(self.metrics['failed_images'])
        
        return summary

def main():
    """Main function for validating synthetic dataset."""
    parser = argparse.ArgumentParser(
        description="Validate synthetic facial expression dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stylegan_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to dataset metadata CSV"
    )
    args = parser.parse_args()
    
    try:
        # Run validation
        validator = SyntheticDataValidator(args.config)
        metrics = validator.validate_dataset(args.metadata)
        
        # Print summary
        logger.info("Validation Summary:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")
        
        # Save metrics
        metrics_path = os.path.join(
            os.path.dirname(args.metadata),
            'validation_metrics.yaml'
        )
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f)
        logger.info(f"Saved validation metrics to: {metrics_path}")
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise

if __name__ == "__main__":
    main()