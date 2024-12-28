"""Generate and process synthetic facial expression dataset."""

import os
import argparse
import yaml
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from stylegan2_generator import StyleGAN2Generator
from image_processor import SyntheticImageProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_dataset(config_path):
    """Generate and process synthetic dataset.
    
    Args:
        config_path (str): Path to configuration file
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set up directories
    raw_output_dir = os.path.join(config['paths']['output_dir'], 'raw')
    processed_output_dir = os.path.join(config['paths']['output_dir'], 'processed')
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)
    
    # Initialize generator and processor
    generator = StyleGAN2Generator(config_path)
    processor = SyntheticImageProcessor(config_path)
    
    metadata = []
    
    # Generate images for each category
    for category in config['categories']:
        logger.info(f"Generating images for category: {category}")
        
        # Create category directories
        category_raw_dir = os.path.join(raw_output_dir, category)
        category_processed_dir = os.path.join(processed_output_dir, category)
        os.makedirs(category_raw_dir, exist_ok=True)
        os.makedirs(category_processed_dir, exist_ok=True)
        
        # Generate raw images
        n_images = config['generation']['n_samples']
        generator.save_generated_images(
            n_images=n_images,
            output_dir=category_raw_dir,
            category=category
        )
        
        # Process generated images
        logger.info(f"Processing generated images for category: {category}")
        processor.process_directory(
            input_dir=category_raw_dir,
            output_dir=category_processed_dir
        )
        
        # Collect metadata for successfully processed images
        with open(os.path.join(category_processed_dir, 'valid_images.txt'), 'r') as f:
            valid_images = f.read().splitlines()
            
        for image_name in valid_images:
            metadata.append({
                'image_path': os.path.join(category_processed_dir, image_name),
                'category': category,
                'is_synthetic': True
            })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(config['paths']['output_dir'], 'synthetic_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    return metadata_df

def main():
    """Main function for generating synthetic dataset."""
    parser = argparse.ArgumentParser(
        description="Generate and process synthetic facial expression dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stylegan_config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    try:
        metadata_df = generate_synthetic_dataset(args.config)
        logger.info(f"Successfully generated {len(metadata_df)} synthetic images")
    except Exception as e:
        logger.error(f"Error generating synthetic dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()