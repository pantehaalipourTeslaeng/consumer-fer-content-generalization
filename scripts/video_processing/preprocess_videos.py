"""Script to run video preprocessing pipeline."""

import os
import argparse
import logging
from pathlib import Path
import yaml
from src.data_processing.video_processor import VideoProcessor
from src.data_processing.preprocess_utils import setup_directories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess videos for FER")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing_config.yaml",
        help="Path to preprocessing configuration file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save processed frames"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to metadata CSV file"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Run the preprocessing pipeline."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.input_dir:
        config['input_dir'] = args.input_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.metadata:
        config['metadata_file'] = args.metadata
    
    # Set up directories
    setup_directories(config['output_dir'], ['frames', 'logs'])
    
    # Initialize video processor
    processor = VideoProcessor(
        face_cascade_path=config['face_cascade_path'],
        output_size=tuple(config['output_size']),
        frame_interval=config['frame_interval']
    )
    
    # Process videos
    logger.info("Starting video processing...")
    frame_info_df = processor.process_video_directory(
        input_dir=config['input_dir'],
        output_dir=os.path.join(config['output_dir'], 'frames'),
        metadata_file=config.get('metadata_file')
    )
    
    logger.info(f"Processed {len(frame_info_df)} frames")
    logger.info("Preprocessing completed successfully")

if __name__ == "__main__":
    main()