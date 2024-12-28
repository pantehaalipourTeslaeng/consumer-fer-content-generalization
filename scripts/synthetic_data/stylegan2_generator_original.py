"""StyleGAN2 generator for synthetic facial expression data using official implementation."""

import os
import numpy as np
import tensorflow as tf
import yaml
import logging
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StyleGAN2Generator:
    def __init__(self, config_path):
        """Initialize StyleGAN2 generator.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Initialize TensorFlow
        tflib.init_tf()
        
        # Load pre-trained model
        self.load_pretrained_model()
        
        self._setup_directories()
        
    def load_pretrained_model(self):
        """Load pre-trained StyleGAN2 model."""
        model_path = self.config['paths'].get('pretrained_model')
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Pre-trained model not found at: {model_path}")
            
        self.Gs = tflib.Network('Gs', 
                               num_channels=3,
                               resolution=self.config['model']['resolution'],
                               **dnnlib.EasyDict(tflib.Network.get_default_vars()))
        self.Gs.copy_vars_from_source(model_path)
            
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.config['paths']['checkpoint_dir'],
            self.config['paths']['output_dir']
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def generate_images(self, n_images, truncation_psi=0.7, seed=None):
        """Generate synthetic images.
        
        Args:
            n_images (int): Number of images to generate
            truncation_psi (float): Truncation psi value for controlling variation
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            numpy.ndarray: Generated images
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate latent vectors
        latents = np.random.randn(n_images, self.Gs.input_shape[1])
        
        # Generate images
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = self.Gs.run(latents, None, truncation_psi=truncation_psi, 
                           randomize_noise=True, output_transform=fmt)
        
        return images
    
    def save_generated_images(self, n_images, output_dir, category=None):
        """Generate and save synthetic images.
        
        Args:
            n_images (int): Number of images to generate
            output_dir (str): Directory to save generated images
            category (str, optional): Category for conditional generation
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get generation parameters from config
        truncation_psi = self.config['generation'].get('truncation_psi', 0.7)
        seed = self.config['generation'].get('seed', None)
        batch_size = self.config['training'].get('batch_size', 4)
        
        logger.info(f"Generating {n_images} synthetic images...")
        
        for i in tqdm(range(0, n_images, batch_size)):
            current_batch_size = min(batch_size, n_images - i)
            
            # Generate images
            images = self.generate_images(
                current_batch_size, 
                truncation_psi=truncation_psi,
                seed=seed + i if seed is not None else None
            )
            
            # Save images
            for j, image in enumerate(images):
                image_path = os.path.join(
                    output_dir,
                    f"synthetic_{category}_{i+j:06d}.png"
                )
                tf.io.write_file(image_path, tf.image.encode_png(image))

def main():
    """Main function to run StyleGAN2 generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic facial expression data")
    parser.add_argument("--config", type=str, default="configs/stylegan_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--n_images", type=int, help="Number of images to generate")
    parser.add_argument("--category", type=str, help="Category for conditional generation")
    args = parser.parse_args()
    
    # Initialize generator
    generator = StyleGAN2Generator(args.config)
    
    # Generate images
    n_images = args.n_images or generator.config['generation']['n_samples']
    output_dir = generator.config['paths']['output_dir']
    
    if args.category:
        output_dir = os.path.join(output_dir, args.category)
    
    generator.save_generated_images(n_images, output_dir, args.category)
    logger.info("Image generation completed successfully!")

if __name__ == "__main__":
    main()