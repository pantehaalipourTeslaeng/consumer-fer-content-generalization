"""StyleGAN2 generator for synthetic facial expression data."""

import os
import numpy as np
import tensorflow as tf
import yaml
import logging
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
            
        self.model = self._build_generator()
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.config['paths']['checkpoint_dir'],
            self.config['paths']['output_dir']
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def _build_generator(self):
        """Build StyleGAN2 generator model.
        
        Returns:
            tf.keras.Model: StyleGAN2 generator model
        """
        # This is a simplified version. In practice, you would load
        # a pre-trained StyleGAN2 model or implement the full architecture
        latent_size = self.config['model']['latent_size']
        resolution = self.config['model']['resolution']
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(latent_size,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((8, 8, 256)),
            
            # Upsampling blocks
            *self._create_upsampling_blocks(),
            
            tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')
        ])
        
        return model
    
    def _create_upsampling_blocks(self):
        """Create upsampling blocks for the generator.
        
        Returns:
            list: List of upsampling layers
        """
        blocks = []
        current_res = 8
        target_res = self.config['model']['resolution']
        
        while current_res < target_res:
            blocks.extend([
                tf.keras.layers.Conv2DTranspose(
                    current_res * 2, kernel_size=4, strides=2, padding='same', use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU()
            ])
            current_res *= 2
            
        return blocks
    
    def generate_images(self, n_images, category=None):
        """Generate synthetic images.
        
        Args:
            n_images (int): Number of images to generate
            category (str, optional): Category for conditional generation
            
        Returns:
            numpy.ndarray: Generated images
        """
        latent_vectors = self._sample_latent_vectors(n_images)
        images = self.model(latent_vectors, training=False)
        return images.numpy()
    
    def _sample_latent_vectors(self, n_samples):
        """Sample latent vectors for image generation.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            tf.Tensor: Sampled latent vectors
        """
        return tf.random.normal([n_samples, self.config['model']['latent_size']])
    
    def save_generated_images(self, n_images, output_dir, category=None):
        """Generate and save synthetic images.
        
        Args:
            n_images (int): Number of images to generate
            output_dir (str): Directory to save generated images
            category (str, optional): Category for conditional generation
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Generating {n_images} synthetic images...")
        batch_size = self.config['training']['batch_size']
        
        for i in tqdm(range(0, n_images, batch_size)):
            current_batch_size = min(batch_size, n_images - i)
            images = self.generate_images(current_batch_size, category)
            
            # Save images
            for j, image in enumerate(images):
                image = ((image + 1) * 127.5).astype(np.uint8)  # Denormalize
                image_path = os.path.join(
                    output_dir,
                    f"synthetic_{category}_{i+j:06d}.png"
                )
                tf.keras.preprocessing.image.save_img(image_path, image)

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