"""Download YouTube videos based on provided URLs."""

import os
import pandas as pd
import logging
import yaml
from pathlib import Path
from pytube import YouTube
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeDownloader:
    """Handle YouTube video downloading operations."""
    
    def __init__(self, config_path):
        """Initialize YouTubeDownloader.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.config['paths']['download_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
        
    def download_video(self, url, output_path=None):
        """Download a single YouTube video.
        
        Args:
            url (str): YouTube video URL
            output_path (str, optional): Path to save the video
            
        Returns:
            dict: Download status and information
        """
        try:
            # Create YouTube object
            yt = YouTube(url)
            
            # Get the highest resolution stream
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not stream:
                raise ValueError("No suitable stream found")
                
            # Generate output filename if not provided
            if output_path is None:
                filename = f"{yt.video_id}.mp4"
                output_path = os.path.join(self.config['paths']['download_dir'], filename)
            
            # Download the video
            stream.download(filename=output_path)
            
            return {
                'status': 'success',
                'url': url,
                'title': yt.title,
                'video_id': yt.video_id,
                'output_path': output_path,
                'resolution': stream.resolution,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return {
                'status': 'failed',
                'url': url,
                'title': None,
                'video_id': None,
                'output_path': None,
                'resolution': None,
                'error': str(e)
            }
            
    def process_video_list(self, csv_path, max_workers=4):
        """Download videos from a CSV file containing URLs.
        
        Args:
            csv_path (str): Path to CSV file containing video URLs
            max_workers (int): Maximum number of concurrent downloads
            
        Returns:
            pandas.DataFrame: Download results
        """
        # Read video URLs
        df = pd.read_csv(csv_path)
        required_columns = ['url', 'category', 'reaction_type']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        results = []
        
        # Download videos using thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for _, row in df.iterrows():
                url = row['url']
                category = row['category']
                output_dir = os.path.join(self.config['paths']['download_dir'], category)
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate output path
                output_path = os.path.join(output_dir, f"{url.split('=')[-1]}.mp4")
                
                # Submit download task
                future = executor.submit(self.download_video, url, output_path)
                futures.append((future, row))
            
            # Process results with progress bar
            for future, row in tqdm(futures, desc="Downloading videos"):
                result = future.result()
                result.update({
                    'category': row['category'],
                    'reaction_type': row['reaction_type']
                })
                results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(self.config['paths']['log_dir'], 'download_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Log summary
        success_count = len(results_df[results_df['status'] == 'success'])
        total_count = len(results_df)
        logger.info(f"Successfully downloaded {success_count}/{total_count} videos")
        
        return results_df

def main():
    """Main function to run YouTube video downloading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download YouTube videos")
    parser.add_argument("--config", type=str, default="configs/youtube_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to CSV file containing video URLs")
    parser.add_argument("--workers", type=int, default=4,
                      help="Number of concurrent downloads")
    args = parser.parse_args()
    
    try:
        downloader = YouTubeDownloader(args.config)
        results_df = downloader.process_video_list(args.input, args.workers)
        
        # Print summary
        print("\nDownload Summary:")
        print(f"Total videos: {len(results_df)}")
        print(f"Successful downloads: {len(results_df[results_df['status'] == 'success'])}")
        print(f"Failed downloads: {len(results_df[results_df['status'] == 'failed'])}")
        
    except Exception as e:
        logger.error(f"Error during download process: {str(e)}")
        raise

if __name__ == "__main__":
    main()