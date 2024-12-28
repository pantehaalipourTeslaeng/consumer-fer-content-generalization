"""Extract frames from downloaded YouTube videos."""

import os
import cv2
import pandas as pd
import numpy as np
import logging
import yaml
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.data_processing.preprocess_utils import detect_face, crop_and_align_face

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameExtractor:
    """Handle video frame extraction operations."""
    
    def __init__(self, config_path):
        """Initialize FrameExtractor.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.config['paths']['frames_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
    
    def extract_frames(self, video_path, output_dir, metadata=None):
        """Extract frames from a video file.
        
        Args:
            video_path (str): Path to video file
            output_dir (str): Directory to save extracted frames
            metadata (dict, optional): Additional metadata for the frames
            
        Returns:
            dict: Extraction results and metadata
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            if total_frames == 0:
                raise ValueError("Empty video file")
            
            # Calculate frame extraction interval
            frame_interval = self.config['extraction']['frame_interval']
            if isinstance(frame_interval, str) and frame_interval.endswith('s'):
                # If interval is specified in seconds
                interval_seconds = float(frame_interval[:-1])
                frame_interval = int(interval_seconds * fps)
            else:
                frame_interval = int(frame_interval)
            
            extracted_frames = []
            frame_count = 0
            
            with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_count % frame_interval == 0:
                        # Detect faces
                        faces = detect_face(frame, self.config['face_detection']['cascade_path'])
                        
                        for i, face_rect in enumerate(faces):
                            # Crop and align face
                            face_img = crop_and_align_face(
                                frame, 
                                face_rect,
                                desired_size=tuple(self.config['face_detection']['output_size']),
                                padding=self.config['face_detection']['padding']
                            )
                            
                            # Generate output filename
                            timestamp = frame_count / fps
                            filename = f"frame_{frame_count:06d}_face_{i:02d}_{timestamp:.2f}s.jpg"
                            if metadata and 'video_id' in metadata:
                                filename = f"{metadata['video_id']}_{filename}"
                            
                            output_path = os.path.join(output_dir, filename)
                            
                            # Save face image
                            cv2.imwrite(output_path, face_img)
                            
                            # Record frame information
                            frame_info = {
                                'frame_path': output_path,
                                'frame_number': frame_count,
                                'timestamp': timestamp,
                                'face_number': i
                            }
                            if metadata:
                                frame_info.update(metadata)
                            
                            extracted_frames.append(frame_info)
                    
                    frame_count += 1
                    pbar.update(1)
            
            cap.release()
            
            return {
                'status': 'success',
                'video_path': video_path,
                'total_frames': total_frames,
                'extracted_frames': len(extracted_frames),
                'fps': fps,
                'frame_interval': frame_interval,
                'frames': extracted_frames,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            return {
                'status': 'failed',
                'video_path': video_path,
                'error': str(e),
                'frames': []
            }
    
    def process_download_results(self, results_path, max_workers=4):
        """Process downloaded videos and extract frames.
        
        Args:
            results_path (str): Path to download results CSV
            max_workers (int): Maximum number of concurrent processes
            
        Returns:
            pandas.DataFrame: Frame extraction results
        """
        # Read download results
        df = pd.read_csv(results_path)
        successful_downloads = df[df['status'] == 'success']
        
        all_frames = []
        
        # Process videos using process pool
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for _, row in successful_downloads.iterrows():
                video_path = row['output_path']
                category = row['category']
                output_dir = os.path.join(self.config['paths']['frames_dir'], category)
                os.makedirs(output_dir, exist_ok=True)
                
                # Prepare metadata
                metadata = {
                    'video_id': row['video_id'],
                    'category': category,
                    'reaction_type': row['reaction_type']
                }
                
                # Submit extraction task
                future = executor.submit(self.extract_frames, video_path, output_dir, metadata)
                futures.append(future)
            
            # Process results
            for future in tqdm(futures, desc="Extracting frames"):
                result = future.result()
                if result['status'] == 'success':
                    all_frames.extend(result['frames'])
        
        # Create results DataFrame
        frames_df = pd.DataFrame(all_frames)
        
        # Save results
        frames_path = os.path.join(self.config['paths']['log_dir'], 'frame_extraction_results.csv')
        frames_df.to_csv(frames_path, index=False)
        
        logger.info(f"Extracted {len(frames_df)} frames from {len(successful_downloads)} videos")
        
        return frames_df

def main():
    """Main function to run frame extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("--config", type=str, default="configs/youtube_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to download results CSV")
    parser.add_argument("--workers", type=int, default=4,
                      help="Number of concurrent processes")
    args = parser.parse_args()
    
    try:
        extractor = FrameExtractor(args.config)
        frames_df = extractor.process_download_results(args.input, args.workers)
        
        # Print summary
        print("\nFrame Extraction Summary:")
        print(f"Total frames extracted: {len(frames_df)}")
        print(f"Categories processed: {frames_df['category'].nunique()}")
        print("\nFrames per category:")
        print(frames_df['category'].value_counts())
        
    except Exception as e:
        logger.error(f"Error during frame extraction: {str(e)}")
        raise

if __name__ == "__main__":
    main()