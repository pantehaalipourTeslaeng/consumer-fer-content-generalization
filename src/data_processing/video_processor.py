"""Video processing module for extracting frames and faces."""

import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from .preprocess_utils import detect_face, crop_and_align_face, save_image, validate_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process videos to extract frames and detect faces."""
    
    def __init__(self, face_cascade_path, output_size=(224, 224), frame_interval=10):
        """Initialize VideoProcessor.
        
        Args:
            face_cascade_path (str): Path to cascade classifier XML file
            output_size (tuple): Output size for face images
            frame_interval (int): Number of frames to skip between extractions
        """
        self.face_cascade_path = face_cascade_path
        self.output_size = output_size
        self.frame_interval = frame_interval
        
    def process_video(self, video_path, output_dir, participant_id=None, metadata=None):
        """Process a single video file.
        
        Args:
            video_path (str): Path to input video file
            output_dir (str): Directory to save extracted frames
            participant_id (str): Optional participant identifier
            metadata (dict): Optional metadata for the video
            
        Returns:
            list: List of dictionaries containing frame information
        """
        frame_info = []
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                logger.error(f"Empty video file: {video_path}")
                return frame_info
            
            logger.info(f"Processing video: {video_path}")
            frame_count = 0
            
            with tqdm(total=total_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_count % self.frame_interval == 0:
                        # Detect faces
                        faces = detect_face(frame, self.face_cascade_path)
                        
                        for i, face_rect in enumerate(faces):
                            # Crop and align face
                            face_img = crop_and_align_face(
                                frame, face_rect, self.output_size)
                            
                            # Generate output filename
                            filename = f"frame_{frame_count:06d}_face_{i:02d}.jpg"
                            if participant_id:
                                filename = f"{participant_id}_{filename}"
                            
                            output_path = os.path.join(output_dir, filename)
                            
                            # Save face image
                            if save_image(face_img, output_path):
                                # Record frame information
                                info = {
                                    'video_path': video_path,
                                    'frame_number': frame_count,
                                    'timestamp': frame_count / fps,
                                    'face_number': i,
                                    'output_path': output_path,
                                    'participant_id': participant_id
                                }
                                if metadata:
                                    info.update(metadata)
                                frame_info.append(info)
                    
                    frame_count += 1
                    pbar.update(1)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            
        return frame_info
    
    def process_video_directory(self, input_dir, output_dir, metadata_file=None):
        """Process all videos in a directory.
        
        Args:
            input_dir (str): Directory containing input videos
            output_dir (str): Directory to save extracted frames
            metadata_file (str): Optional path to metadata CSV file
            
        Returns:
            pandas.DataFrame: DataFrame containing information about all processed frames
        """
        # Load metadata if provided
        metadata_dict = {}
        if metadata_file and os.path.exists(metadata_file):
            metadata_df = pd.read_csv(metadata_file)
            metadata_dict = metadata_df.set_index('video_id').to_dict('index')
        
        all_frame_info = []
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        
        # Process each video file
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(video_extensions):
                video_path = os.path.join(input_dir, filename)
                video_id = os.path.splitext(filename)[0]
                
                # Get metadata for this video if available
                metadata = metadata_dict.get(video_id, {})
                
                # Process video and collect frame information
                frame_info = self.process_video(
                    video_path,
                    output_dir,
                    participant_id=video_id,
                    metadata=metadata
                )
                all_frame_info.extend(frame_info)
        
        # Create DataFrame with all frame information
        df = pd.DataFrame(all_frame_info)
        
        # Save frame information to CSV
        output_csv = os.path.join(output_dir, 'frame_info.csv')
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved frame information to {output_csv}")
        
        return df