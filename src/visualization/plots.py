"""Visualization tools for dataset analysis and statistics."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetVisualizer:
    """Visualize dataset statistics and distributions."""
    
    def __init__(self, save_dir=None):
        """Initialize DatasetVisualizer.
        
        Args:
            save_dir (str, optional): Directory to save visualization plots
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_data_distribution(self, data, column, title=None, figsize=(10, 6)):
        """Plot distribution of data in a specific column.
        
        Args:
            data (pandas.DataFrame): Dataset
            column (str): Column name to plot
            title (str, optional): Plot title
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        if data[column].dtype == 'object' or data[column].nunique() < 10:
            # Categorical data
            sns.countplot(data=data, x=column)
            plt.xticks(rotation=45)
        else:
            # Numerical data
            sns.histplot(data=data, x=column, bins=30)
        
        plt.title(title or f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, f'{column}_distribution.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_class_balance(self, data, class_column, split_column=None, 
                          figsize=(10, 6)):
        """Plot class distribution, optionally split by another column.
        
        Args:
            data (pandas.DataFrame): Dataset
            class_column (str): Column containing class labels
            split_column (str, optional): Column to split the data by
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        if split_column:
            # Stacked bar plot
            pd.crosstab(data[class_column], data[split_column]).plot(
                kind='bar', stacked=True)
            plt.title(f'Class Distribution by {split_column}')
            plt.xlabel(class_column)
            plt.ylabel('Count')
        else:
            # Simple bar plot
            sns.countplot(data=data, x=class_column)
            plt.title('Class Distribution')
            plt.xlabel(class_column)
            plt.ylabel('Count')
        
        plt.xticks(rotation=45)
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'class_balance.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_image_statistics(self, image_dir, figsize=(12, 4)):
        """Plot statistics about image dataset.
        
        Args:
            image_dir (str): Directory containing images
            figsize (tuple): Figure size
        """
        image_paths = list(Path(image_dir).rglob('*.jpg')) + \
                     list(Path(image_dir).rglob('*.png'))
        
        resolutions = []
        aspect_ratios = []
        file_sizes = []
        
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is not None:
                h, w = img.shape[:2]
                resolutions.append((w, h))
                aspect_ratios.append(w/h)
                file_sizes.append(os.path.getsize(path) / 1024)  # KB
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot resolution distribution
        resolutions = np.array(resolutions)
        ax1.scatter(resolutions[:, 0], resolutions[:, 1], alpha=0.5)
        ax1.set_title('Image Resolutions')
        ax1.set_xlabel('Width')
        ax1.set_ylabel('Height')
        
        # Plot aspect ratio distribution
        sns.histplot(aspect_ratios, bins=30, ax=ax2)
        ax2.set_title('Aspect Ratio Distribution')
        ax2.set_xlabel('Aspect Ratio')
        
        # Plot file size distribution
        sns.histplot(file_sizes, bins=30, ax=ax3)
        ax3.set_title('File Size Distribution')
        ax3.set_xlabel('File Size (KB)')
        
        plt.tight_layout()
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'image_statistics.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_data_splits(self, data, split_column, class_column=None, 
                        figsize=(10, 6)):
        """Plot distribution of data across different splits.
        
        Args:
            data (pandas.DataFrame): Dataset
            split_column (str): Column indicating data split
            class_column (str, optional): Column containing class labels
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        if class_column:
            # Stacked bar plot showing class distribution in each split
            pd.crosstab(data[split_column], data[class_column]).plot(
                kind='bar', stacked=True)
            plt.title(f'Class Distribution Across Data Splits')
            plt.xlabel('Data Split')
            plt.ylabel('Count')
            plt.legend(title=class_column)
        else:
            # Simple bar plot of split sizes
            sns.countplot(data=data, x=split_column)
            plt.title('Data Split Distribution')
            plt.xlabel('Data Split')
            plt.ylabel('Count')
        
        plt.xticks(rotation=45)
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'data_splits.png'))
            plt.close()
        else:
            plt.show()