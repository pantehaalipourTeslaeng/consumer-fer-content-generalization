"""Visualization tools for model metrics and performance analysis."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsVisualizer:
    """Visualize model training and evaluation metrics."""
    
    def __init__(self, save_dir=None):
        """Initialize MetricsVisualizer.
        
        Args:
            save_dir (str, optional): Directory to save visualization plots
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_training_history(self, history_df, metrics=None, figsize=(12, 6)):
        """Plot training history metrics.
        
        Args:
            history_df (pandas.DataFrame): Training history data
            metrics (list, optional): Specific metrics to plot
            figsize (tuple): Figure size
        """
        if metrics is None:
            # Exclude validation metrics
            metrics = [col for col in history_df.columns if not col.startswith('val_')]
        
        plt.figure(figsize=figsize)
        
        for metric in metrics:
            plt.plot(history_df[metric], label=f'Training {metric}')
            if f'val_{metric}' in history_df.columns:
                plt.plot(history_df[f'val_{metric}'], '--', 
                        label=f'Validation {metric}')
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, classes=None, normalize=True, 
                            figsize=(8, 6)):
        """Plot confusion matrix.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            classes (list, optional): Class names
            normalize (bool): Whether to normalize the matrix
            figsize (tuple): Figure size
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=classes, yticklabels=classes)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curves(self, y_true, y_pred_prob, classes=None, figsize=(8, 6)):
        """Plot ROC curves.
        
        Args:
            y_true (array-like): True labels
            y_pred_prob (array-like): Predicted probabilities
            classes (list, optional): Class names
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        if y_pred_prob.shape[1] == 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            
        else:  # Multi-class
            for i in range(y_pred_prob.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i, y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                
                label = f'Class {i}' if classes is None else classes[i]
                plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'roc_curves.png'))
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curves(self, y_true, y_pred_prob, classes=None, 
                                   figsize=(8, 6)):
        """Plot precision-recall curves.
        
        Args:
            y_true (array-like): True labels
            y_pred_prob (array-like): Predicted probabilities
            classes (list, optional): Class names
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        if y_pred_prob.shape[1] == 2:  # Binary classification
            precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, 
                    label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
            
        else:  # Multi-class
            for i in range(y_pred_prob.shape[1]):
                precision, recall, _ = precision_recall_curve(
                    y_true == i, y_pred_prob[:, i])
                pr_auc = auc(recall, precision)
                
                label = f'Class {i}' if classes is None else classes[i]
                plt.plot(recall, precision, 
                        label=f'{label} (AUC = {pr_auc:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        if self.save_dir:
            plt.savefig(os.path.join(self.save_dir, 'precision_recall_curves.png'))
            plt.close()
        else:
            plt.show()