"""Custom metrics for model evaluation."""

import tensorflow as tf
from tensorflow.keras.metrics import Metric, Precision, Recall

class F1Score(Metric):
    """F1 Score metric for binary classification."""
    
    def __init__(self, name='f1_score', **kwargs):
        """Initialize F1 Score metric.
        
        Args:
            name (str): Name of the metric
            **kwargs: Additional arguments to pass to parent class
        """
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the states of precision and recall metrics.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            sample_weight: Optional sample weights
        """
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        """Compute the F1 score from precision and recall.
        
        Returns:
            float: F1 score value
        """
        p = self.precision.result()
        r = self.recall.result()
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_states(self):
        """Reset the states of precision and recall metrics."""
        self.precision.reset_states()
        self.recall.reset_states()