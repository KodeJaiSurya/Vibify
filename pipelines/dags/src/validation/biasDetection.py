# Fairness Analysis install fairlearn
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score, f1_score
import logging


def fairness_analysis(y_test, y_pred, data, indices_test):
    """
    Performs fairness analysis on the model's predictions using Fairlearn.

    Args:
        y_test (np.ndarray): True labels for the test set.
        y_pred (np.ndarray): Predicted labels for the test set.
        data (pd.DataFrame): Original DataFrame containing sensitive feature columns.
        indices_test (pd.Index): Indices for test data rows.

    Returns:
        MetricFrame: MetricFrame object with fairness metrics by group.
    """
    # Define sensitive feature slices (e.g., age group and gender)
    slices = {
        'emotion': data.loc[indices_test, 'emotion'],
        # 'age_group': data.loc[indices_test, 'age_group'],
        'gender': data.loc[indices_test, 'gender']
    }

    # Define metric functions
    metrics = {
        'accuracy': accuracy_score,
        'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        'selection_rate': selection_rate
    }

    # Create MetricFrame to analyze metrics by slices
    metric_frame = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=slices)

    # Convert the MetricFrame results to a DataFrame
    results = metric_frame.by_group.reset_index()
    
    max_metrics = results[['accuracy', 'f1_score', 'selection_rate']].max()
    relative_threshold = 0.5  # Flag if below 50% of the best-performing group

    def flag_bias(row):
        for metric in ['accuracy', 'f1_score']:
            if row[metric] < max_metrics[metric] * relative_threshold:
                return 1  # Flag as biased
        return 0  # No bias

    # Bias Alert Flag
    results['bias'] = results.apply(flag_bias, axis=1)
    logging.info("Bias Validation Results: \n", results)
    if 1 in results['bias']:
      return 1
    else:
      return 0