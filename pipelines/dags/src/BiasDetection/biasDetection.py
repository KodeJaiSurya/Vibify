# Fairness Analysis install fairlearn
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score


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

    # Create a MetricFrame to analyze accuracy and selection rate by sensitive features
    metric_frame = MetricFrame(metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
                               y_true=y_test, y_pred=y_pred, sensitive_features=slices)
    return metric_frame