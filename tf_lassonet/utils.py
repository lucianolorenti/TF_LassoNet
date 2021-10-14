import numpy as np


def feature_importance_time_series(feature_importances: np.ndarray, window_size: int):
    return np.mean(
        feature_importances.reshape(window_size, int(feature_importances.shape[1] / window_size)), axis=0
    )
