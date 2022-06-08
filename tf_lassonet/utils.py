import numpy as np

from tf_lassonet.model import LassoNet
from tf_lassonet.path import LassoPath

from typing import Iterable

def feature_importance_time_series(feature_importances: np.ndarray, window_size: int):
    return np.mean(
        feature_importances.reshape(window_size, int(feature_importances.shape[1] / window_size)), axis=0
    )





def eval_on_path(model: LassoNet, path: LassoPath, ds_test, *, score_function=None):
    if score_function is None:
        score_fun = model.score
    else:
        assert callable(score_function)

        def score_fun(ds_test):
            y_test = np.squeeze(list(map(lambda x: x[1], ds_test)))
            return score_function(y_test, np.squeeze(model.predict(ds_test)))

    score = []
    for save in path:

        model.set_weights(save.weights)
        score.append(score_fun(ds_test))
    return score


def scatter_logsumexp(input, index, *, dim=-1, output_size=None):
    """Inspired by torch_scatter.logsumexp
    Uses torch.scatter_reduce for performance
    """
    max_value_per_index = torch.scatter_reduce(
        input, dim=dim, index=index, output_size=output_size, reduce="amax"
    )
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores = input - max_per_src_element
    sum_per_index = torch.scatter_reduce(
        recentered_scores.exp(),
        dim=dim,
        index=index,
        output_size=output_size,
        reduce="sum",
    )
    return max_value_per_index + sum_per_index.log()


def log_substract(x, y):
    """log(exp(x) - exp(y))"""
    return x + torch.log1p(-(y - x).exp())


def confidence_interval(data, confidence=0.95):
    if isinstance(data[0], Iterable):
        return [confidence_interval(d, confidence) for d in data]
    return scipy.stats.t.interval(
        confidence,
        len(data) - 1,
        scale=scipy.stats.sem(data),
    )[1]