from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from tf_lassonet.model import LassoNet
from tf_lassonet.path import HistoryItem, LassoPath

from tf_lassonet.utils import eval_on_path


def feature_importance_histogram(
    feature_importances: np.ndarray,
    feature_names: Optional[List[str]] = None,
    horizontal: bool = True,
    N:Optional[int] = None,
    ax=None,
    **kwargs
):
    if ax is None:
        _, ax = plt.subplots(**kwargs)
    importance_indices = np.argsort(feature_importances)
    x_ticks = []
    labels = []
    if N is None:
        N = feature_importances.shape[0]
    for i, j in enumerate(importance_indices[::-1][:N][::-1]):
        if feature_names is not None:
            labels.append(feature_names[j])
        if horizontal:
            ax.barh(
                y=i,
                width=feature_importances[j],
                height=1,
                edgecolor="#4477DD",
                facecolor="blue",
            )
        else:
            ax.bar(
                x=i,
                height=feature_importances[j],
                width=1,
                edgecolor="#4477DD",
                facecolor="blue",
            )
        x_ticks.append(i)
    if feature_names is not None:
        if horizontal:
            ax.set_yticks(x_ticks)
            ax.set_yticklabels(labels)
        else:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(labels)
    ax.set_ylabel("Importance")
    ax.set_xlabel("Feature")
    return ax





def plot_path(model:LassoNet, path:List[HistoryItem], ds_test, *, score_function=None, ax=None, **kwargs):
    """
    Plot the evolution of the model on the path, namely:
    - lambda
    - number of selected variables
    - score


    Parameters
    ==========
    model : LassoNetClassifier or LassoNetRegressor
    path
        output of model.path
    X_test : array-like
    y_test : array-like
    score_function : function or None
        if None, use score_function=model.score
        score_function must take as input X_test, y_test
    """
    # TODO: plot with manually computed score
    
    score = eval_on_path(model, path, ds_test, score_function=score_function)
    n_selected = [save.selected_features.sum() for save in path]
    lambda_ = [save.lambda_ for save in path]
    if ax is None:
        fig, ax = plt.subplots(1,3, **kwargs)

    
    ax[0].plot(n_selected, score, ".-")
    ax[0].set_xlabel("number of selected features")
    ax[1].set_ylabel("score")

    
    ax[1].plot(lambda_, score, ".-")
    ax[1].set_xlabel("lambda")
    ax[1].set_xscale("log")
    ax[1].set_ylabel("score")

    ax[2].plot(lambda_, n_selected, ".-")
    ax[2].set_xlabel("lambda")
    ax[2].set_xscale("log")
    ax[2].set_ylabel("number of selected features")

    fig.tight_layout()
    return ax


def plot_cv(model, X_test, y_test, *, score_function=None):
    # TODO: plot with manually computed score
    lambda_ = [save.lambda_ for save in model.path_]
    lambdas = [[h.lambda_ for h in p] for p in model.raw_paths_]

    score = eval_on_path(
        model, model.path_, X_test, y_test, score_function=score_function
    )

    plt.figure(figsize=(16, 16))

    plt.subplot(211)
    plt.grid(True)
    first = True
    for sl, ss in zip(lambdas, model.raw_scores_):
        plt.plot(
            sl,
            ss,
            "r.-",
            markersize=5,
            alpha=0.2,
            label="cross-validation" if first else None,
        )
        first = False
    avg = model.interp_scores_.mean(axis=1)
    ci = confidence_interval(model.interp_scores_)
    plt.plot(
        model.lambdas_,
        avg,
        "g.-",
        markersize=5,
        alpha=0.2,
        label="average cv with 95% CI",
    )
    plt.fill_between(model.lambdas_, avg - ci, avg + ci, color="g", alpha=0.1)
    plt.plot(lambda_, score, "b.-", markersize=5, alpha=0.2, label="test")
    plt.legend()
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("score")

    plt.subplot(212)
    plt.grid(True)
    first = True
    for sl, path in zip(lambdas, model.raw_paths_):
        plt.plot(
            sl,
            [save.selected.sum() for save in path],
            "r.-",
            markersize=5,
            alpha=0.2,
            label="cross-validation" if first else None,
        )
        first = False
    plt.plot(
        lambda_,
        [save.selected.sum() for save in model.path_],
        "b.-",
        markersize=5,
        alpha=0.2,
        label="test",
    )
    plt.legend()
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.tight_layout()