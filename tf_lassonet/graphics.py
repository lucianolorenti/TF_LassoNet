from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt


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