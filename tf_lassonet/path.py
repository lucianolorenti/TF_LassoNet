from tf_lassonet.model import LassoNet
from typing import Optional, List, Union
from dataclasses import dataclass
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tqdm.auto import tqdm


@dataclass
class HistoryItem:
    lambda_: float
    objective: float  # loss + lambda_ * regulatization
    loss: float
    val_objective: float  # val_loss + lambda_ * regulatization
    val_loss: float
    regularization: float
    n_selected_features: int
    selected_features: np.ndarray
    n_iters: int
    test_predictions: np.ndarray
    weights: any


def compute_feature_importances(path: List[HistoryItem]):
    """When does each feature disappear on the path?
    Parameters
    ----------
    path : List[HistoryItem]
    Returns
    -------
        feature_importances_
    """

    current = path[0].selected_features.copy()
    ans = ans = np.full(current.shape, float("inf"))
    for save in path[1:]:
        lambda_ = save.lambda_
        diff = current & ~save.selected_features
        ans[diff.nonzero()] = lambda_
        current &= save.selected_features
    return ans


class LambdaSequence:
    def __init__(self, start: float, multiplier: float):
        self.start = start
        self.multiplier = multiplier
        self.curr_value = start

    def __iter__(self):
        self.curr_value = self.start
        return self

    def __next__(self):
        r = self.curr_value

        self.curr_value *= self.multiplier
        return r


class LassoPath:
    def __init__(
        self,
        model,
        n_iters_init: int,
        patience_init: int,
        n_iters_path: int,
        patience_path: int,
        lambda_seq: Optional[List[float]] = None,
        lambda_start: Optional[Union[float, str]] = None,
        path_multiplier: float = 1.02,
        M: float = 10,
        restore_best_weights: bool = False,
        store_weights: bool = True
    ):
        self.lassonet = LassoNet(model, M=M)

        self.n_iters_init = n_iters_init
        self.patience_init = patience_init
        self.n_iters_path = n_iters_path
        self.patience_path = patience_path
        self.lambda_seq = lambda_seq
        self.lambda_start = lambda_start
        self.path_multiplier = path_multiplier
        self.M = M
        
        self.restore_best_weights = restore_best_weights
        self.store_weights = store_weights



    def lambda_sequences(self):
        lambda_seq = self.lambda_seq
        if lambda_seq is None:
            if isinstance(self.lambda_start, str):
                if self.lambda_start == 'auto':
                    self.lambda_start = (
                        self.lassonet.lambda_start(M=self.M)
                        / self.lassonet.optimizer.learning_rate.numpy()
                        / 10
                    )

                else:
                    raise ValueError("Invalid lambda_start value, should be auto or a number")

            return LambdaSequence(self.lambda_start, self.path_multiplier)

        else:
            return lambda_seq

    def fit_one_model(
        self, train_dataset, val_dataset, *, epochs:int, test_dataset=None, lambda_,  **kwargs
    ) -> HistoryItem:
        self.lassonet.lambda_.assign(lambda_)

        history = self.lassonet.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[
                EarlyStopping(
                    patience=self.patience_init,
                    restore_best_weights=self.restore_best_weights,
                )
            ],
            verbose=False,
            **kwargs
        )

        reg = self.lassonet.regularization()
        val_loss = self.lassonet.evaluate(val_dataset, verbose=False)

        test_predictions = None
        if test_dataset is not None:            
            test_predictions = self.lassonet.predict(test_dataset)
        return HistoryItem(
            lambda_=lambda_,
            loss=history.history["loss"][-1],
            objective=history.history["loss"][-1] + lambda_ * reg,
            val_loss=val_loss,
            val_objective=val_loss + lambda_ * reg,
            regularization=reg.numpy(),
            n_iters=len(history.history["loss"]),
            n_selected_features=self.lassonet.selected_count().numpy(),
            selected_features=self.lassonet.input_mask().numpy(),
            test_predictions=test_predictions,
            weights=self.lassonet.get_weights() if self.store_weights else None
        )

    def _update_bar(self,  bar, h, lambda_: float):
        bar.update(1)
        bar.set_postfix(
            {
                "Lambda": lambda_,
                "Val loss": h.val_loss[0],
                "Selected features": h.n_selected_features[0],
                "Regularization": h.regularization,
            }
        ),

    def fit(
        self, train_dataset, val_dataset, verbose: bool = False, **kwargs
    ) -> List[HistoryItem]:
        self.history = []
        if verbose:
            bar = tqdm()
            bar.update(0)

        h = self.fit_one_model(train_dataset, val_dataset, epochs=self.n_iters_init, lambda_=0, **kwargs)
        self.history.append(h)

        if verbose:
            self._update_bar(bar, h, 0)

        for i, current_lambda in enumerate(self.lambda_sequences()):
            
            h = self.fit_one_model(
                train_dataset, val_dataset, epochs=self.n_iters_path, lambda_=current_lambda, **kwargs
            )
            self.history.append(h)
            finalize = self.lassonet.selected_count()[0] == 0
            
            if verbose:
            
                self._update_bar( bar, h, current_lambda)
                if finalize:
                    bar.close()

            if finalize:
                break
        return self.history

    def compute_feature_importances(self):
        """When does each feature disappear on the path?
        Parameters
        ----------
        path : List[HistoryItem]
        Returns
        -------
            feature_importances_
        """
        return compute_feature_importances(self.history)
