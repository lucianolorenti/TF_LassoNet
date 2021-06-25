from tf_lassonet.model import LassoNet
from typing import Optional, List
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np 


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

def compute_feature_importances(path: List[HistoryItem]):
    """When does each feature disappear on the path?
    Parameters
    ----------
    path : List[HistoryItem]
    Returns
    -------
        feature_importances_
    """

    current = path[0].selected_features.clone()
    ans = np.fill(path[0].selected_features)     
    for save in path[1:]:
        lambda_ = save.lambda_
        diff = current & ~save.selected_features
        ans[diff.nonzero().flatten()] = lambda_
        current &= save.selected
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
        lambda_start: Optional[float] = None,
        path_multiplier:float=1.02,
        M:float=10,
        eps_start:float=1,
    ):
        self.lassonet = LassoNet(model, M=M)
        self.lassonet.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        self.n_iters_init = n_iters_init
        self.patience_init = patience_init
        self.n_iters_path = n_iters_path
        self.patience_path = patience_path
        self.lambda_seq = lambda_seq
        self.lambda_start = lambda_start
        self.path_multiplier = path_multiplier
        self.eps_start = eps_start
   

    def lambda_sequences(self, history: List[HistoryItem]):
        lambda_seq = self.lambda_seq
        if lambda_seq is None:
            start = (
                self.lambda_start
                if self.lambda_start is not None
                else (self.eps_start * history[-1].val_loss)
            )
            return LambdaSequence(start, self.path_multiplier)

        else:
            return lambda_seq

    def fit_one_model(self, train_dataset, val_dataset, *, lambda_) -> HistoryItem:
        self.lassonet.lambda_.assign( lambda_)

        history = self.lassonet.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.n_iters_init,
            callbacks=[EarlyStopping(patience=self.patience_init)],
            verbose=False,
        )
       
        reg = self.lassonet.regularization()
        return HistoryItem(
            lambda_=lambda_,
            loss=history.history["loss"][-1],
            objective=history.history["loss"][-1] + lambda_ * reg,
            val_loss=history.history["val_loss"][-1],
            val_objective=history.history["val_loss"][-1] + lambda_ * reg,
            regularization=reg,
            n_iters=len(history.history["loss"]),
            n_selected_features=self.lassonet.selected_count(),
            selected_features=self.lassonet.input_mask()
        )

    def report(self, h):
        print(f"""Val Loss: {h.val_loss:.3} | Selected features: {h.n_selected_features} | Regularization: {h.regularization} """)
        

    def fit(
        self,  train_dataset, val_dataset
    ) -> List[HistoryItem]:
        self.history = []
        print(f'Lambda: {0}')           
        self.history.append(self.fit_one_model(train_dataset, val_dataset, lambda_=0))
        self.report(self.history[-1])
        

        for current_lambda in self.lambda_sequences(self.history):
            if self.lassonet.selected_count()[0] == 0:
                break
            print(f'Lambda: {current_lambda}')           

    
            h = self.fit_one_model(train_dataset, val_dataset, lambda_=current_lambda)
            self.report(h)
            self.history.append(h)
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
        return compute_feature_importance(self.history)
        

      