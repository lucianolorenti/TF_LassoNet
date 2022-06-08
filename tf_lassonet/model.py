from tf_lassonet.proximal import hier_prox_group
from typing import Optional, List
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf

from dataclasses import dataclass


class LassoNet(tf.keras.Model):
    def __init__(self, model: Model, lambda_: float = 1.0, M:float=1):
        super().__init__()
        self.model = model
        input_n_dim = np.prod(self.model.input.shape[1:])
        output_dim = np.prod(model.layers[-1].output_shape[1:])

       
        self.skip = Dense(output_dim, use_bias=False)
      

        self.W = Dense(input_n_dim, use_bias=False)
        self.lambda_ = tf.Variable(lambda_, trainable=False)
        self.M = M
       

    def regularization(self):
        return tf.math.reduce_sum(tf.norm(self.skip.weights, ord='euclidean', axis=2))

    def input_mask(self):        
        return tf.norm(self.skip.weights, ord='euclidean', axis=2) != 0
        
    def selected_count(self):
        return tf.math.reduce_sum(tf.cast(self.input_mask(), dtype=tf.int32), axis=1)

    def test_step(self, data):
        return super().test_step(data)

    def proximal_update(self):      
        skip_star, W_star = hier_prox_group(
            self.skip.weights[0],
            self.W.weights[0],
            lambda_=self.optimizer.learning_rate * self.lambda_,
            lambda_bar=0,
            M=self.M,
        )



        self.skip.weights[0].assign(skip_star)
        
        self.W.weights[0].assign(W_star)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.proximal_update()
        
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        d = {m.name: m.result() for m in self.metrics}
        d['regularization'] = self.regularization()
        d['number_of_features'] = self.selected_count()
        return d
        
        
    def lambda_start(
        self,
        M=1,
        lambda_bar=0,
        factor=2,
    ):
        """Estimate when the model will start to sparsify."""

        def is_sparse(lambda_):
        
            beta = self.skip.weights[0]
            theta = self.W.weights[0]
            
            for _ in range(200):
                new_beta, theta = hier_prox_group(
                    beta,
                    theta,
                    lambda_=lambda_,
                    lambda_bar=lambda_bar,
                    M=M,
                )
         
                if tf.math.reduce_max(tf.math.abs(beta - new_beta)) < 1e-5:
                    break
                beta = new_beta

            v=  tf.math.reduce_sum(tf.cast((tf.norm(beta, ord=1, axis=0) == 0), tf.float32)).numpy()
 
            return v != 0

        start = 1e-6
        while not is_sparse(factor * start):
            start *= factor
        return start

    def call(self, inputs):
        input_shape = inputs.shape[1:]

        x_flattened = Flatten()(inputs)
        x = self.W(x_flattened)
        x = Reshape(input_shape)(x)
        x = self.model(x)
        x = self.skip(x_flattened) + x
        return x

