import tensorflow as tf
from tf_lassonet.proximal import hier_prox_group

class TestProximal:
    def group_lasso(self):
        n_inputs = 256
        n_hidden = 100
        n_outputs = 10
        M = 25
        lambda_ = 50
        W = tf.random.normal((n_inputs, n_hidden))*500
        skip = tf.random.normal((n_inputs, n_outputs))* 15
        
        v = tf.random.normal((n_inputs, n_outputs)) * 15

        U = tf.random.normal((n_inputs, n_hidden)) * 15

        loss1 = tf.norm(v-skip)**2 + tf.norm(U-W)**2 + lambda_*tf.norm(skip)

        skip_star, W_star = hier_prox_group(skip, W, lambda_=50, M=M)

        loss2 = tf.norm(skip_star-skip)**2 + tf.norm(W_star-W)**2 + lambda_*tf.norm(skip_star)

        assert loss2 < loss1

