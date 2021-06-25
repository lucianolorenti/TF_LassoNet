import tensorflow as tf
from tf_lassonet.proximal import hier_prox_group

class TestProximal:
    def test_group_lasso(self):
        n_inputs = 256
        n_hidden = 100
        n_outputs = 10
        M = 25
        lambda_ = 50
        W = tf.random.normal((n_inputs, n_hidden))*1500
        skip = tf.random.normal((n_inputs, n_outputs))* 15
        
        v = tf.random.normal((n_inputs, n_outputs)) * 15

        U = tf.random.normal((n_inputs, n_hidden)) * 15

        loss1 = tf.norm(v-skip)**2 + tf.norm(U-W)**2 + lambda_*tf.norm(skip)

        skip_star, W_star = hier_prox_group(skip, W, lambda_=50, M=M)

        loss2 = tf.norm(skip_star-skip)**2 + tf.norm(W_star-W)**2 + lambda_*tf.norm(skip_star)

        assert tf.reduce_max(W[0,:]) >=  tf.norm(skip[0,:])*M
        assert tf.reduce_max(W_star[0,:]) <=  tf.norm(skip_star[0,:])*M

        assert tf.reduce_max(W[5,:]) >=  tf.norm(skip[5,:])*M
        assert tf.reduce_max(W_star[5,:]) <=  tf.norm(skip_star[5,:])*M
        assert loss2 < loss1

