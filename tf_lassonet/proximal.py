import tensorflow as tf

def tf_soft_threshold(l, x):
    return tf.sign(x) * tf.nn.relu(tf.abs(x) - l)


def tf_sign_binary(x):
    ones = tf.ones_like(x)
    return tf.where(x >= 0, ones, -ones)

def hier_prox_group(skip, W, *, lambda_, M, lambda_bar=0):
    n_features, output_size = W.shape
    zeros = tf.zeros((n_features,1 ))
    W_abs_sorted = tf.sort(tf.abs(W), axis=1)[:, ::-1]
    cumulative_sum =  tf.concat([zeros, tf.cumsum(W_abs_sorted - lambda_bar, axis=1)], axis=1)



    m = tf.range(0.0, output_size + 1.0, dtype=tf.float32)
    q = 1/ (1 +m*M**2)

    norm_skip = tf.expand_dims(tf.norm(skip, axis=1, ord=2), axis=1)

    #w = q*tf_soft_threshold(lambda_, norm_skip + M*cumulative_sum)

    t = q* tf.nn.relu(1 - (lambda_ - M*cumulative_sum)/norm_skip )
    w = M*t*norm_skip
    x = t

    lower = tf.concat([tf_soft_threshold(lambda_bar, W_abs_sorted), zeros], axis=1)
    idx = tf.reduce_sum(tf.cast(lower > w, dtype=tf.int32), axis=1)

    idx = tf.stack([tf.range(0, w.shape[0]), idx, ], axis=1)


    w =  tf.expand_dims(tf.gather_nd(w, idx), axis=1)
    x =  tf.expand_dims(tf.gather_nd(x, idx), axis=1)

    #skip_star = (1/M)*(tf.sign(skip)*tf.broadcast_to(tf.expand_dims(w,axis=1), skip.shape))
 
    skip_star = skip*x
    W_star = tf.sign(W)*tf.math.minimum(w, tf.abs(W))

    return skip_star, W_star