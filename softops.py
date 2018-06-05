import tensorflow as tf

"""
A collection of not-very-continuous functions, with hacked continuous gradients
"""

def with_grad(f, g, name='custom_gradient'):
    """
    Creates a hacked up tensor that evaluates as `f`, but has the gradient of `g`.

    https://stackoverflow.com/a/36480182/995480
    """
    with tf.name_scope(name):
        return g + tf.stop_gradient(f - g)



def floor(x, name='soft_floor'):
    """
    `tf.ceil`, with derivative=1
    """
    return with_grad(tf.floor(x), x, name = name)

def ceil(x, name='soft_ceil'):
    """
    `tf.ceil`, with derivative=1
    """
    return with_grad(tf.ceil(x), x, name = name)

def round(x, name='soft_round'):
    """
    `tf.round`, with derivative=1
    """
    return with_grad(tf.round(x), x, name = name)



def threshold(x, ym=0, yp=1, softness=1, soft_value=False, soft_grad=True, threshold_func=tf.sigmoid, name='soft_threashold'):
    """
    Generates 2 threshold functions:
    - A hard threshold where the output is `ym` when `x <= 0` and `yp` where `x > 0`
    - Soft threshold, where the value of the output behaves like `ym` for small values of `x` and `yp` for larger values of `x`.
      Around zero, both outputs are smoothly interpolated by the `threshold_func` (Tipically a sigmoid).
      The `softness` of the transition scales the interpolation function: Smaller values cause the transition region to be smaller and harder, while and larger values will make the transition region bigger and smoother.

    Tipically, the hard threshold will be used on the forward pass to calculate the value of the tensor, and the smooth threshold is used on the backwards pass to calculate it's derivative.
    """
    with tf.name_scope(name):
        if softness != 1:
            x /= softness
        soft_q = tf.identity(threshold_func(x), 'soft_q')
        hard_q = tf.where(x > 0, tf.ones(tf.shape(x)),  tf.zeros(tf.shape(x)), name='hard_q')
        soft_y = tf.add((1-soft_q) * ym, soft_q * yp, name='soft_value')
        hard_y = tf.add((1-hard_q) * ym, hard_q * yp, name='hard_value')

        return with_grad(
            soft_y if soft_value else hard_y,
            soft_y if soft_grad else hard_y)



def perc_variation(a, b):
    """
    Return the log-percentual variation of `a` w.r.t. `b`.

    The log-percentual variation calculate `a` w.r.t. `b`, but scales it so that for near-zero values, the percentual and log-percentual values are the same.
    """
    with tf.name_scope('perc_variation'):
        return 100. * tf.log(a / b)

def gt(a, b, ym=0, yp=1, percent = True, *args, **kwargs):
    """
    Soft version of `a > b`

    If `percent` is set, the soft transition is applied on the percentual-variation of `a` w.r.t. `b`, and the `softness` can be seen as a percent value, rather than absolute.
    """
    kwargs.setdefault('name', 'soft_gt')
    return threshold(
        perc_variation(a,b) if percent else a-b,
        ym=ym, yp=yp, *args, **kwargs)

def lt(a, b, ym=0, yp=1, percent = True, *args, **kwargs):
    """
    Soft version of `a < b`

    If `percent` is set, the soft transition is applied on the percentual-variation of `a` w.r.t. `b`, and the `softness` can be seen as a percent value, rather than absolute.
    """
    kwargs.setdefault('name', 'soft_lt')
    return threshold(
        -perc_variation(a, b) if percent else b-a,
        ym=ym, yp=yp, *args, **kwargs)

def lte(a, b, ym=0, yp=1, *args, **kwargs):
    """
    Soft version of `a <= b`

    If `percent` is set, the soft transition is applied on the percentual-variation of `a` w.r.t. `b`, and the `softness` can be seen as a percent value, rather than absolute.
    """
    kwargs.setdefault('name', 'soft_lte')
    return gt(a, b, ym=yp, yp=ym, *args, **kwargs)

def gte(a, b, ym=0, yp=1, *args, **kwargs):
    """
    Soft version of `a >= b`

    If `percent` is set, the soft transition is applied on the percentual-variation of `a` w.r.t. `b`, and the `softness` can be seen as a percent value, rather than absolute.
    """
    kwargs.setdefault('name', 'soft_gte')
    return lt(a, b, ym=yp, yp=ym, *args, **kwargs)

def positive(x, softness=1, soft_value=False, soft_grad=True):
    """
    A function to check if `x >= 1`, with relevant derivatives.

    This function only makes sense if `x` is a non-negative integer.

    The derivative is `1` when `x=0`, and gets increasingly smaller for larger values of `x`.
    """
    with tf.name_scope('notzero'):
        #hard_y = tf.where(x > 0, tf.ones(tf.shape(x)), tf.zeros(tf.shape(x)))
        hard_y = tf.minimum(1., x)
        soft_y = tf.tanh(x / softness)

        return with_grad(
            tf.maximum(0., soft_y if soft_value else hard_y),
            softness * soft_y if soft_grad else hard_y)
