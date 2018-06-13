import tensorflow as tf

eps = 1e-6
class perc:
    """
    Calculates the percentage difference of a value with respect to a reference.

    E.g., 2 wrt 1 is a 100% increase.
    """
    @staticmethod
    def normalize(value, wrt=1.0, name='perc.normalize'):
        with tf.name_scope(name):
            return tf.multiply(100., value/tf.maximum(wrt, eps) - 1., name="value")

    @staticmethod
    def denormalize(normalized_value, wrt=1.0, name='perc.denormalize'):
        with tf.name_scope(name):
            return tf.multiply(wrt, normalized_value / 100. + 1., name="value")

class log_perc:
    """
    Calculates the log-percentage difference of a value with respect to a reference.

    This metric is aproximately equal to the percentage difference for small numbers,
    and can be easily understood by humans.

    At the same time, it is actually a value in log scale, making composing operations completely linear.

    To ilustrate it's usefulness, think using normal percent variations, the "inverse" of a +100% variation is -50%.
    On log-perc, doubling the value is +0.69, and taking half is -0.69
    """
    @staticmethod
    def normalize(value, wrt=1.0, name='log_perc.normalize'):
        with tf.name_scope(name):
            return tf.multiply(100., tf.log(tf.maximum(value, eps)/tf.maximum(wrt, eps)), name="value")

    @staticmethod
    def denormalize(normalized_value, wrt=1.0, name='log_perc.denormalize'):
        with tf.name_scope(name):
            return tf.multiply(wrt, tf.exp(normalized_value / 100.), name="value")

