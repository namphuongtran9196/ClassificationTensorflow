import tensorflow as tf

def other_loss(one_hot=True):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    def binary_loss_one_hot(y_true, y_pred):
        y_true = tf.math.reduce_max(y_true,keepdims=True, axis=-1)
        loss = bce(y_true, y_pred)
        return loss
    def binary_loss_ids(y_true,y_pred):
        pass
    return binary_loss_one_hot if one_hot else binary_loss_ids
