import tensorflow as tf

def pad_square_and_resize(image, target_size):
    """
    Pad image to square and resize to target size.
    1. Pad with zeros on right and bottom to make image shape square
    2. Resize image to the target size
    3. Return image, and ratio
    
    Arguments:
    image: A 3-D tensor of shape `(height, width, channels)` representing an
    target_size: The target size of the image
    
    returns:
    image: Padded and resized image
    """
    h_f = tf.cast(tf.shape(image)[0], tf.int32)
    w_f = tf.cast(tf.shape(image)[1], tf.int32)
    largest_side = w_f if w_f > h_f else h_f
    image = tf.image.pad_to_bounding_box(image, 0, 0, tf.cast(largest_side,tf.int32), tf.cast(largest_side,tf.int32))
    image = tf.image.resize(image, [target_size, target_size],method=tf.image.ResizeMethod.AREA)
    return image