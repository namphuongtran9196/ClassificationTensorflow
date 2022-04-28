import json
import logging
import tensorflow as tf
# set logging config
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def pad_square_and_resize(image, target_size):
    """
    Pad image to square and resize to target size.
    1. Pad with zeros on right and bottom to make image shape square
    2. Resize image to the target size
    3. Return image
    
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
    image = tf.image.resize(image, target_size,method=tf.image.ResizeMethod.AREA)
    return image

def load_class_info(class_info_path):
    """
    Load class info from json file
    
    Arguments:
    class_info_path: Path to json file
    
    returns:
    class_info: A dictionary of class info
    """
    with open(class_info_path, 'r') as f:
        class_info = json.load(f)
    return class_info

def save_class_info(classes, classes_map_to_id, save_path="class_info.json"):
    # Write the classes sequence to a file
    json_classes = {}
    json_classes.update(classes_map_to_id)
    json_classes.update({"classes": classes})
    json_object = json.dumps(json_classes, indent = 4) # Serializing json 
    with open(save_path, 'w') as f:
        f.write(json_object)
    logging.info("Class info saved to {}".format(save_path))