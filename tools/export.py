import os
import argparse
import logging
# Teminal arguments
parser = argparse.ArgumentParser(description='Convert dataset to tfrecord')
parser.add_argument('-m','--model',type=str,help='path to model.h5',default='./model.h5')
parser.add_argument('-o','--output',type=str,help='path to output file',default='./model.tflite')
parser.add_argument('-hp','--high_probs', action="store_true",help='which to use high probs for prediction')
parser.add_argument('--gpu', type=int,help='GPU ID, default is 0, -1 for using CPU', default=0)
args = parser.parse_args()
# Remove logging tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# set logging config
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

def export_model_other(model, threshold=0.8, up_probs_pram=2.0,name='Classification_model'):
    # Cheating to get model with high probability
    branch_other_probs = model.get_layer('other_probs_layer').output
    branch_classes = model.get_layer('prediction_logits_layer').output
    
    x = tf.greater_equal(branch_other_probs, threshold)
    x = tf.cast(x,tf.float32)
    x = tf.multiply(x,tf.constant(up_probs_pram))
    x = tf.multiply(x,branch_classes)
    
    output = layers.Activation("softmax",name='output')(x)
    return Model(model.input, output, name=name)

def keras_to_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(output_path, "wb").write(tflite_model)
    logging.info("Saved model to: {}".format(output_path))

def test_inference_tflite(model_tflite):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_tflite)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    interpreter.get_tensor(output_details[0]['index'])
    logging.info("Successfully tested TFLite model with input shape: {}".format(input_shape))

def main(args):
    # load model
    model = tf.keras.models.load_model(args.model)
    if args.hp:
        model = export_model_other(model)
    keras_to_tflite(model, args.output)
    test_inference_tflite(args.output)
    

if __name__ == '__main__':
    main(args)