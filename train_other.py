import os
import argparse
import logging
# Teminal arguments
parser = argparse.ArgumentParser(description='Convert dataset to tfrecord')
parser.add_argument('-cfg','--config',type=str,help='path to config file',default='./configs/config_coco.py')
parser.add_argument('--gpu', type=int,help='GPU ID, default is 0, -1 for using CPU', default=0)
args = parser.parse_args()

# Remove logging tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import tensorflow as tf
import importlib.util

from src.models import build_classification_model
from src.dataset import load_dataset, data_augmentation
from datetime import datetime

def main(args):
    # set logging config
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    # load config
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    # build dataset
    train_dataset = load_dataset(config.train_dataset, batch_size=128,target_size=config.input_shape[:2],
                                 data_augmentation=data_augmentation(0.5), shuffle=True)
    val_dataset = load_dataset(config.val_dataset, batch_size=128,target_size=config.input_shape[:2],
                               classes=train_dataset.classes, shuffle=True)
    # define checkpoint save time and path
    checkpoint_dir = "./checkpoints/{}".format(config.backbone)
    save_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # create checkpoint dir and logs path
    checkpoint_path = os.path.join(checkpoint_dir, save_time,'weights')
    logs_path = os.path.join(checkpoint_dir, save_time,'logs')
    os.makedirs(checkpoint_path,exist_ok=True)
    os.makedirs(logs_path,exist_ok=True)
    # model save checkpoint
    model_cptk = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path +'/{}.h5'.format(config.backbone),
        monitor="loss",
        save_best_only=True,
        save_weights_only=False
    )
    # model logs checkpoint
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
    # callbacks for training model
    callbacks_list = [model_cptk,tensorboard_callback]
    # init model
    model = build_classification_model(train_dataset.num_classes(),input_shape=(224,224,3),
                                       backbone=config.backbone,dropout=0.2,preprocessing=True)
    if config.old_checkpoints_path is not None:
        model.load_weights(config.old_checkpoints_path)
        logging.info("Restore model from {}".format(config.old_checkpoints_path))
    else:
        logging.info("Training from scratch")
    # compile model 
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  optimizer=tf.optimizers.Adam(config.learning_rate))   
    # Transfer learning with high lerning rate
    model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks_list)
    # Transfer learning with low lerning rate
    model.trainable = True
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  optimizer=tf.optimizers.Adam(1e-5))
    model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks_list)

if __name__ == '__main__':
    main(args)

