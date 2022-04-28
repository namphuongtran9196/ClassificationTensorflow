import os
import argparse
import logging
# Teminal arguments
parser = argparse.ArgumentParser(description='Convert dataset to tfrecord')
parser.add_argument('-cfg','--config',type=str,help='path to config file',default='./configs/config.py')
parser.add_argument('--gpu', type=int,help='GPU ID, default is 0, -1 for using CPU', default=0)
args = parser.parse_args()

# Remove logging tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# set logging config
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

import tensorflow as tf
import importlib.util

from src.models import build_classification_model
from src.dataset import load_dataset, data_augmentation
from src.utils import save_class_info,load_class_info
from datetime import datetime

def main(args):
    
    # load config
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    # load class info
    classes= None
    classes_map_to_id = None
    if config.class_info_path is not None:
        classes_info = load_class_info(config.class_info_path)
        classes = classes_info["classes"]
        classes_map_to_id = classes_info["classes_map_to_id"]
    # build dataset
    train_dataset = load_dataset(config.train_dataset, batch_size=config.batch_size,target_size=config.input_shape[:2],
                                 classes=classes,classes_map_to_id=classes_map_to_id,
                                 data_augmentation=data_augmentation(0.5),one_hot=True, shuffle=True)
    val_dataset = load_dataset(config.val_dataset, batch_size=config.batch_size,target_size=config.input_shape[:2],
                               classes=train_dataset.classes,classes_map_to_id=train_dataset.classes_map_to_id
                               ,one_hot=True, shuffle=True)
    # define checkpoint save time and path
    checkpoint_dir = "./checkpoints/{}".format(config.backbone)
    save_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # create checkpoint dir and logs path
    checkpoint_path = os.path.join(checkpoint_dir, save_time,'weights')
    logs_path = os.path.join(checkpoint_dir, save_time,'logs')
    os.makedirs(checkpoint_path,exist_ok=True)
    os.makedirs(logs_path,exist_ok=True)
    # model save checkpoint
    model_cptk_loss = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path +'/{}_loss.h5'.format(config.backbone),
        monitor="val_loss",
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    model_cptk_acc = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path +'/{}_acc.h5'.format(config.backbone),
        monitor="val_categorical_accuracy",
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    # model logs checkpoint
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
    # callbacks for training model
    callbacks_list = [model_cptk_loss,model_cptk_acc,tensorboard_callback]
    # saving class info for inference phase
    save_class_info(train_dataset.classes, train_dataset.classes_map_to_id, 
                    save_path=os.path.join(checkpoint_path,"class_info.json"))
    # loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    # metric 
    metrics = []
    metrics.append(tf.keras.metrics.CategoricalAccuracy())
    metrics.append(tf.keras.metrics.Precision())
    metrics.append(tf.keras.metrics.Recall())
    metrics.append(tf.keras.metrics.AUC())
    
    # init model
    model = build_classification_model(train_dataset.num_classes(),input_shape=config.input_shape,
                                       backbone=config.backbone,dropout=config.dropout,
                                       preprocessing=config.preprocessing)
    if config.old_checkpoints_path is not None:
        model.load_weights(config.old_checkpoints_path)
        logging.info("Restore model from {}".format(config.old_checkpoints_path))
    else:
        logging.info("Training from scratch")
    # compile model 
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)   
    # Transfer learning with high lerning rate
    logging.info("Transfer learning with high lerning rate for first {} epochs".format(int(config.epochs/4)))
    model.fit(train_dataset, epochs=int(config.epochs/4), validation_data=val_dataset, callbacks=callbacks_list)
    # Transfer learning with low lerning rate
    logging.info("Unfreeze layers and train for the rest of the epochs")
    model.trainable = True # Unfreeze all layers
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)   
    model.fit(train_dataset, epochs=int(config.epochs*3/4), validation_data=val_dataset, callbacks=callbacks_list)
    model.save(checkpoint_path +'/{}_all_epochs.h5'.format(config.backbone))

if __name__ == '__main__':
    main(args)

