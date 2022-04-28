import os
import argparse
import logging
# Teminal arguments
parser = argparse.ArgumentParser(description='Convert dataset to tfrecord')
parser.add_argument('-cfg','--config',type=str,help='path to config file',default='./configs/config_other.py')
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

from src.models import add_other_class
from src.losses import other_loss
from src.dataset import load_dataset, data_augmentation
from src.utils import load_class_info
from datetime import datetime

def main(args):
    
    # load config
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    # load pretrained model
    pretrained_model = tf.keras.models.load_model(config.model_path)
    pretrained_model.trainable=False
    # load class info
    classes= None
    classes_map_to_id = None
    if config.class_info_path is not None:
        classes_info = load_class_info(config.class_info_path)
        classes = classes_info["classes"]
        classes_map_to_id = classes_info["classes_map_to_id"]
    # add other path when building dataset
    classes.append("other")
    # build dataset
    train_dataset = load_dataset(config.train_dataset, batch_size=config.batch_size,target_size=config.input_shape[:2],
                                 classes=classes,classes_map_to_id=classes_map_to_id,
                                 data_augmentation=data_augmentation(0.5),one_hot=True, shuffle=True)
    # remove other in classes id
    train_dataset.classes.remove("other")
    val_dataset = load_dataset(config.val_dataset, batch_size=config.batch_size,target_size=config.input_shape[:2],
                               classes=train_dataset.classes,classes_map_to_id=train_dataset.classes_map_to_id
                               ,one_hot=True, shuffle=True)
    # define checkpoint save time and path
    checkpoint_dir = "./checkpoints/{}".format(pretrained_model.name)
    save_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # create checkpoint dir and logs path
    checkpoint_path = os.path.join(checkpoint_dir, save_time,'weights')
    logs_path = os.path.join(checkpoint_dir, save_time,'logs')
    os.makedirs(checkpoint_path,exist_ok=True)
    os.makedirs(logs_path,exist_ok=True)
    # model save checkpoint
    model_cptk = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path +'/{}_other.h5'.format(pretrained_model.name),
        monitor="loss",
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    # model logs checkpoint
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
    # callbacks for training model
    callbacks_list = [model_cptk,tensorboard_callback]
    # loss function
    loss_classes_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    loss_other_fn = other_loss()
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    # metric 
    metrics = {"classes_probs_layer":[tf.keras.metrics.CategoricalAccuracy(),
                                        tf.keras.metrics.Precision(),
                                        tf.keras.metrics.Recall(),
                                        tf.keras.metrics.AUC()]
               }
    # init model
    model = add_other_class(pretrained_model)
    # compile model 
    model.compile(loss={"classes_probs_layer":loss_classes_fn,
                        "other_probs_layer":loss_other_fn}, optimizer=optimizer, metrics=metrics)   
    # Transfer learning with high lerning rate
    logging.info("Fine-tuning with other classes".format(int(config.epochs)))
    model.fit(train_dataset, epochs=int(config.epochs), validation_data=val_dataset, callbacks=callbacks_list)

if __name__ == '__main__':
    main(args)

