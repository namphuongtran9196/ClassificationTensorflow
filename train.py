import os
import argparse

# Teminal arguments
parser = argparse.ArgumentParser(description='Convert dataset to tfrecord')
parser.add_argument('-cfg','--config',type=str,help='path to config file',default='./configs/config_coco.py')
parser.add_argument('--gpu', type=int,help='GPU ID, default is 0, -1 for using CPU', default=0)
args = parser.parse_args()

# Remove logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import tensorflow as tf

from src.models import build_resnet50_classification
from src.dataset import load_dataset, data_augmentation
from datetime import datetime

def main(args):
    # build dataset
    train_dataset_path = ''
    train_dataset = load_dataset(train_dataset_path, 128,(224,224),data_augmentation=data_augmentation, shuffle=True)
    
    val_dataset_path = ''
    val_dataset = load_dataset(val_dataset_path, 128,(224,224),classes=train_dataset.classes, shuffle=True)
    # init model
    model = build_resnet50_classification(train_dataset.num_classes(),input_shape=(224,224,3),dropout=0.2,preprocessing=True)
    
    # compile model 
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.optimizers.Adam(0.1))
    
    # define checkpoint callback
    checkpoint_dir = "./checkpoints/resnet50_classification"
    save_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, save_time,'weights')
    logs_path = os.path.join(checkpoint_dir, save_time,'logs')
    samples_path = os.path.join(checkpoint_dir, save_time,'samples')
    os.makedirs(checkpoint_path,exist_ok=True)
    os.makedirs(logs_path,exist_ok=True)
    os.makedirs(samples_path,exist_ok=True)
    
    model_cptk = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path +'/retinanet.h5',
        monitor="loss",
        save_best_only=False,
        save_weights_only=True
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
    
    callbacks_list = [model_cptk,tensorboard_callback]
    
    # Training
    model.fit(train_dataset,epochs=5,callbacks=callbacks_list,verbose=1)

if __name__ == '__main__':
    main(args)

