import os
import glob
import tqdm
import logging
import skimage.io
import numpy as np
import tensorflow as tf
from utils import pad_square_and_resize

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def load_dataset(dataset_dir,batch_size,target_size=(224,224),classes=None, data_augmentaion=None,shuffle=True):
    """load coco dataset"""
    logging.info("load dataset from {}".format(dataset_dir))
    dataset = Dataset_COCO(batch_size=batch_size,
                           dataset_dir=dataset_dir,
                           target_size =target_size,
                           classes = classes,
                           data_augmentaion=data_augmentaion,
                           shuffle=shuffle)
    return dataset


class Dataset_COCO(tf.keras.utils.Sequence):
    """Dataset for loading images and annotations from the coco dataset"""
    def __init__(self, 
                 batch_size,
                 dataset_dir,
                 target_size,
                 classes,
                 data_augmentation=None,
                 shuffle= True):

        self.batch_size = batch_size
        self.samples=[]
        self.classes = classes
        self.build(dataset_dir)
        self.indexes = np.arange(len(self.samples))
        self.shuffle = shuffle
        self.target_size = target_size
        self.data_augmentation = data_augmentation
        
    def __len__(self):
        return int(np.floor(len(self.samples) / self.batch_size))
    
    def __getitem__(self, idx):
        batch = self.indexes[idx * self.batch_size:(idx + 1) *self.batch_size]
        return self.make_samples(batch)
    
    def num_classes(self):
        """Return the number of classes."""
        return len(self.classes)
    
    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def build(self,dataset_dir):
        """
        Preprocess the coco dataset, remove class, filter images, and generate class mapping
        """
        logging.info('Building dataset...')
        if self.classes is None:
            self.classes = os.listdir(dataset_dir)
        self.classes_map_to_id = {self.classes[i]: i for i in range(len(self.classes))}
        with open('classes_info.txt', 'w') as f:
            for i, c in self.classes_map_to_id.items():
                f.write("\n{} {}".format(i, c))
        for cls in tqdm.tqdm(self.classes):
            images_path = glob.glob(os.path.join(dataset_dir, 'images', '*'))
            for img_path in images_path:
                self.samples.append({
                        'image_path': img_path,
                        'label': cls
                    })
        self.samples = np.asarray(self.samples)
    
    def make_samples(self, batch_idx):
        """
        Make a sample for training
        :param idx: index of batch
        :return: a batch training dataset
        """
        samples = self.samples[batch_idx]
        batch_images = []
        batch_labels = []
        for sample in samples:
            # load the image from path
            image = self.load_image(sample['image_path'])
            # augment the image
            if self.data_augmentation:
                image = self.data_augmentation(image)
            # padding and resize image
            image = pad_square_and_resize(image, self.target_size)
            # load class name
            class_ids = self.classes_map_to_id(sample['label'])
            batch_images.append(image)
            batch_labels.append(class_ids)
        batch_images = tf.convert_to_tensor(batch_images,dtype=tf.float32)
        batch_labels = tf.convert_to_tensor(batch_labels,dtype=tf.int32)
        return batch_images, batch_labels
    
    def load_image(self, path):
        """Load the specified image and return a [H,W,3] Numpy array."""
        # Load image
        image = skimage.io.imread(path)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
def data_augmentation(image):
    """data augmentation"""
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    # image = tf.image.random_rot90(image, k=1)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image