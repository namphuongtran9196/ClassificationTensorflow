import os
import glob
import tqdm
import logging
import skimage.io
import numpy as np
import tensorflow as tf
from src.utils import pad_square_and_resize

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def load_dataset(dataset_dir,batch_size,target_size=(224,224),classes=None, data_augmentation=None,one_hot=True,shuffle=True):
    """load dataset"""
    logging.info("Load dataset from {}".format(dataset_dir))
    dataset = Dataset(batch_size=batch_size,
                           dataset_dir=dataset_dir,
                           target_size =target_size,
                           classes = classes,
                           data_augmentation=data_augmentation,
                           one_hot=one_hot,
                           shuffle=shuffle)
    return dataset

def data_augmentation(prob=0.5):
    def augment(image):
        random_prob = tf.random.uniform([], 0, 1)
        if random_prob > prob:
            """data augmentation"""
            image = tf.keras.preprocessing.image.random_rotation(image,25,row_axis=1,col_axis=0,channel_axis=2,fill_mode='constant',cval=tf.reduce_mean(image))
            image = tf.image.random_flip_left_right(image)
            # image = tf.image.random_flip_up_down(image)
            # image = tf.image.random_rot90(image, k=1)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        return image
    return augment

class Dataset(tf.keras.utils.Sequence):
    """Dataset for loading images and annotations from the coco dataset"""
    def __init__(self, 
                 dataset_dir,
                 batch_size,
                 target_size,
                 classes = None,
                 data_augmentation=None,
                 one_hot = False,
                 shuffle= True):
        """create dataset for the given path:
           dataset should have the following structure:
           ----dataset_dir/
                    ----person/
                            ----image1.jpg
                            ----image2.jpg
                                    :
                                    :
                            ----imageN.jpg
                    ----car/
                            ----image1.jpg
                            ----image2.jpg
                                    :
                                    :
                            ----imageN.jpg
                            :
                            :
                    ----dog/
        Args:
            dataset_dir (str): path to the dataset with the above structure
            batch_size (int): the number of samples in a batch  
            target_size (tuple): the size of the image after padding and resizing. For example (224,224)
            classes (list): list of classes in the dataset. For example ['person','car','dog']
            data_augmentation (object or function, optional): An object or function that takes a numpy array of 
                                                            shape (H, W, C) and augments it.
            one_hot (bool, optional): While creating the dataset, if one_hot is True, the label will be 
                                        converted to one-hot encoding. Default is False.
            shuffle (bool, optional): If True, the samples will be shuffled after every epoch. Default is True.
        """

        self.batch_size = batch_size
        self.samples=[]
        self.classes = classes
        self.build(dataset_dir)
        self.indexes = np.arange(len(self.samples))
        self.shuffle = shuffle
        self.target_size = target_size
        self.data_augmentation = data_augmentation
        self.one_hot = one_hot
        self.on_epoch_end()
    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.floor(len(self.samples) / self.batch_size))
    
    def __getitem__(self, idx):
        """Return a batch of samples"""
        batch = self.indexes[idx * self.batch_size:(idx + 1) *self.batch_size]
        return self.make_samples(batch)
    
    def num_classes(self):
        """Return the number of classes."""
        return len(self.classes)
    
    def on_epoch_end(self):
        """Shuffle the dataset after every epoch if shuffle is True"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def build(self,dataset_dir):
        """Building dataset with the given path"""
        logging.info('Building dataset...')
        # Create classes name if not given
        if self.classes is None:
            self.classes = os.listdir(dataset_dir)
        # Map classes to integers
        self.classes_map_to_id = {self.classes[i]: i for i in range(len(self.classes))}
        # Write the classes sequence to a file
        with open('classes_info.txt', 'w') as f:
            for i, c in self.classes_map_to_id.items():
                f.write("\n{} {}".format(i, c))
        # Build image path for each class
        for cls in tqdm.tqdm(self.classes):
            images_path = glob.glob(os.path.join(dataset_dir, cls, '*'))
            for img_path in images_path:
                self.samples.append({
                        'image_path': img_path,
                        'label': cls
                    })
        # Convert samples to numpy array
        self.samples = np.asarray(self.samples)
    
    def make_samples(self, batch_idx):
        """Make a bacth sample for training"""
        # get batch images and labels
        samples = self.samples[batch_idx]
        # init return value
        batch_images = []
        batch_labels = []
        # loop through the bacth samples
        for sample in samples:
            # load the image from path
            image = self.load_image(sample['image_path'])
            # augment the image
            if self.data_augmentation:
                image = self.data_augmentation(image)
            # padding and resize image
            image = pad_square_and_resize(image, self.target_size)
            # load class labels
            if self.one_hot:
                # create one-hot encoding
                label = [name == sample['label'] for name in self.classes]
            else:
                # create integer encoding
                label = self.classes_map_to_id[sample['label']]
            # append to batch
            batch_images.append(image)
            batch_labels.append(label)
        # convert to tensor
        batch_images = tf.convert_to_tensor(batch_images,dtype=tf.float32)
        batch_labels = tf.convert_to_tensor(batch_labels,dtype=tf.float32)
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

if __name__=="__main__":
    import cv2
    img_raw = cv2.imread('../ace_0000.jpg')
    while(True):
        img = data_augmentation(0.5)(img_raw)
        img = tf.cast(img,tf.uint8)
        cv2.imshow('img',img.numpy())
        key = cv2.waitKey(0)
        if key == ord('q'):
            break