import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage import zoom
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Dropout
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from medpy.metric import binary as medpy_metrics
from unet import unet_model

# Load and preprocess the data
def load_nifti_file(filepath):
    scan = nib.load(filepath)
    return scan.get_fdata()

def preprocess_volume(volume, target_shape=(128, 128, 128)):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    volume = resize(volume, target_shape)
    return volume

def slice_volume(volume):
    
    slices = np.moveaxis(volume, -1, 0)  
    slices = np.expand_dims(slices, axis=-1) 
    return slices

def preprocess_data(image_path, label_path, resize_factor=None):
    image = preprocess_volume(load_nifti_file(image_path))
    label = preprocess_volume(load_nifti_file(label_path))
    if resize_factor:
        image = zoom(image, resize_factor)
        label = zoom(label, resize_factor)
    return slice_volume(image), slice_volume(label)

class BrainTumorDataGenerator(Sequence):
    def __init__(self, image_dir, label_dir, image_filenames, batch_size=4, resize_factor=None, shuffle=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = image_filenames
        self.batch_size = batch_size
        self.resize_factor = resize_factor
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_filenames = [self.image_filenames[k] for k in indexes]
        images, labels = self.__data_generation(batch_filenames)
        return np.array(images), np.array(labels)

    def __data_generation(self, batch_filenames):
        images = []
        labels = []
        for filename in batch_filenames:
            image_path = os.path.join(self.image_dir, filename)
            label_path = os.path.join(self.label_dir, filename.replace('_BRATS_001', '_BRATS_166'))
            image_slices, label_slices = preprocess_data(image_path, label_path, self.resize_factor)
            images.extend(image_slices)
            labels.extend(label_slices)
        images = np.array(images)  
        labels = np.array(labels)
        return images, labels



