import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import KFold
from skimage.measure import label, regionprops
from scipy.spatial.distance import directed_hausdorff
from data import BrainTumorDataGenerator
from unet import unet_model



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU devices found.")
    
    
def dice_coeff(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def hausdorff_distance(y_true, y_pred):
 
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    hausdorff_dist = directed_hausdorff(y_true_np, y_pred_np)[0]
    return hausdorff_dist


def train_model(model, train_gen, val_gen, optimizer, loss_fn, num_epochs=10):
    for epoch in range(num_epochs):
        # Training loop
        for images, labels in train_gen:
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)  # Training mode
                loss = loss_fn(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Validation loop
        dice_scores = []
        hausdorff_distances = []
        for images, labels in val_gen:
            predictions = model(images, training=False) 
            preds = predictions > 0.5  

           
            dice_scores.append(dice_coeff(labels, preds))
            hausdorff_distances.append(hausdorff_distance(labels, preds))

        # Print average metrics for the epoch
        avg_dice = np.mean(dice_scores)
        avg_hausdorff = np.mean(hausdorff_distances)
        print(f"Epoch {epoch+1}/{num_epochs}, Dice Score: {avg_dice}, Hausdorff Distance: {avg_hausdorff}")


image_dir='C:/SURANADI/UCF/Med_Image/Programming_Assignment_2/Task01_BrainTumour/Task01_BrainTumour/imagesTr/'
label_dir='C:/SURANADI/UCF/Med_Image/Programming_Assignment_2/Task01_BrainTumour/Task01_BrainTumour/labelsTr/'
image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
dataset = BrainTumorDataGenerator(image_dir, label_dir, image_filenames, batch_size=4, shuffle=True)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
model = unet_model(128, 128, 128, 1, 4) 
optimizer = Adam(learning_rate=0.001)
loss_fn = BinaryCrossentropy()

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f"FOLD {fold}")
    train_gen = [dataset[i] for i in train_ids]
    val_gen = [dataset[i] for i in val_ids]

    train_model(model, train_gen, val_gen, optimizer, loss_fn)
