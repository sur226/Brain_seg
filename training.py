# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
import pandas as pd
import os
from custom_data import imageLoader
from tensorflow import keras
import glob
import random
from Unet3d import unet_3D_model
from sklearn.model_selection import KFold
import segmentation_models_3D as sm

train_img_dir = "/home/cap6412.student28/BraTS2020 Dataset/output_data/"
train_mask_dir = "/home/cap6412.student28/BraTS2020 Dataset/output_data/"
val_img_dir = "/home/cap6412.student28/BraTS2020 Dataset/output_data/"
val_mask_dir = "/home/cap6412.student28/BraTS2020 Dataset/output_data/"

train_img_list = ['train/images/' + f for f in os.listdir(train_img_dir + 'train/images/')]
train_mask_list = ['train/masks/' + f for f in os.listdir(train_img_dir + 'train/masks/')]
val_img_list = ['val/images/' + f for f in os.listdir(train_img_dir + 'val/images/')]
val_mask_list = ['val/masks/' + f for f in os.listdir(train_img_dir + 'val/masks/')]

all_img_list = train_img_list + val_img_list
all_mask_list = train_mask_list + val_mask_list

batch_size = 2
i = 0

kf = KFold(n_splits=5)
for train, test in kf.split(all_img_list):
    train_img_datagen = imageLoader(train_img_dir, all_img_list, train_mask_dir, all_mask_list, batch_size, train, train)
    val_img_datagen = imageLoader(val_img_dir, all_img_list, val_mask_dir, all_mask_list, batch_size, test, test)

    wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

    LR = 0.0001
    optim = keras.optimizers.Adam(LR)

    steps_per_epoch = len(train_img_list) // batch_size
    val_steps_per_epoch = len(val_img_list) // batch_size

    model = unet_3D_model(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=3, num_classes=4)
    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
    print(model.summary())

    print(model.input_shape)
    print(model.output_shape)

    history = model.fit(train_img_datagen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=100,
                        verbose=1,
                        validation_data=val_img_datagen,
                        validation_steps=val_steps_per_epoch)
    model.save(str(i) + '_brats.hdf5')
    i += 1

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss.png")
