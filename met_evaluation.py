# -*- coding: utf-8 -*-

from tensorflow.keras.models import load_model
import numpy as np
import os
from custom_data import imageLoader
from sklearn.model_selection import KFold
from monai.metrics import DiceMetric
import torch
import sys
import matplotlib.pyplot as plt

print("start")
sys.stdout.flush()

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

mymodel_1 = load_model("/home/cap6412.student28/0_brats.hdf5", compile=False)
mymodel_2 = load_model("/home/cap6412.student28/1_brats.hdf5", compile=False)
mymodel_3 = load_model("/home/cap6412.student28/2_brats.hdf5", compile=False)
mymodel_4 = load_model("/home/cap6412.student28/3_brats.hdf5", compile=False)
mymodel_5 = load_model("/home/cap6412.student28/4_brats.hdf5", compile=False)

img_num = 110

img = np.load("/home/cap6412.student28/BraTS2020 Dataset/output_data/val/images/_" + str(img_num) + ".npy")
mask = np.load("/home/cap6412.student28/BraTS2020 Dataset/output_data/val/masks/_" + str(img_num) + ".npy")
mask_argmax = np.argmax(mask, axis=3)

img_input = np.expand_dims(img, axis=0)
prediction = mymodel_1.predict(img_input)
prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]

n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(img[:, :, n_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(mask_argmax[:, :, n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction_argmax[:, :, n_slice])
plt.savefig("image_test.png")

dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

print("after model load")
sys.stdout.flush()

batch_size = 4

kf = KFold(n_splits=5)
for (train, test), model in zip(kf.split(all_img_list), [mymodel_1, mymodel_2, mymodel_3, mymodel_4, mymodel_5]):
    print("first loop")
    sys.stdout.flush()

    steps_per_epoch = len(test) // batch_size
    i = 0

    test_img_datagen = imageLoader(val_img_dir, all_img_list, val_mask_dir, all_mask_list, batch_size, test, test)

    for test_image_batch, test_mask_batch in test_img_datagen:

        test_mask_batch = test_mask_batch.transpose(0, 4, 2, 3, 1)
        test_mask_batch = torch.round(torch.tensor(test_mask_batch))

        test_pred_batch = model.predict(test_image_batch)
        test_pred_batch = test_pred_batch.transpose(0, 4, 2, 3, 1)
        test_pred_batch = torch.round(torch.tensor(test_pred_batch))

        dice_metric_batch(y_pred=test_pred_batch, y=test_mask_batch)

        i += 1
        print(str(i) + '/' + str(steps_per_epoch))
        sys.stdout.flush()
        if i == steps_per_epoch:
            print("break")
            break

    metric_batch = dice_metric_batch.aggregate()
    metric_tc = metric_batch[1].item()
    metric_wt = metric_batch[2].item()
    metric_et = metric_batch[3].item()
    dice_metric_batch.reset()
    print("Dice score")
    sys.stdout.flush()
    print(metric_tc, metric_wt, metric_et)
