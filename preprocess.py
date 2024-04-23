# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import glob
import splitfolders
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

scaler = MinMaxScaler()

t2 = sorted(glob.glob("/home/cap6412.student28/BraTS2020 Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii"))
t1ce = sorted(glob.glob("/home/cap6412.student28/BraTS2020 Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii"))
flair = sorted(glob.glob("/home/cap6412.student28/BraTS2020 Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii"))
mask = sorted(glob.glob("/home/cap6412.student28/BraTS2020 Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii"))

for img in range(len(t2)):
    print("Now preparing image and masks number: ", img)

    t2_img = nib.load(t2[img]).get_fdata()
    t2_img = scaler.fit_transform(t2_img.reshape(-1, t2_img.shape[-1])).reshape(t2_img.shape)

    t1ce_img = nib.load(t1ce[img]).get_fdata()
    t1ce_img = scaler.fit_transform(t1ce_img.reshape(-1, t1ce_img.shape[-1])).reshape(t1ce_img.shape)

    flair_img = nib.load(flair[img]).get_fdata()
    flair_img = scaler.fit_transform(flair_img.reshape(-1, flair_img.shape[-1])).reshape(flair_img.shape)

    mask_img = nib.load(mask[img]).get_fdata()
    mask_img = mask_img.astype(np.uint8)
    mask_img[mask_img == 4] = 3

    combined_images = np.stack([flair_img, t1ce_img, t2_img], axis=3)
    combined_images = combined_images[56:184, 56:184, 13:141]
    mask_img = mask_img[56:184, 56:184, 13:141]

    val, counts = np.unique(mask_img, return_counts=True)

    if (1 - (counts[0] / counts.sum())) > 0.01:
        print("Save")
        mask_img = to_categorical(mask_img, num_classes=4)
        np.save("/home/cap6412.student28/BraTS2020 Dataset/input_data/images/_" + str(img) + '.npy', combined_images)
        np.save("/home/cap6412.student28/BraTS2020 Dataset/input_data/masks/_" + str(img) + '.npy', mask_img)
    else:
        print("Null")

input_folder = "/home/cap6412.student28/BraTS2020 Dataset/input_data/"
output_folder = "/home/cap6412.student28/BraTS2020 Dataset/output_data/"

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)
