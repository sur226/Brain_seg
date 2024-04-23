# -*- coding: utf-8 -*-

import numpy as np

def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):
        if image_name.split('.')[1] == 'npy':
            image = np.load(img_dir + image_name)
            images.append(image)
    images = np.array(images)
    return images

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size, x, y):
    L = len(x)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X1 = x[batch_start:limit]
            Y1 = y[batch_start:limit]
            X = load_img(img_dir, [img_list[y] for y in X1])
            Y = load_img(mask_dir, [mask_list[y] for y in Y1])
            yield (X, Y)
            batch_start += batch_size
            batch_end += batch_size
