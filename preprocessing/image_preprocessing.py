
# coding: utf-8

# Image preprocessing script
# Based on code from Eben Nolson's Neural networks with Theano and Lasagne tutorial, available at:
# https://github.com/ebenolson/pydata2015/blob/master/3%20-%20Convolutional%20Networks/Finetuning%20for%20Image%20Classification.ipynb

import numpy as np

import matplotlib.pyplot as plt
import skimage.transform
import sklearn.cross_validation
import hickle as hkl
import os
import gc

# Seed for reproducibility
np.random.seed(42)

CLASSES = ['drink', 'food', 'inside', 'menu', 'outside']
LABELS = {cls: i for i, cls in enumerate(CLASSES)}

model_list = ['VGG_16', 'VGG_19', 'googlenet', 'inception_v3']


# The network expects input in a particular format and size.
# We define a preprocessing function to load a file and apply the necessary transformations
def prep_image_vgg(fn, image_mean, ext='jpg'):
    im = plt.imread(fn, ext)

    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # discard alpha channel if present
    im = im[:3]

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - image_mean
    gc.collect()
    return im[np.newaxis].astype('float32')


def prep_image_googlenet(fn, image_mean, ext='jpg'):
    im = plt.imread(fn, ext)

    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2)
    # Resize so smallest dim = 224, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (224, w * 224 / h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h * 224 / w, 224), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - image_mean
    gc.collect()
    return im[np.newaxis].astype('float32')


def prep_image_inception_v3(fn, ext='jpg'):
    im = plt.imread(fn, ext)
    im = skimage.transform.resize(im, (299, 299), preserve_range=True)

    im = (im - 128) / 128.
    im = np.rollaxis(im, 2)[np.newaxis].astype('float32')
    
    gc.collect()
    return im


def images_to_array(model):
    # Load and preprocess the entire dataset into numpy arrays
    X = []
    y = []

    print('Current model: {}'.format(model))
    if model in model_list[0:1]:
        # Model is VGG
        bgr_mean = np.asarray([103.939, 116.779, 123.68])
        image_mean = bgr_mean[:, np.newaxis, np.newaxis]
        for cls in CLASSES:
            print('Working on class: {}'.format(cls))
            for fn in os.listdir('./images/{}'.format(cls)):
                im = prep_image_vgg('./images/{}/{}'.format(cls, fn), image_mean)
                X.append(im)
                y.append(LABELS[cls])
                gc.collect()
    elif model in model_list[2]:
        # Model is googlenet
        image_mean = np.array([104, 117, 123]).reshape((3, 1, 1))
        for cls in CLASSES:
            print('Working on class: {}'.format(cls))
            for fn in os.listdir('./images/{}'.format(cls)):
                im = prep_image_googlenet('./images/{}/{}'.format(cls, fn), image_mean)
                X.append(im)
                y.append(LABELS[cls])
                gc.collect()
    else:
        # Model is inception_v3
        for cls in CLASSES:
            print('Working on class: {}'.format(cls))
            for fn in os.listdir('./images/{}'.format(cls)):
                im = prep_image_inception_v3('./images/{}/{}'.format(cls, fn))
                X.append(im)
                y.append(LABELS[cls])
                gc.collect()

    print('Concatenate array')
    X = np.concatenate(X)
    y = np.array(y).astype('int32')

    print('Split into train/test indices')
    # Split into train, validation and test sets
    train_ix, test_ix = sklearn.cross_validation.train_test_split(range(len(y)))
    train_ix, val_ix = sklearn.cross_validation.train_test_split(range(len(train_ix)))

    X_tr = X[train_ix]
    y_tr = y[train_ix]

    X_val = X[val_ix]
    y_val = y[val_ix]

    X_te = X[test_ix]
    y_te = y[test_ix]
    
    gc.collect()

    print('Start dumping data')
    hkl.dump(X_tr, model + '_X_tr.hpy', mode='w', compression='gzip')
    hkl.dump(y_tr, model + '_y_tr.hpy', mode='w', compression='gzip')
    gc.collect()
        
    hkl.dump(X_val, model + '_X_val.hpy', mode='w', compression='gzip')
    hkl.dump(y_val, model + '_y_val.hpy', mode='w', compression='gzip')
    gc.collect()
    
    hkl.dump(X_te, model + '_X_te.hpy', mode='w', compression='gzip')
    hkl.dump(y_te, model + '_y_te.hpy', mode='w', compression='gzip')
    gc.collect()

for model in model_list:
    images_to_array(model)
    gc.collect()
