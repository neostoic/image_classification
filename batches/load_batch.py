import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps
from multiprocessing import freeze_support

from batchgen import BatchGen

BATCH_SIZE = 8
MODELS = ['VGG_16', 'VGG_19', 'googlenet', 'inception_v3']


def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _load_batch(pixels, input_pkl, img_path, batch_size=32, seed=None, n=5):
    data = pd.read_pickle(input_pkl)

    sample = data.sample(n=batch_size, random_state=seed)

    iter_images = iter(sample['photo_id'])
    first_image = next(iter_images)
    im = Image.open(img_path + first_image + '.jpg', 'r')
    im = ImageOps.fit(im, (pixels, pixels), Image.ANTIALIAS)
    im = (np.array(im))
    r = im[:, :, 0].flatten()
    g = im[:, :, 1].flatten()
    b = im[:, :, 2].flatten()

    img_list = np.array(list(r) + list(g) + list(b), dtype='uint8')
    img_list = img_list[np.newaxis, :]

    for img_name in iter_images:
        im = Image.open(img_path + img_name + '.jpg', 'r')
        im = ImageOps.fit(im, (pixels, pixels), Image.ANTIALIAS)
        im = (np.array(im))
        r = im[:, :, 0].flatten()
        g = im[:, :, 1].flatten()
        b = im[:, :, 2].flatten()

        img = np.array(list(r) + list(g) + list(b), dtype='uint8')
        img_list = np.vstack((img_list, img[np.newaxis, :]))

    label_list = sample['label']
    return img_list, label_list


def _preprocess_batch(data, labels, pixels, dtype='float64', model='VGG_16'):
    """
    load a batch of images
    """
    if model in MODELS[0:2]:
        data = _preprocess_vgg(data.astype(dtype), pixels)
    elif model in MODELS[2]:
        data = _preprocess_googlenet(data.astype(dtype), pixels)
    elif model in MODELS[3]:
        data = _preprocess_inception_v3(data.astype(dtype))
    else:
        raise ValueError
    return data, labels.astype('int32')


def batch(seed, input_pkl, img_path, dtype='float32', batch_size=32, grayscale=False, pixels=64, model='VGG_16'):
    data, labels = _load_batch(pixels, input_pkl, img_path, batch_size=batch_size, seed=seed, n=5)

    x, y = _preprocess_batch(data, labels, pixels, dtype=dtype, model=model)

    x = x.reshape((x.shape[0], 3, pixels, pixels))
    if grayscale:
        x = _grayscale(x, pixels)

    return x, y


def _grayscale(a, pixels):
    return a.reshape(a.shape[0], 3, pixels, pixels).mean(1).reshape(a.shape[0], -1)


def _preprocess_vgg(batch, pixels):
    rgb_mean = [123.68, 116.779, 103.939]
    img_size = pixels * pixels
    batch[:, 0:img_size] -= rgb_mean[0]
    batch[:, img_size:(2 * img_size)] -= rgb_mean[1]
    batch[:, (2 * img_size):(3 * img_size)] -= rgb_mean[2]
    return batch


def _preprocess_googlenet(batch, pixels):
    rgb_mean = np.asarray([123, 117, 104])
    img_size = pixels * pixels
    batch[:, 0:img_size] -= rgb_mean[0]
    batch[:, img_size:(2 * img_size)] -= rgb_mean[1]
    batch[:, (2 * img_size):(3 * img_size)] -= rgb_mean[2]
    return batch


def _preprocess_inception_v3(batch):
    data = (batch - 128) / 128.
    return data


def _preprocess_original(batch):
    return batch / 255.0  # scale between [0, 1]
