# Helper script to load batches of data for image classification using the BatchGen class
# Adapted from the Theano tutorial from Sander Dieleman,
# available at: https://github.com/benanne/theano-tutorial/blob/master/load.py
# See LICENSE file for copyright details.
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps

# Define the batch size and the models available
BATCH_SIZE = 8
MODELS = ['VGG_16', 'VGG_19', 'googlenet', 'inception_v3']

def one_hot(x, n):
    """
    convert index representation to one-hot representation
    :param x: Numpy vector
    :param n: Number of classes
    :return: One-hot representation of x
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _load_batch(pixels, input_pkl, img_path, batch_size=32, seed=None, n=5):
    """
    Helper function to load images and format it for classification
    :param pixels: size of images (one side)
    :param input_pkl: path to pickle
    :param img_path: path to the images
    :param batch_size: batch size
    :param seed: random seed for repeatability
    :param n: number of classes
    :return: list with images and list with labels
    """
    data = pd.read_pickle(input_pkl)
    # Sample the data randomly
    sample = data.sample(n=batch_size, random_state=seed)

    # Iterate the samples based on the photo_id
    iter_images = iter(sample['photo_id'])

    # Initialize arrays
    first_image = next(iter_images)
    # Open image and fit to size
    im = Image.open(img_path + first_image + '.jpg', 'r')
    im = ImageOps.fit(im, (pixels, pixels), Image.ANTIALIAS)
    # Convert image to numpy array and extract RGB channels
    im = (np.array(im))
    r = im[:, :, 0].flatten()
    1
    g = im[:, :, 1].flatten()
    b = im[:, :, 2].flatten()

    # Create image list
    img_list = np.array(list(r) + list(g) + list(b), dtype='uint8')
    img_list = img_list[np.newaxis, :]

    # Do the same for all the images in the iterator
    for img_name in iter_images:
        im = Image.open(img_path + img_name + '.jpg', 'r')
        im = ImageOps.fit(im, (pixels, pixels), Image.ANTIALIAS)
        im = (np.array(im))
        r = im[:, :, 0].flatten()
        g = im[:, :, 1].flatten()
        b = im[:, :, 2].flatten()

        img = np.array(list(r) + list(g) + list(b), dtype='uint8')
        # Stack the image vectors vertically
        img_list = np.vstack((img_list, img[np.newaxis, :]))

    # Labels are already the 'label' column of the sample data
    label_list = sample['label']
    return img_list, label_list


def _preprocess_batch(data, labels, pixels, dtype='float64', model='VGG_16'):
    """
    Preprocess the batch of images

    :param data: image dataset in RGB
    :param labels: label dataset
    :param pixels: size of a side of the image
    :param dtype: data type for the image vectors
    :param model: what model is being used
    :return: preprocessed image and label data.

    """
    # Check if model is VGG, googlenet or inception_v3, and preprocess accordingly
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
    """
    Main function called by the batch generator
    :param seed: random seed
    :param input_pkl: the input pickled file
    :param img_path: path where images are stored
    :param dtype: data type of image vector
    :param batch_size: number of images per batch
    :param grayscale: if True, grayscale images
    :param pixels: size of image
    :param model: model selected
    :return:
    """
    # Load the data from memory
    data, labels = _load_batch(pixels, input_pkl, img_path, batch_size=batch_size, seed=seed, n=5)

    # Preprocess the data for the selected model
    x, y = _preprocess_batch(data, labels, pixels, dtype=dtype, model=model)

    # Change the shape of the RGB array so that it is compatible with the classifiers.
    x = x.reshape((x.shape[0], 3, pixels, pixels))
    # Grayscale images if requested
    if grayscale:
        x = _grayscale(x, pixels)

    return x, y


def _grayscale(a, pixels):
    """
    Grayscale images
    :param a: image array
    :param pixels: size of image
    :return: grayscaled a
    """
    return a.reshape(a.shape[0], 3, pixels, pixels).mean(1).reshape(a.shape[0], -1)


def _preprocess_vgg(batch, pixels):
    """
    Preprocess data for the VGG-16 or VGG-19 models.
    :param batch: image dataset
    :param pixels: image size
    :return: preprocessed image data
    """
    rgb_mean = [123.68, 116.779, 103.939]
    img_size = pixels * pixels
    # Subtract the mean for the red/green/blue channels
    batch[:, 0:img_size] -= rgb_mean[0]
    batch[:, img_size:(2 * img_size)] -= rgb_mean[1]
    batch[:, (2 * img_size):(3 * img_size)] -= rgb_mean[2]
    return batch


def _preprocess_googlenet(batch, pixels):
    """
        Preprocess data for the GoogleNet model.
        :param batch: image dataset
        :param pixels: image size
        :return: preprocessed image data
        """
    rgb_mean = np.asarray([123, 117, 104])
    img_size = pixels * pixels
    # Subtract the mean for the red/green/blue channels
    batch[:, 0:img_size] -= rgb_mean[0]
    batch[:, img_size:(2 * img_size)] -= rgb_mean[1]
    batch[:, (2 * img_size):(3 * img_size)] -= rgb_mean[2]
    return batch


def _preprocess_inception_v3(batch):
    """
        Preprocess data for the inception_v3 model.
        :param batch: image dataset
        :return: preprocessed image data
        """
    # Change data to zero-mean
    data = (batch - 128) / 128.
    return data


def _preprocess_original(batch):
    """
        Preprocess data for first implementation
    :param batch: image dataset
    :return: preprocessed image dataset
    """
    return batch / 255.0  # scale between [0, 1]
