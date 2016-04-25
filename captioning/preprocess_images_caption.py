# coding: utf-8

# # Image Captioning with LSTM
# 

import cPickle as pickle
import json

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import theano
from lasagne.utils import floatX

from models import googlenet

# Build the model and select layers we need - the features are taken from the final network layer, before the softmax nonlinearity.
cnn_layers = googlenet.build_model()
cnn_input_var = cnn_layers['input'].input_var
cnn_feature_layer = cnn_layers['loss3/classifier']
cnn_output_layer = cnn_layers['prob']

get_cnn_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_feature_layer))

# Load the pretrained weights into the network
model_param_values = \
pickle.load(open(r'/home/rcamachobarranco/datasets/googlenet.pkl', mode='rb'))[
    'param values']
lasagne.layers.set_all_param_values(cnn_output_layer, model_param_values)

# The images need some preprocessing before they can be fed to the CNN

MEAN_VALUES = np.array([104, 117, 123]).reshape((3, 1, 1))


def prep_image(im):
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

    im = im - MEAN_VALUES
    return floatX(im[np.newaxis])


# Load the caption data
dataset = json.load(open(r'/home/rcamachobarranco/datasets/caption_dataset/image_caption_dataset.json'))['images']


# Iterate over the dataset and add a field 'cnn features' to each item. This will take quite a while.
def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


for chunk in chunks(dataset, 256):
    cnn_input = floatX(np.zeros((len(chunk), 3, 224, 224)))
    for i, image in enumerate(chunk):
        fn = r'/home/rcamachobarranco/datasets/caption_dataset/{}/{}'.format(image['filepath'], image['filename'])
        try:
            im = plt.imread(fn)
            cnn_input[i] = prep_image(im)
        except IOError:
            continue
    features = get_cnn_features(cnn_input)
    for i, image in enumerate(chunk):
        image['cnn features'] = features[i]

# Save the final product
pickle.dump(dataset, open('/home/rcamachobarranco/datasets/caption_dataset/image_caption_with_cnn_features.pkl', 'w'),
            protocol=pickle.HIGHEST_PROTOCOL)
