# coding: utf-8

# This script uses a CNN model to obtain image features for all of the images in the dataset which will be used by the
# captioning algorithm
# Adapted from the Neural networks with Theano and Lasagne tutorial from Eben Nolson
# available at:
# https://github.com/ebenolson/pydata2015/blob/master/4%20-%20Recurrent%20Networks/COCO%20Preprocessing.ipynb

import cPickle as pickle
import json
import logging

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import theano
from lasagne.layers import DenseLayer, NonlinearityLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.utils import floatX

from models import googlenet

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Setup to obtain predicted labels
train_pkl = r'/home/rcamachobarranco/datasets/restaurant_photos_with_labels_train.pkl'
test_pkl = r'/home/rcamachobarranco/datasets/restaurant_photos_with_labels_test.pkl'
img_path = r'/home/rcamachobarranco/datasets/restaurant_photos/'
CLASSES = pickle.load(open(r'/home/rcamachobarranco/code/data/categories.pkl', 'rb'))
model_path = r'/home/rcamachobarranco/datasets/'

# Build the model and select layers we need - the features are taken from the final network layer, before the softmax nonlinearity.
cnn_layers = googlenet.build_model()
logging.info('Built googlenet model')

cnn_input_var = cnn_layers['input'].input_var
cnn_feature_layer = DenseLayer(cnn_layers['pool5/7x7_s1'],
                                     num_units=len(CLASSES),
                                     nonlinearity=linear)
cnn_output_layer = NonlinearityLayer(cnn_feature_layer,
                                nonlinearity=softmax)

get_cnn_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_feature_layer))
logging.info('Compiled new functions')

# Load the pretrained weights into the network
with np.load('/home/rcamachobarranco/datasets/googlenet_model_84_69.npz') as f:
     model_param_values = [f['arr_%d' % i] for i in range(len(f.files))]
logging.info('Read parameters from file')

lasagne.layers.set_all_param_values(cnn_output_layer, model_param_values)
logging.info('All parameters are set') 

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
logging.info('JSON dataset loaded')


# Iterate over the dataset and add a field 'cnn features' to each item.
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
    # Obtain the CNN features to append them to the JSON data
    features = get_cnn_features(cnn_input)
    for i, image in enumerate(chunk):
        image['cnn features'] = features[i]
logging.info('Finished extracting features')

# Save the dataset as a compressed pickle file
pickle.dump(dataset, open('/home/rcamachobarranco/datasets/caption_dataset/image_caption_with_cnn_features.pkl', 'w'),
            protocol=pickle.HIGHEST_PROTOCOL)
logging.info('Data pickled correctly, we are done!')
