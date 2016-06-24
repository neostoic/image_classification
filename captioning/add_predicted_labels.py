# coding: utf-8

# This script uses a CNN model to obtain the predicted label for all of the images in the dataset and creates a new
# JSON file that contains the predicted label
# Adapted from the Neural networks with Theano and Lasagne tutorial from Eben Nolson
# available at: https://github.com/ebenolson/pydata2015/blob/master/4%20-%20Recurrent%20Networks/COCO%20Preprocessing.ipynb


import cPickle as pickle
import json

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import theano
import theano.tensor as T
from lasagne.layers import DenseLayer, NonlinearityLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.utils import floatX

from models import googlenet

# Setup to obtain predicted labels
train_pkl = r'../data/restaurant_photos_with_labels_train.pkl'
test_pkl = r'../data/restaurant_photos_with_labels_test.pkl'
img_path = r'../data/restaurant_photos/'
CLASSES = pickle.load(open(r'../data/categories.pkl', 'rb'))
model_path = r'../data/models/'


def prep_image(im):
    # The images need some preprocessing before they can be fed to the CNN
    MEAN_VALUES = np.array([104, 117, 123]).reshape((3, 1, 1))

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


def init_model():
    # Build the model and select layers we need - the features are taken from the final network layer,
    # before the softmax nonlinearity.
    cnn_layers = googlenet.build_model()
    cnn_input_var = cnn_layers['input'].input_var
    cnn_feature_layer = DenseLayer(cnn_layers['pool5/7x7_s1'],
                                   num_units=len(CLASSES),
                                   nonlinearity=linear)
    cnn_output_layer = NonlinearityLayer(cnn_feature_layer,
                                         nonlinearity=softmax)
    get_cnn_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_feature_layer))
    logging.info('Compiled new functions')
    # Define the function for label prediction.
    X_sym = T.tensor4()
    prediction = lasagne.layers.get_output(cnn_output_layer, X_sym)
    pred_fn = theano.function([X_sym], prediction)

    # Load the pretrained weights into the network
    with np.load(r'../data/googlenet_model.npz') as f:
        model_param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    lasagne.layers.set_all_param_values(cnn_output_layer, model_param_values)
    return get_cnn_features, pred_fn


# Iterate over the dataset and add a field 'cnn features' to each item.
def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def get_labels_features():
    # Initialize model and obtain prediction function
    get_cnn_features, pred_fn = init_model()
    # Load the caption data
    dataset = json.load(open(r'../data/image_caption_dataset.json'))['images']
    # Obtain the predicted label for each image and store it in a dictionary
    for chunk in chunks(dataset, 256):
        cnn_input = floatX(np.zeros((len(chunk), 3, 224, 224)))
        for i, image in enumerate(chunk):
            fn = r'../data/restaurant_photos_split/{}/{}'.format(image['filepath'], image['filename'])
            try:
                im = plt.imread(fn)
                cnn_input[i] = prep_image(im)
            except IOError:
                continue
        labels = pred_fn(cnn_input)
        # Obtain the CNN features to append them to the JSON data
        features = get_cnn_features(cnn_input)
        for i, image in enumerate(chunk):
            # We use the argmax function to select the class with the highest probability, we could actually add top 2
            # or more and display the probability for each class.
            image['predicted_label'] = CLASSES[np.argmax(labels[i])]
            image['cnn features'] = features[i]

    # Store the dictionary with the results as a JSON file
    with open(r'../data/image_caption_dataset_labels.json', 'wb') as outfile:
        dump_dict = dict()
        dump_dict['images'] = dataset
        json.dump(dump_dict, outfile)

    # Save the dataset as a compressed pickle file
    pickle.dump(dataset,
                open(r'../data/image_caption_with_cnn_features.pkl', 'w'),
                protocol=pickle.HIGHEST_PROTOCOL)
