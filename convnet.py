'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function

import lasagne
import numpy as np
import cPickle as pickle
import os
import theano
import theano.tensor as T

import matplotlib.pyplot as plt

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.utils import floatX
from multiprocessing import freeze_support

from batches.batchgen import BatchGen
from batches.load_batch import batch
import models.vgg16 as vgg16
import models.vgg19 as vgg19
import models.googlenet as googlenet
import models.inception_v3 as inception_v3

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
MODELS = ['VGG_16', 'VGG_19', 'googlenet', 'inception_v3']

# Config 23090 (8, 100, 5, 10, 0.001)
# Config 23091 (8, 100, 5, 10, 0.0001)
# Config 23092 (32, 100, 5, 50, 0.01)
# Config 23093 (32, 100, 5, 50, 0.001)
# Config 23094 (16, 100, 5, 50, 0.0001)
# Config 23095 (16, 100, 5, 50, 0.001) vgg16
# Config 23096 (16, 100, 5, 50, 0.001) googlenet
# Config 23101 (16, 100, 5, 50, 0.0001) vgg16 with 1000 test samples
# Config 23102 (16, 100, 5, 50, 0.0001) googlenet with 1000 test samples
# Config 23103 (16, 100, 5, 50, 0.00001) googlenet with 1000 test samples
# Config 23124 (16, 100, 5, 50, 0.001) vgg16 with 1000 test samples
# Config 23125 (16, 100, 5, 50, 0.001) googlenet with 1000 test samples
# Config 23127 (16, 100, 5, 50, 0.001) vgg19 with 1000 test samples
# Config 23154 (16, 100, 5, 50, 0.001) vgg16 with 1000 test samples
# Config 23155 (16, 100, 5, 50, 0.001) googlenet with 1000 test samples
# Config 23157 (16, 100, 5, 50, 0.001) inception_v3 with 1000 test samples
# Config 23197 googlenet (16, 50, 5, 50, 0.0001)
# Config 23198 googlenet (16, 50, 5, 50, 0.001)

def train(model_='VGG_16'):
    batch_size = 16
    nb_samples = 50
    nb_classes = 5
    nb_test_samples = 1000
    nb_epoch = 50
    data_augmentation = False

    if os.name in ['nt']:
        train_pkl = r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_labels_train.pkl'
        test_pkl = r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_labels_test.pkl'
        img_path = r'D:\Yelp\restaurant_photos\\'
        CLASSES = pickle.load(open(r'C:\Users\crobe\Google Drive\DataMiningGroup\Code\data\categories.pkl', 'rb'))
        model_path = r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\\'
    else:
        train_pkl = r'/home/rcamachobarranco/datasets/restaurant_photos_with_labels_train.pkl'
        test_pkl = r'/home/rcamachobarranco/datasets/restaurant_photos_with_labels_test.pkl'
        img_path = r'/home/rcamachobarranco/datasets/restaurant_photos/'
        CLASSES = pickle.load(open(r'/home/rcamachobarranco/code/data/categories.pkl', 'rb'))
        model_path = r'/home/rcamachobarranco/datasets/'

    # input image dimensions
    if model_ in MODELS[0:3]:
        img_rows, img_cols = 224, 224
    if model_ in MODELS[3]:
        img_rows, img_cols = 299, 299

    # generate model
    if model_ in MODELS[0]:
        net = vgg16.build_model()
        weights = pickle.load(open(model_path + 'vgg16.pkl', 'rb'))
        net_prob = net['prob']
    elif model_ in MODELS[1]:
        net = vgg19.build_model()
        weights = pickle.load(open(model_path + 'vgg19.pkl', 'rb'))
        net_prob = net['prob']
    elif model_ in MODELS[2]:
        net = googlenet.build_model()
        weights = pickle.load(open(model_path + 'googlenet.pkl', 'rb'))
        net_prob = net['prob']
    elif model_ in MODELS[3]:
        net = inception_v3.build_network()
        weights = pickle.load(open(model_path + 'inception_v3.pkl', 'rb'))
	net_prob = net['softmax']

    lasagne.layers.set_all_param_values(net_prob, weights['param values'])
    logging.info('Finished setting up weights')

    if model_ in MODELS[0]:
        output_layer = DenseLayer(net['fc7'], num_units=len(CLASSES), nonlinearity=softmax)
    elif model_ in MODELS[1]:
        output_layer = DenseLayer(net['fc7_dropout'], num_units=len(CLASSES), nonlinearity=softmax)
    elif model_ in MODELS[2]:
        net['classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                         num_units=len(CLASSES),
                                         nonlinearity=linear)
        output_layer = NonlinearityLayer(net['classifier'],
                                        nonlinearity=softmax)
    elif model_ in MODELS[3]:
        output_layer = DenseLayer(net['pool3'], num_units=len(CLASSES), nonlinearity=softmax)

    # Define loss function and metrics, and get an updates dictionary
    X_sym = T.tensor4()
    y_sym = T.ivector()

    prediction = lasagne.layers.get_output(output_layer, X_sym)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
    loss = loss.mean()

    acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym),
                 dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.001, momentum=0.9)

    logging.info('Created model, compiling functions')
    # Compile functions for training, validation and prediction
    train_fn = theano.function([X_sym, y_sym], loss, updates=updates)
    val_fn = theano.function([X_sym, y_sym], [loss, acc])
    pred_fn = theano.function([X_sym], prediction)
    logging.info('Compiled functions, creating batchgen object')

    with BatchGen(batch, seed=1, num_workers=1, input_pkl=train_pkl, img_path=img_path,
                  dtype=np.float32, pixels=img_rows, model=model_, batch_size=batch_size) as train_bg:
        with BatchGen(batch, seed=1, num_workers=1, input_pkl=test_pkl, img_path=img_path,
                      dtype=np.float32, pixels=img_rows, model=model_, batch_size=batch_size) as test_bg:
            logging.info('Created batchgen object')
            for epoch in range(nb_epoch):
                for sample in range(nb_samples):
                    sample = next(train_bg)
                    X_train = sample['x']
                    y_train = sample['y']

                    train_loss = train_fn(X_train, y_train)
                    loss_tot = 0.
                    acc_tot = 0.

                    for chunk in range(nb_test_samples):
                        test_sample = next(test_bg)
                        X_test = test_sample['x']
                        y_test = test_sample['y']
                        test_loss, acc = val_fn(X_test, y_test)
                        loss_tot += test_loss
                        acc_tot += acc

                    loss_tot /= nb_test_samples
                    acc_tot /= nb_test_samples

                    if acc_tot > 0.86:
                        np.savez('/home/rcamachobarranco/datasets/googlenet_{0:.4g}.npz'.format(acc_tot * 100), *lasagne.layers.get_all_param_values(output_layer))
                        #param_values = lasagne.layers.get_all_params(output_layer)
                        #pickle.dump(param_values, open(r'/home/rcamacho/dm/datasets/googlenet_{0:.4g}.pkl'.format(acc_tot * 100), 'wb'),
                        #protocol=pickle.HIGHEST_PROTOCOL)

                    logging.info('Epoch {0} Train_loss {1} Test_loss {2} Test_accuracy {3}'.format(epoch, train_loss, loss_tot, acc_tot * 100))
    

if __name__ == '__main__':
    freeze_support()
    train('googlenet')
