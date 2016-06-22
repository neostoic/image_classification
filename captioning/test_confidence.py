# This script uses the CNN model and the LSTM model trained previously to obtain the predicted captions for all of the
# images in the dataset and creates a new JSON file that contains the predicted captions

import json
import logging
import pickle

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import theano
import theano.tensor as T
from lasagne.layers import DenseLayer, NonlinearityLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.utils import floatX
from scipy.stats import entropy

from models import googlenet

# Read the dictionary with the classes
CLASSES = pickle.load(open(r'/home/rcamachobarranco/code/data/categories.pkl', 'rb'))

# Initialize logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Build the CNN model and select layers we need - the features are taken from the final network layer,
# before the softmax classifier.
cnn_layers = googlenet.build_model()
logging.info('GoogleNet model built')
cnn_input_var = cnn_layers['input'].input_var
cnn_feature_layer = DenseLayer(cnn_layers['pool5/7x7_s1'],
                               num_units=len(CLASSES),
                               nonlinearity=linear)
cnn_output_layer = NonlinearityLayer(cnn_feature_layer,
                                     nonlinearity=softmax)

get_cnn_features = theano.function([cnn_input_var], lasagne.layers.get_output(cnn_feature_layer))

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


# Initialize parameters from RNN
SEQUENCE_LENGTH = 32
MAX_SENTENCE_LENGTH = SEQUENCE_LENGTH - 3  # 1 for image, 1 for start token, 1 for end token
BATCH_SIZE = 1
CNN_FEATURE_SIZE = 5
EMBEDDING_SIZE = 512

d = pickle.load(open(r'/home/rcamachobarranco/datasets/caption_dataset/lstm_yelp_trained.pkl', mode='rb'))
# d = pickle.load(open(r'/home/rcamachobarranco/datasets/caption_dataset/lstm_coco_trained.pkl', mode='rb'))
vocab = d['vocab']
word_to_index = d['word_to_index']
index_to_word = d['index_to_word']

# Build the LSTM model without the parameters
l_input_sentence = lasagne.layers.InputLayer((BATCH_SIZE, SEQUENCE_LENGTH - 1))
l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input_sentence,
                                                     input_size=len(vocab),
                                                     output_size=EMBEDDING_SIZE,
                                                     )

l_input_cnn = lasagne.layers.InputLayer((BATCH_SIZE, CNN_FEATURE_SIZE))
l_cnn_embedding = lasagne.layers.DenseLayer(l_input_cnn, num_units=EMBEDDING_SIZE,
                                            nonlinearity=lasagne.nonlinearities.identity)

l_cnn_embedding = lasagne.layers.ReshapeLayer(l_cnn_embedding, ([0], 1, [1]))

l_rnn_input = lasagne.layers.ConcatLayer([l_cnn_embedding, l_sentence_embedding])
l_dropout_input = lasagne.layers.DropoutLayer(l_rnn_input, p=0.5)
l_lstm = lasagne.layers.LSTMLayer(l_dropout_input,
                                  num_units=EMBEDDING_SIZE,
                                  unroll_scan=True,
                                  grad_clipping=5.)
l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)
l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, EMBEDDING_SIZE))
l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=len(vocab), nonlinearity=lasagne.nonlinearities.softmax)

l_out = lasagne.layers.ReshapeLayer(l_decoder, (BATCH_SIZE, SEQUENCE_LENGTH, len(vocab)))

# Initialize the parameters for the trained model
lasagne.layers.set_all_param_values(l_out, d['param values'])

x_cnn_sym = T.matrix()
x_sentence_sym = T.imatrix()

# Define the output function
output = lasagne.layers.get_output(l_out, {
    l_input_sentence: x_sentence_sym,
    l_input_cnn: x_cnn_sym
})

f = theano.function([x_cnn_sym, x_sentence_sym], output)


# Function to compute non-linearity score.
def compute_kl_divergence(orig_distr):
    n = len(orig_distr)
    n_inv = 1.0 / n
    sum_distr = np.sum(orig_distr)
    norm_distr = orig_distr / sum_distr

    unif_distr = []
    for i in range(n):
        unif_distr.append(n_inv)
    unif_distr = np.array(unif_distr)

    nonunif_distr = np.zeros(n)
    nonunif_distr[0] = 1.0

    ref_divergence = entropy(nonunif_distr, unif_distr, 10)
    kl_divergence = entropy(norm_distr, unif_distr, 10)
    return kl_divergence * 1.0 / ref_divergence


# Function to compute non-linearity score.
def compute_non_uniformity(orig_distr):
    n = len(orig_distr)
    n_inv = 1.0 / n
    sum_distr = np.sum(orig_distr)
    norm_distr = orig_distr / sum_distr

    unif_distr = []
    for i in range(n):
        unif_distr.append(n_inv)
    unif_distr = np.array(unif_distr)

    num = np.sum(np.abs(norm_distr - unif_distr))
    non_uniformity = num/(2.0 - (2.0 / n))

    return non_uniformity


def compute_confidence(probabilities, non_uniformity_scores):
    m = len(probabilities)
    probabilities = np.array(probabilities)
    non_uniformity_scores = np.array(non_uniformity_scores)

    confidence = (np.sum(np.exp(probabilities * non_uniformity_scores)) - m) / (m * np.exp(1) - m)
    #confidence = np.sum(np.log(probabilities * non_uniformity_scores))

    return confidence


# Define the function used to predict a caption using #START# and #END# as markers.
def predict(x_cnn):
    x_sentence = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH - 1), dtype='int32')
    words = []
    non_uniformity_scores = []
    probabilities = []
    i = 0
    while True:
        i += 1
        p0 = f(x_cnn, x_sentence)
        arr = p0[0][i]
        pa = p0.argmax(-1)
        tok = pa[0][i]
        probabilities.append(arr[tok])
        word = index_to_word[tok]
        n = 3
        top_n = np.argsort(-arr)[:n]
        prob_top_n = arr[top_n]
        non_uniformity_scores.append(compute_non_uniformity(prob_top_n))
        logging.info('Predicted words: {0} Probability: {1}'.format(word.encode('utf8'), probabilities[i-1]))
        logging.info('Top {0} probs: {1}'.format(n, prob_top_n))
        logging.info('Non-uniformity scores: {0} KL-divergence: {1}'.format(non_uniformity_scores[i-1],
                                                                            compute_kl_divergence(prob_top_n)))
        if word == '#END#' or i >= SEQUENCE_LENGTH - 1:
            confidence = compute_confidence(probabilities, non_uniformity_scores)
            return ' '.join(words), confidence
        else:
            x_sentence[0][i] = tok
            if word != '#START#':
                words.append(word)


# # Iterate over the dataset and add a field 'cnn features' to each item. This will take quite a while.
# def chunks(l, n):
#     for i in xrange(0, len(l), n):
#         yield l[i:i + n]


# Load the caption data
dataset = json.load(open(r'/home/rcamachobarranco/datasets/caption_dataset/image_caption_dataset.json'))['images']

chunk = dataset[1:10]
cnn_input = floatX(np.zeros((len(chunk), 3, 224, 224)))
for i, image in enumerate(chunk):
    fn = r'/home/rcamachobarranco/datasets/caption_dataset/{}/{}'.format(image['filepath'], image['filename'])
    try:
        im = plt.imread(fn)
        cnn_input = prep_image(im)
        features = get_cnn_features(cnn_input)
        predicted = []
        confidence_scores = []
        for _ in range(5):
            new_prediction, new_score = predict(features)
            confidence_scores.append(new_score)
            predicted.append(new_prediction.encode('utf8'))
        image['predicted caption'] = predicted
        image['caption confidence'] = confidence_scores
        for idx in range(len(predicted)):
            logging.info('Prediction: {0} Confidence: {1}'.format(predicted[idx], confidence_scores[idx]))
        image['predicted label'] = CLASSES[np.argmax(features)]
        logging.info(str(i) + ': Actual: ' + image['sentences'][0]['raw'].encode('utf8') + ' Predicted: ' + '; '.join(
            predicted))
    except IOError:
        continue

        # # Predict 5 captions per image for all of the images in the dataset
        # for chunk in chunks(dataset, 256):
        #     cnn_input = floatX(np.zeros((len(chunk), 3, 224, 224)))
        #     for i, image in enumerate(chunk):
        #         fn = r'/home/rcamachobarranco/datasets/caption_dataset/{}/{}'.format(image['filepath'],
        #                                                                              image['filename'])
        #         try:
        #             im = plt.imread(fn)
        #             cnn_input = prep_image(im)
        #             features = get_cnn_features(cnn_input)
        #             predicted = []
        #             for _ in range(5):
        #                 predicted.append(predict(features).encode('utf8'))
        #             image['predicted caption'] = predicted
        #             image['predicted label'] = CLASSES[np.argmax(features)]
        #             print str(i) + ': Actual: ' + image['sentences'][0]['raw'].encode(
        #                 'utf8') + ' Predicted: ' + '; '.join(
        #                 predicted)
        #         except IOError:
        #             continue

        # # Save the dictionary to a JSON file
        # with open('/home/rcamachobarranco/datasets/caption_dataset/results_yelp_84_62.json', 'w') as outfile:
        #     # We use dump_dict to create the proper JSON structure
        #     dump_dict = dict()
        #     dump_dict['images'] = dataset_images
        #     json.dump(dump_dict, outfile)
