# This script uses the CNN model and the LSTM model trained previously to obtain the predicted captions for all of the
# images in the dataset and creates a new JSON file that contains the predicted captions
#  This is a partial implementation of "Show and Tell: A Neural Image Caption Generator"
# (http://arxiv.org/abs/1411.4555), borrowing heavily from Andrej Karpathy's NeuralTalk (https://github.com/karpathy/neuraltalk)
# And adapted from the Neural networks with Theano and Lasagne tutorial from Eben Nolson
# available at: https://github.com/ebenolson/pydata2015/blob/master/4%20-%20Recurrent%20Networks/COCO%20Caption%20Generation.ipynb

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

from models import googlenet

# Read the dictionary with the classes
#CLASSES = pickle.load(open(r'/home/rcamachobarranco/code/data/categories.pkl', 'rb'))
CLASSES = pickle.load(open(r'C:\Users\crobe\Google Drive\DataMiningGroup\Code/data/categories.pkl', 'rb'))

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
#with np.load('/home/rcamachobarranco/datasets/googlenet_model_87_71.npz') as f:
with np.load(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\caption_results/googlenet_model_87_71.npz') as f:
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
SEQUENCE_LENGTH = 11
MAX_SENTENCE_LENGTH = SEQUENCE_LENGTH - 3  # 1 for image, 1 for start token, 1 for end token
BATCH_SIZE = 1
CNN_FEATURE_SIZE = 5
EMBEDDING_SIZE = 1024

d = pickle.load(open(r'C:\Users\crobe\Google Drive\DataMiningGroup\Code\captioning/lstm_yelp_trained_9_75_1024_0.001_webpage.pkl', mode='rb'))
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


# Iterate over the dataset and add a field 'cnn features' to each item. This will take quite a while.
def chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


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
    non_uniformity = num / (2.0 - (2.0 / n))

    return non_uniformity


def compute_confidence(probabilities, non_uniformity_scores):
    m = len(probabilities)
    probabilities = np.array(probabilities)
    non_uniformity_scores = np.array(non_uniformity_scores)

    confidence = (np.sum(np.exp(probabilities * non_uniformity_scores)) - m) / (m * np.exp(1) - m)

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

        # Obtain the word with maximum probability
        pa = p0.argmax(-1)
        tok = pa[0][i]
        word = index_to_word[tok]
        # Store the probability to use with confidence computation
        probabilities.append(arr[tok])

        # Define number of top probabilities per position used to compute confidence
        n = 3
        # Obtain the indices of the n most-probable terms for position i
        top_n = np.argsort(-arr)[:n]
        # Obtain the probabilities for the n most-probable terms for position i
        prob_top_n = arr[top_n]
        # Compute non-uniformity value per position to compute confidence later
        non_uniformity_scores.append(compute_non_uniformity(prob_top_n))
        logging.debug('Predicted words: {0} Probability: {1}'.format(word.encode('utf8'), probabilities[i - 1]))
        logging.debug('Top {0} probs: {1}'.format(n, prob_top_n))
        logging.debug('Non-uniformity score: {0}'.format(non_uniformity_scores[i - 1]))
        if word == '#END#' or i >= SEQUENCE_LENGTH - 1:
            confidence = compute_confidence(probabilities, non_uniformity_scores)
            return ' '.join(words), confidence
        else:
            x_sentence[0][i] = tok
            if word != '#START#':
                words.append(word)


# Load the caption data
#dataset = json.load(open(r'/home/rcamachobarranco/datasets/caption_dataset/image_caption_dataset.json'))['images']
dataset = json.load(open(r'C:\Users\crobe\Google Drive\DataMiningGroup\Code\captioning/image_caption_dataset.json'))['images']


results_dataset = []

# Predict 5 captions per image for all of the images in the dataset
for chunk in chunks(dataset, 256):
    cnn_input = floatX(np.zeros((len(chunk), 3, 224, 224)))
    for i, image in enumerate(chunk):
        fn = r'D:\Yelp\caption_dataset/{}/{}'.format(image['filepath'], image['filename'])
        #fn = r'/home/rcamachobarranco/datasets/caption_dataset/{}/{}'.format(image['filepath'], image['filename'])
        try:
            # # Obtain the original image filenames and id
            # img_dataset_dict = dict()
            # img_dataset_dict["license"] = 3
            # img_dataset_dict["url"] = image['filepath']
            # img_dataset_dict["file_name"] = image['filename']
            # img_dataset_dict["id"] = int(image['photoid'])
            # img_dataset_dict["width"] = 224
            # img_dataset_dict["date_captured"] = "Unknown"
            # img_dataset_dict["height"] = 224
            # eval_img_dataset.append(img_dataset_dict)
            # ann_dataset_dict = dict()
            # ann_dataset_dict["image_id"] = int(image['photoid'])
            # ann_dataset_dict["id"] = int(image['photoid'])
            # ann_dataset_dict["caption"] = image['sentences'][0]['raw'].encode('utf8')
            # ann_dataset_dict["split"] = image['split'].encode('utf8')
            # eval_ann_dataset.append(ann_dataset_dict)

            result_dataset_dict = dict()
            im = plt.imread(fn)
            cnn_input = prep_image(im)
            features = get_cnn_features(cnn_input)
            predicted = []
            confidence_scores = []
            result_dataset_dict['image_id'] = int(image['photoid'])
            for _ in range(5):
                new_prediction, new_score = predict(features)
                # if new_prediction not in predicted:
                #     confidence_scores.append(new_score)
                #     predicted.append(new_prediction.encode('utf8'))
                confidence_scores.append(new_score)
                predicted.append(new_prediction.encode('utf8'))
                result_dataset_dict['caption'] = new_prediction.encode('utf8')
                result_dataset_dict['confidence_score'] = new_score
                if image["split"] in ["test"]:
                    results_dataset.append(result_dataset_dict)
            image['predicted caption'] = predicted

            best_caption = np.argmax(confidence_scores)
            image['top caption'] = predicted[best_caption]
            image['confidence'] = confidence_scores
            image['top confidence'] = confidence_scores[best_caption]

            label_idx = np.argmax(features)
            image['predicted label'] = CLASSES[label_idx]

            logging.debug('Actual label: {0}, Predicted label: {1}'.format(image['label'], CLASSES[label_idx]))
            logging.info('Actual caption: {0}'.format(image['sentences'][0]['raw'].encode('utf8')))
            for idx in range(len(predicted)):
                logging.debug('Generated caption: {0} ; Confidence: {1}'.format(predicted[idx], confidence_scores[idx]))
            logging.info(
                'Best caption: {0}; Confidence: {1}'.format(predicted[best_caption], confidence_scores[best_caption]))

        except IOError:
            continue

# Save the dictionary to a JSON file
#with open('/home/rcamachobarranco/datasets/caption_dataset/results_yelp_webpage.json', 'w') as outfile:
with open('results_yelp_webpage.json', 'w') as outfile:
    # We use dump_dict to create the proper JSON structure
    dump_dict = dict()
    dump_dict['images'] = dataset
    json.dump(dump_dict, outfile)

# eval_dataset["images"] = eval_img_dataset
# eval_dataset["annotations"] = eval_ann_dataset
#
# with open('/home/rcamachobarranco/datasets/caption_dataset/annotations/captions_yelp_23323.json', 'w') as outfile:
#     json.dump(eval_dataset, outfile)

with open('captions_yelp_nic_results_webpage.json', 'w') as outfile:
#with open('/home/rcamachobarranco/datasets/caption_dataset/results/captions_yelp_nic_results_webpage.json', 'w') as outfile:
    json.dump(results_dataset, outfile)
