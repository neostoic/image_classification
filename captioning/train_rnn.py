import cPickle as pickle
import random
from collections import Counter

import lasagne
import numpy as np
from lasagne.utils import floatX
import theano.tensor as T
import theano
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load the preprocessed dataset containing features extracted by GoogLeNet
dataset = pickle.load(open('/home/rcamachobarranco/datasets/caption_dataset/image_caption_with_cnn_features.pkl'))
#dataset = pickle.load(
#    open(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\image_caption_with_cnn_features.pkl', mode='rb'))
# Count words occuring at least 5 times and construct mapping int <-> word
allwords = Counter()
for item in dataset:
    for sentence in item['sentences']:
        allwords.update(sentence['tokens'])

vocab = [k for k, v in allwords.items() if (v >= 2 and k not in ['best', 'Best', '!', 'ever'])]
vocab.insert(0, '#START#')
vocab.append('#END#')

word_to_index = {w: i for i, w in enumerate(vocab)}
index_to_word = {i: w for i, w in enumerate(vocab)}

logging.info('Size of vocabulary: {0}'.format(len(vocab)))

SEQUENCE_LENGTH = 9
MAX_SENTENCE_LENGTH = SEQUENCE_LENGTH - 3  # 1 for image, 1 for start token, 1 for end token
BATCH_SIZE = 75
CNN_FEATURE_SIZE = 5
EMBEDDING_SIZE = 1024
LR = 0.001
BATCH_SIZE = 75
ITERATIONS = 19000
# Configuration 23213: 512:0.001:20000
# Configuration 23214: 256:0.001:20000
# Configuration 23216: 512:0.0001:20000
# Configuration 23217: 1024:0.0001:20000
# Configuration 23227: 512:0.01:50000
# Configuration 23229: 512:0.001:50000
# Configuration 23231: 512:0.001:20000 BS=200
# Configuration 23232: 1024:0.001:20000 BS=200
# Configuration 23233: 1024:0.0001:20000 BS=200 SEQ = 32
# Configuration 23234: 512:0.0001:20000 BS=200 SEQ=16
# Configuration 23235: 512:0.0001:20000 BS=100 SEQ=8
# Configuration 23242: 1024:0.0001:20000 BS=200 SEQ=13
# Configuration 23243: 512:0.0001:20000 BS=200 SEQ=13 v >= 5
# Configuration 23246: 512:0.0001:75000 BS=100 SEQ=15 removing best ! AND v >= 4
# Config        23318: 512:0.001:25000 BS=200 SEQ=13 removing best ! AND v >= 3
# Config        23319: 512:0.0001:25000 BS=100 SEQ=11 removing best ! AND v >= 3
# Config        23322: 512:0.0001:25000 BS=100 SEQ=11 removing best ! AND v >= 2
# Config        23323: 1024:0.001:25000 BS=100 SEQ=11 removing best ! AND v >= 2


logging.info('Embeddings: {0} Learning rate: {1} Iter: {2}'.format(EMBEDDING_SIZE, LR, ITERATIONS))

# Returns a list of tuples (cnn features, list of words, image ID)
def get_data_batch(dataset, size, split='train'):
    items = []

    while len(items) < size:
        item = random.choice(dataset)
        if item['split'] != split:
            continue
        sentence = random.choice(item['sentences'])['tokens']
        if len(sentence) > MAX_SENTENCE_LENGTH:
            continue
        items.append((item['cnn features'], sentence, item['yelpid']))

    return items


# Convert a list of tuples into arrays that can be fed into the network
def prep_batch_for_network(batch):
    x_cnn = floatX(np.zeros((len(batch), CNN_FEATURE_SIZE)))
    x_sentence = np.zeros((len(batch), SEQUENCE_LENGTH - 1), dtype='int32')
    y_sentence = np.zeros((len(batch), SEQUENCE_LENGTH), dtype='int32')
    mask = np.zeros((len(batch), SEQUENCE_LENGTH), dtype='bool')

    for j, (cnn_features, sentence, _) in enumerate(batch):
        x_cnn[j] = cnn_features
        i = 0
        for word in ['#START#'] + sentence + ['#END#']:
            if word in word_to_index:
                mask[j, i] = True
                y_sentence[j, i] = word_to_index[word]
                x_sentence[j, i] = word_to_index[word]
                i += 1
        mask[j, 0] = False

    return x_cnn, x_sentence, y_sentence, mask


# sentence embedding maps integer sequence with dim (BATCH_SIZE, SEQUENCE_LENGTH - 1) to
# (BATCH_SIZE, SEQUENCE_LENGTH-1, EMBEDDING_SIZE)
l_input_sentence = lasagne.layers.InputLayer((BATCH_SIZE, SEQUENCE_LENGTH - 1))
l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input_sentence,
                                                     input_size=len(vocab),
                                                     output_size=EMBEDDING_SIZE,
                                                     )

# cnn embedding changes the dimensionality of the representation from 1000 to EMBEDDING_SIZE,
# and reshapes to add the time dimension - final dim (BATCH_SIZE, 1, EMBEDDING_SIZE)
l_input_cnn = lasagne.layers.InputLayer((BATCH_SIZE, CNN_FEATURE_SIZE))
l_cnn_embedding = lasagne.layers.DenseLayer(l_input_cnn, num_units=EMBEDDING_SIZE,
                                            nonlinearity=lasagne.nonlinearities.identity)

l_cnn_embedding = lasagne.layers.ReshapeLayer(l_cnn_embedding, ([0], 1, [1]))

# the two are concatenated to form the RNN input with dim (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE)
l_rnn_input = lasagne.layers.ConcatLayer([l_cnn_embedding, l_sentence_embedding])

l_dropout_input = lasagne.layers.DropoutLayer(l_rnn_input, p=0.5)
l_lstm = lasagne.layers.LSTMLayer(l_dropout_input,
                                  num_units=EMBEDDING_SIZE,
                                  unroll_scan=True,
                                  grad_clipping=5.)
l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)

# the RNN output is reshaped to combine the batch and time dimensions
# dim (BATCH_SIZE * SEQUENCE_LENGTH, EMBEDDING_SIZE)
l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, EMBEDDING_SIZE))

# decoder is a fully connected layer with one output unit for each word in the vocabulary
l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=len(vocab), nonlinearity=lasagne.nonlinearities.softmax)

# finally, the separation between batch and time dimension is restored
l_out = lasagne.layers.ReshapeLayer(l_decoder, (BATCH_SIZE, SEQUENCE_LENGTH, len(vocab)))

# Define symbolic variables for the various inputs
# cnn feature vector
x_cnn_sym = T.matrix()

# sentence encoded as sequence of integer word tokens
x_sentence_sym = T.imatrix()

# mask defines which elements of the sequence should be predicted
mask_sym = T.imatrix()

# ground truth for the RNN output
y_sentence_sym = T.imatrix()

output = lasagne.layers.get_output(l_out, {
    l_input_sentence: x_sentence_sym,
    l_input_cnn: x_cnn_sym
})


def calc_cross_ent(net_output, mask, targets):
    # Helper function to calculate the cross entropy error
    preds = T.reshape(net_output, (-1, len(vocab)))
    targets = T.flatten(targets)
    cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask).nonzero()]
    return cost


loss = T.mean(calc_cross_ent(output, mask_sym, y_sentence_sym))

MAX_GRAD_NORM = 15

all_params = lasagne.layers.get_all_params(l_out, trainable=True)

all_grads = T.grad(loss, all_params)
all_grads = [T.clip(g, -5, 5) for g in all_grads]
all_grads, norm = lasagne.updates.total_norm_constraint(
    all_grads, MAX_GRAD_NORM, return_norm=True)

updates = lasagne.updates.adam(all_grads, all_params, learning_rate=LR)

f_train = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym],
                          [loss, norm],
                          updates=updates
                          )

f_val = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym], loss)

for iteration in range(ITERATIONS):
    x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(get_data_batch(dataset, BATCH_SIZE))
    loss_train, norm = f_train(x_cnn, x_sentence, mask, y_sentence)
    if not iteration % 250:
        logging.info('Iteration {} loss_train: {} norm: {}'.format(iteration, loss_train, norm))
        try:
            batch = get_data_batch(dataset, BATCH_SIZE, split='val')
            x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(batch)
            loss_val = f_val(x_cnn, x_sentence, mask, y_sentence)
            logging.info('Val loss: {}'.format(loss_val))
        except IndexError:
            continue

param_values = lasagne.layers.get_all_param_values(l_out)
d = {'param values': param_values,
     'vocab': vocab,
     'word_to_index': word_to_index,
     'index_to_word': index_to_word,
     }
pickle.dump(d, open(r'lstm_yelp_trained_{}_{}_{}_webpage.pkl'.format(EMBEDDING_SIZE, LR*10000, ITERATIONS), 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)
