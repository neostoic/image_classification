# Experiments to optimize the hyperparameters for the NIC caption generator.
# This is a partial implementation of "Show and Tell: A Neural Image Caption Generator"
# (http://arxiv.org/abs/1411.4555),
# borrowing heavily from Andrej Karpathy's NeuralTalk (https://github.com/karpathy/neuraltalk)
# And adapted from the Neural networks with Theano and Lasagne tutorial from Eben Nolson
# available at:
# https://github.com/ebenolson/pydata2015/blob/master/4%20-%20Recurrent%20Networks/COCO%20RNN%20Training.ipynb

import cPickle as pickle
import logging
import random
from collections import Counter

import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.utils import floatX
from nltk.tokenize import TweetTokenizer
from nltk.translate.bleu_score import corpus_bleu

CNN_FEATURE_SIZE = 5

logging.basicConfig(filename='results_phase_2_2.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Returns a list of tuples (cnn features, list of words, image ID)
def get_data_batch(dataset_in, size, max_sentence_length, split='train'):
    items = []

    while len(items) < size:
        item_in = random.choice(dataset_in)
        if item_in['split'] != split:
            continue
        sentence_in = random.choice(item_in['sentences'])['tokens']
        if len(sentence_in) > max_sentence_length:
            continue
        items.append((item_in['cnn features'], sentence_in, item_in['yelpid']))

    return items


# Convert a list of tuples into arrays that can be fed into the network
def prep_batch_for_network(batch_in, sequence_length):
    x_cnn_in = floatX(np.zeros((len(batch_in), CNN_FEATURE_SIZE)))
    x_sentence_in = np.zeros((len(batch_in), sequence_length - 1), dtype='int32')
    y_sentence_in = np.zeros((len(batch_in), sequence_length), dtype='int32')
    mask_in = np.zeros((len(batch_in), sequence_length), dtype='bool')

    for j, (cnn_features, sentence_in, _) in enumerate(batch_in):
        x_cnn_in[j] = cnn_features
        i = 0
        for word in ['#START#'] + sentence_in + ['#END#']:
            if word in word_to_index:
                mask_in[j, i] = True
                y_sentence_in[j, i] = word_to_index[word]
                x_sentence_in[j, i] = word_to_index[word]
                i += 1
        mask_in[j, 0] = False

    return x_cnn_in, x_sentence_in, y_sentence_in, mask_in


def calc_cross_ent(net_output, mask_in, targets):
    # Helper function to calculate the cross entropy error
    preds = T.reshape(net_output, (-1, len(vocab)))
    targets = T.flatten(targets)
    cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask_in).nonzero()]
    return cost


def create_rnn_model(batch_size, sequence_length, embedding_size, lr):
    # sentence embedding maps integer sequence with dim (BATCH_SIZE, SEQUENCE_LENGTH - 1) to
    # (BATCH_SIZE, SEQUENCE_LENGTH-1, EMBEDDING_SIZE)
    l_input_sentence = lasagne.layers.InputLayer((batch_size, sequence_length - 1))
    l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input_sentence,
                                                         input_size=len(vocab),
                                                         output_size=embedding_size,
                                                         )

    # cnn embedding changes the dimensionality of the representation from 1000 to EMBEDDING_SIZE,
    # and reshapes to add the time dimension - final dim (BATCH_SIZE, 1, EMBEDDING_SIZE)
    l_input_cnn = lasagne.layers.InputLayer((batch_size, CNN_FEATURE_SIZE))
    l_cnn_embedding = lasagne.layers.DenseLayer(l_input_cnn, num_units=embedding_size,
                                                nonlinearity=lasagne.nonlinearities.identity)

    l_cnn_embedding = lasagne.layers.ReshapeLayer(l_cnn_embedding, ([0], 1, [1]))

    # the two are concatenated to form the RNN input with dim (BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_SIZE)
    l_rnn_input = lasagne.layers.ConcatLayer([l_cnn_embedding, l_sentence_embedding])

    l_dropout_input = lasagne.layers.DropoutLayer(l_rnn_input, p=0.5)
    l_lstm = lasagne.layers.LSTMLayer(l_dropout_input,
                                      num_units=embedding_size,
                                      unroll_scan=True,
                                      grad_clipping=5.)
    l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)

    # the RNN output is reshaped to combine the batch and time dimensions
    # dim (BATCH_SIZE * SEQUENCE_LENGTH, EMBEDDING_SIZE)
    l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, embedding_size))

    # decoder is a fully connected layer with one output unit for each word in the vocabulary
    l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=len(vocab), nonlinearity=lasagne.nonlinearities.softmax)

    # finally, the separation between batch and time dimension is restored
    l_out = lasagne.layers.ReshapeLayer(l_decoder, (batch_size, sequence_length, len(vocab)))

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

    loss = T.mean(calc_cross_ent(output, mask_sym, y_sentence_sym))

    MAX_GRAD_NORM = 15

    all_params = lasagne.layers.get_all_params(l_out, trainable=True)

    all_grads = T.grad(loss, all_params)
    all_grads = [T.clip(g, -5, 5) for g in all_grads]
    all_grads, norm = lasagne.updates.total_norm_constraint(
        all_grads, MAX_GRAD_NORM, return_norm=True)

    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=lr)

    f_train = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym],
                              [loss, norm],
                              updates=updates
                              )

    f_val = theano.function([x_cnn_sym, x_sentence_sym, mask_sym, y_sentence_sym], loss)

    f_pred = theano.function([x_cnn_sym, x_sentence_sym], output)

    return f_train, f_val, f_pred, l_out


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
def predict(x_cnn_in, batch_size, sequence_length):
    x_sentence = np.zeros((batch_size, sequence_length - 1), dtype='int32')
    gen_captions = ["" for k in range(batch_size)]
    gen_confidence = [0 for k in range(batch_size)]
    indices = [k for k in range(batch_size)]
    words = [[] for k in range(batch_size)]
    non_uniformity_scores = [[] for k in range(batch_size)]
    probabilities = [[] for k in range(batch_size)]
    i = 0
    while len(indices) > 0:
        i += 1
        p0 = f_pred(x_cnn_in, x_sentence)
        pa = p0.argmax(-1)
        logging.debug(indices)
        for idx in indices[:]:
            arr = p0[idx][i]
            logging.debug('Index: {}'.format(idx))
            logging.debug(arr)
            # Obtain the word with maximum probability
            logging.debug(pa)
            logging.debug(p0.shape)
            logging.debug(pa.shape)
            tok = pa[idx][i]
            logging.debug(tok)
            word = index_to_word[tok]
            logging.debug(word)
            # Store the probability to use with confidence computation
            probabilities[idx].append(arr[tok])

            # Define number of top probabilities per position used to compute confidence
            n = 5
            # Obtain the indices of the n most-probable terms for position i
            top_n = np.argsort(-arr)[:n]
            # Obtain the probabilities for the n most-probable terms for position i
            prob_top_n = arr[top_n]
            # Compute non-uniformity value per position to compute confidence later
            non_uniformity_scores[idx].append(compute_non_uniformity(prob_top_n))
            logging.debug(
                'Predicted words: {0} Probability: {1}'.format(word.encode('utf8'), probabilities[idx][i - 1]))
            logging.debug('Top {0} probs: {1}'.format(n, prob_top_n))
            logging.debug('Non-uniformity score: {0}'.format(non_uniformity_scores[i - 1]))
            if word == '#END#' or i >= sequence_length - 1:
                gen_confidence[idx] = compute_confidence(probabilities[idx], non_uniformity_scores[idx])
                gen_captions[idx] = ' '.join(words[idx])
                indices.remove(idx)
                logging.debug(gen_captions[idx])
                logging.debug(gen_confidence[idx])
                logging.debug('Current index: {}'.format(idx))
                logging.debug(indices)
            else:
                x_sentence[idx][i] = tok
                if word != '#START#':
                    words[idx].append(word)

    return gen_captions, gen_confidence


def compute_bleu(batch_in, predicted):
    weights = []
    n = 4
    for i in range(n):
        weights.append(float(1.0 / n))

    # Create hypothesis and reference arrays taking 5 predicted captions per image (maybe we could modify this?)
    # Initialize hypothesis and reference arrays
    hypotheses = []
    references = []

    for j, (_, sentence_in, _) in enumerate(batch_in):
        references.append([sentence_in])
        hypotheses.append(tkn.tokenize(predicted[j]))

    # Compute BLEU score
    score = corpus_bleu(references, hypotheses, weights=weights)

    # Display BLEU score
    # logging.info('BLEU-{} score: {}'.format(n, score))

    return score


if __name__ == '__main__':
    # Load the preprocessed dataset containing features extracted by GoogLeNet
    with open('./data/image_caption_with_cnn_features.pkl', 'rb') as f_in:
        dataset = pickle.load(f_in)

    # Count words occuring at least 3 times and construct mapping int <-> word
    allwords = Counter()
    for item in dataset:
        for sentence in item['sentences']:
            allwords.update(sentence['tokens'])

    vocab = [k for k, v in allwords.items() if (v >= 3 and k not in ['best', 'Best', '!', 'ever', 'Dear', 'dear'])]
    vocab.insert(0, '#START#')
    vocab.append('#END#')

    word_to_index = {w: i for i, w in enumerate(vocab)}
    index_to_word = {i: w for i, w in enumerate(vocab)}

    logging.info('Size of vocabulary: {0}'.format(len(vocab)))

    # Initialize tokenizer
    tkn = TweetTokenizer()

    #sequence_lengths = [9, 13, 18, 23]  # Max sentence length from 6 to 20
    #batch_sizes = [50, 75, 100, 150, 200]
    #embedding_sizes = [512, 768, 1024]  # Missing 2048 if time allows
    #lrs = [0.001, 0.0001]
    #iterations = 25000
    sequence_lengths = [23]  # Max sentence length from 6 to 20
    batch_sizes = [200]
    embedding_sizes = [1024]  # Missing 2048 if time allows
    lrs = [0.001]
    iterations = 20000

    # Training
    for sequence_length in sequence_lengths:
        max_sentence_length = sequence_length - 3
        for batch_size in batch_sizes:
            for embedding_size in embedding_sizes:
                for lr in lrs:
                    logging.info(
                        'Seq_length: {}, Batch_size: {}, Embedding_size: {}, Learning_rate: {}'.format(sequence_length,
                                                                                                       batch_size,
                                                                                                       embedding_size,
                                                                                                       lr))
                    logging.info('Iteration\tLoss_train\tNorm_train\tLoss_val\tConf_Avg\tConf_Median\tBLEU-4')
                    f_train, f_val, f_pred, l_out = create_rnn_model(batch_size, sequence_length, embedding_size, lr)
                    for iteration in range(iterations):
                        x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(
                            get_data_batch(dataset, batch_size, max_sentence_length), sequence_length)
                        loss_train, norm = f_train(x_cnn, x_sentence, mask, y_sentence)
                        if not iteration % 250:
                            # logging.info('Iteration {} loss_train: {} norm: {}'.format(iteration, loss_train, norm))
                            try:
                                batch = get_data_batch(dataset, batch_size, max_sentence_length, split='val')
                                x_cnn, x_sentence, y_sentence, mask = prep_batch_for_network(batch, sequence_length)
                                loss_val = f_val(x_cnn, x_sentence, mask, y_sentence)
                                # logging.info('Val loss: {}'.format(loss_val))
                                predicted_captions, predicted_scores = predict(x_cnn, batch_size, sequence_length)
                                avg_score = np.average(predicted_scores)
                                med_score = np.median(predicted_scores)
                                # logging.info(predicted_captions)
                                bleu_score = compute_bleu(batch, predicted_captions)
                                logging.info(
                                    '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(iteration, loss_train, norm, loss_val, avg_score, med_score, bleu_score))

                            except IndexError:
                                continue

                    param_values = lasagne.layers.get_all_param_values(l_out)
                    d = {'param values': param_values,
                         'vocab': vocab,
                         'word_to_index': word_to_index,
                         'index_to_word': index_to_word,
                         }

                    # Save trained model
                    pickle.dump(d, open(
                        r'trained_lstm_{}_{}_{}_{}.pkl'.format(sequence_length, batch_size, embedding_size, lr),
                        'wb'), protocol=pickle.HIGHEST_PROTOCOL)
