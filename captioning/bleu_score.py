# Compute the BLEU-n score for our corpus
# Based on the corpus-level BLEU function from: http://www.nltk.org/_modules/nltk/translate/bleu_score.html
# Instead of averaging the sentence level BLEU scores (i.e. macro-average precision),
# the original BLEU metric (Papineni et al. 2002) accounts for the micro-average precision
# (i.e. summing the numerators and denominators for each hypothesis-reference(s) pairs before the division).
#
# Input:
#     JSON file with predicted captions and actual caption
#     n: the ngram order
#
# Output:
#     BLEU-n score
#


import json

from nltk.tokenize import TweetTokenizer
from nltk.translate.bleu_score import corpus_bleu


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

# Inputs
input_file = r'results_yelp_webpage.json'
#n = 1

# Initialize tokenizer
tkn = TweetTokenizer()

# Open JSON dataset
with open(input_file, mode='rb') as f:
    dataset = json.load(f)

confidence = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#confidence = [0.2]

for n in [1, 2, 3, 4]:
    # Initialize weights for n-gram input to NLTK
    #weights = np.zeros(n)
    #weights[n-1] = 1
    weights = []
    for i in range(n):
        weights.append(float(1.0 / n))

    # Create hypothesis and reference arrays taking 5 predicted captions per image (maybe we could modify this?)
    for conf_value in confidence:
        samples = 0
        # Initialize hypothesis and reference arrays
        hypotheses = []
        references = []
        for row in dataset['images']:
            caption = row['sentences'][0]['tokens']
            #for idx in range(len(row['predicted caption'])):
            #    if float(row['confidence'][idx]) >= conf_value and row['split'] in ['test'] and row['predicted caption'][idx]:
            #        samples += 1
            #        predicted = row['predicted caption'][idx]
            #        references.append([caption])
            #        hypotheses.append(tkn.tokenize(predicted))
            #if float(row['top confidence']) >= conf_value and row['split'] in ['test'] and row['top caption']:
            if float(row['top confidence']) >= conf_value and row['top caption']:
                samples += 1
                predicted = row['top caption']
                references.append([caption])
                hypotheses.append(tkn.tokenize(predicted))

        # Compute BLEU score
        bleu_score = corpus_bleu(references, hypotheses, weights=weights)

        # Display BLEU score
        print 'Confidence: {0} BLEU-{1} score: {2} Samples: {3}'.format(conf_value, n, bleu_score, samples)