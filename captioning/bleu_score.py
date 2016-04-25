"""Compute the BLEU-n score for the captioning results

Instead of averaging the sentence level BLEU scores (i.e. marco-average precision), 
the original BLEU metric (Papineni et al. 2002) accounts for the micro-average precision 
(i.e. summing the numerators and denominators for each hypothesis-reference(s) pairs before the division).

Input:
    JSON file with predicted captions and actual caption
    n: the ngram order

Output:
    BLEU-n score

It would be good to obtain BLEU-4 score but the NLTK library appears to not have this option at a corpus level.

"""
import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import TweetTokenizer

# Inputs
# input_file = r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\caption_results\coco_caption_results.json'
input_file = r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\caption_results\yelp_caption_results.json'
n = 4

# Initialize tokenizer
tkn = TweetTokenizer()

# Open JSON dataset
with open(input_file, mode='rb') as f:
    dataset = json.load(f)

# Initialize weights for n-gram input to NLTK
weights = []
for i in range(n):
    weights.append(float(1.0/n))

# Initialize hypothesis and reference arrays
hypotheses = []
references = []

# Create hypothesis and reference arrays taking 5 predicted captions per image (maybe we could modify this?)
for row in dataset['images']:
    caption = row['sentences'][0]['tokens']
    predicted_0 = row['predicted caption'][0]
    predicted_1 = row['predicted caption'][1]
    predicted_2 = row['predicted caption'][2]
    predicted_3 = row['predicted caption'][3]
    predicted_4 = row['predicted caption'][4]
    references.append([caption])
    references.append([caption])
    references.append([caption])
    references.append([caption])
    references.append([caption])
    hypotheses.append(tkn.tokenize(predicted_0))
    hypotheses.append(tkn.tokenize(predicted_1))
    hypotheses.append(tkn.tokenize(predicted_2))
    hypotheses.append(tkn.tokenize(predicted_3))
    hypotheses.append(tkn.tokenize(predicted_4))

# Compute BLEU score
bleu_score = corpus_bleu(references, hypotheses, weights=weights)

# Display BLEU score
print 'BLEU score: ' + str(bleu_score)