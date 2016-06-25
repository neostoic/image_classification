# Utilities for LDA modeling
# Adapted from Radim Rehurek's tutorial on Topics and transformations with Gensim, available at:
# https://radimrehurek.com/gensim/tut2.html
from __future__ import print_function
import pickle
import pandas as pd
import itertools
import numpy as np
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from gensim.utils import simple_preprocess
import json
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

adjectives=['perfect','excellent','better','best','pretty','great','good','nice','delicious','little','friendly','amazing','new','bad','old','loud','high','right']
verbs=['like','love','think','wasnt','select','wait','come','came', 'ordered','got','know','order','said','sit','taste','closed','try','went','seated','cut','eat','going','asked','took','seating']
common=['food','day','people','place', 'meal','morning','view','restaurant','service','time','lunch','price','prices','way','room','minutes','plus','reviews','manager','staff']
odd=['le','la','las', 'est','ve','et','les','wasn','ayce','ll','wall','walls', 'home','yelp','www','http','com','oh','yes','kids']
more_stopwords=[];
more_stopwords.extend(adjectives)
more_stopwords.extend(verbs)
more_stopwords.extend(common)
more_stopwords.extend(odd)
more_stopwords.extend(STOPWORDS)

def tokenize(text):
    #logging.info('Tokenize in CreateBoWCaptions')
    #logging.info('Text:')
    #logging.info(text)
    return [token for token in simple_preprocess(text) if token not in more_stopwords]

def extract_reviews(dump_file):
    f = open(dump_file, 'rb')
    revs=[]
    logging.info('Extracting reviews')
    while 1:
        try:
            rev=pickle.load(f)
            #print( rev )
            revs.append(rev)
        except EOFError:
            break
    return revs


def iter_rev(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    logging.info ("inside iter_rev")
    revs=[]
    revs= extract_reviews(dump_file);
    #print ("Reviews: %s"%revs)
    for text in revs:
        #print "Unfiltered: %s"%text
        #text = filter_wiki(text);
        #print "filtered: %s"%text
        tokens = tokenize(text);
        #logging.info (tokens)
        #if len(tokens) < 10 :#or any(title.startswith(ns + ':') for ns in ignore_namespaces):
        #   continue  # ignore short articles and various meta-articles
        yield tokens


def createRestaurantDictionary(pklFileName,restaurantName):
    documents=extract_reviews('../data/'+pklFileName+'.pkl');
    restaurantName=restaurantName.lower().split();
    city=['vegas']
    more_stopwords.extend(restaurantName)
    more_stopwords.extend(city)
    #print("Texts before STOPWORDS: ",documents)
    texts=[]
    texts = [tokenize(document) for document in documents];
    print("Texts after STOPWORDS: ",texts);
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
           frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts];
    dictionaryW = corpora.Dictionary(texts);
    dictionaryW.save('../data/'+pklFileName+'.dict') # store the dictionary, for future reference
    return dictionaryW


def createBowVector(document):
    logging.info('Transform text into the bag-of-words space');
    logging.info('Original Text: %s'%document)
    bow_vector = id2word.doc2bow(tokenize(document))
    logging.info('Bow vector Text: %s'%bow_vector)
    logging.info([(id2word[id], count) for id, count in bow_vector])
    return bow_vector

def transform2lda(bow_vector):
    logging.info('transform into LDA space');
    lda_vector = lda_model[bow_vector]
    return lda_vector

def getTopTopic(document):
    lda_vector=transform2lda(createBowVector(document))
    logging.info("print the document's single most prominent LDA topic");
    print(lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))

def intra_inter(model, test_docs, num_pairs=10000):
    logging.info("Evaluating Model\nTest docs:")
    logging.info(test_docs)
    # split each test document into two halves and compute topics for each half
    part1 = [model[model.id2word.doc2bow(tokens[: len(tokens)// 2])] for tokens in test_docs];
    part2 = [model[model.id2word.doc2bow(tokens[len(tokens)// 2:])] for tokens in test_docs];

    # print computed similarities (uses cossim)
    print("average cosine similarity between corresponding parts (higher is better):")
    print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))

    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    print("average cosine similarity between 10,000 random parts (lower is better):")
    print(np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs]))

def getTopicNumAndMatchWords(document):
    lda_vector=transform2lda(createBowVector(document));
    #lda_vector = lda_model.get_document_topics(id2word.doc2bow(document))
    # print(lda_vector)
    top_topic = max(lda_vector, key=lambda item: item[1])
    logging.info(top_topic)
    top_topic_words = lda_model.show_topic(top_topic[0])
    # print(top_topic_words)
    topic_document_words = []
    # logging.info('Document')
    # logging.info(document)
    #for word in document:
    for word in tokenize(document):
        for topic_word in top_topic_words:
            # logging.info('word in document')
            # logging.info(word)
            # logging.info('topic_word in document')
            # logging.info(topic_word)
            if (word == topic_word[0]):
                # print('Word in topic:')
                # print(word)
                topic_document_words.extend([word])
    return top_topic[0], top_topic[1], topic_document_words;

################CREATING Captions Dictionary
#fileName='MonAmiGabiCapsTraining';
#restaurantName='Mon Ami Gabi';
#dictionaryR=createRestaurantDictionary(fileName,restaurantName)
#dictionaryR = corpora.Dictionary.load('../data/'+fileName+'.dict')
#for idx in dictionaryR:
#    print(dictionaryR[idx])

import pandas as pd
from gensim import corpora, models
################Loading LDA model
full='../data/wiki_results/reviews';
# MonAmiCaps='./MonAmiGabiCaps'
MonAmiFull='../data/MonAmiGabi'
lda_model=models.LdaModel.load(MonAmiFull+'TrainingLDAtopicModel.mm',mmap='r')
lda_model.print_topics(20);
logging.info('Assigning id2word')
id2word=lda_model.id2word;

reviews=iter_rev('../data/MonAmiGabiTesting.pkl');
# print(review for review in reviews);
# review=reviews[1];
# getTopTopic(review)
# print("LDA review results:")
# print(reviews)

logging.info('Evaluating LDA reviews')
rev_stream = (tokens for tokens in iter_rev('../data/MonAmiGabiTesting.pkl'));  # generator
test_reviews = list(itertools.islice(rev_stream,1,5000));
intra_inter(lda_model, test_reviews)
logging.info('Evaluating LDA captions')
caption_stream = (tokens for tokens in iter_rev('../data/MonAmiGabiCapsTesting.pkl'));  # generator
test_captions = list(itertools.islice(caption_stream,5000));
#intra_inter(lda_model, test_captions)
totalRevs=len(reviews);
print(totalRevs)

#caption=captions[1];
#getTopTopic(caption)
#intra_inter(lda_model, captions)
#print("LDA caption results:")
#print(captions)

#logging.info('Opening photos')
#photos = pd.read_pickle('../data/restaurant_photos.pkl');

# logging.info('Opening business')
# business = pd.read_pickle('../data/restaurant_business.pkl');
# count =1;
# restaurantName="Mon Ami Gabi"
# restaurantId='4bEjOyTaDG24SY5TxsaUNQ';
# print(restaurantId)
#revs=reviews['text'];
#captions=photos['caption']

#print(reviews['business_id'][count])

#print(business.query('business_id == "' + restaurantName + '"')['name'])
#print(revs[count])
#print(business['name'])
#print(photos['business_id'][count])

# Load the caption data
# photos = pd.read_pickle('../data/mon_ami_gabi_photos.pkl')
# dataset = json.load(
#    open('../data/image_caption_dataset_no_business.json'))
#
# dataset_images = dataset['images']
# logging.info('Getting business id and topics')
# for item in dataset_images:
#    item['business_id'] = photos.query('photo_id == "' + item['yelpid'] + '"')['business_id'].iloc[0]
#    caption=(item['sentences'])[0]['raw'];
#    print(caption)
#    item['topic_info']=getTopicNumAndMatchWords(caption)
#    logging.info(item['topic_info'])
#
# logging.info('Saving image_caption_dataset')
# with open("../data/image_caption_dataset.json", "w") as outfile:
#     dump_dict = dict()
#     dump_dict['images'] = dataset_images
#     json.dump(dump_dict, outfile)

# logging.info('Opening business')
# business = pd.read_pickle('../data/restaurant_business.pkl');
# count =1;
# restaurantName="Mon Ami Gabi"
# restaurantId='4bEjOyTaDG24SY5TxsaUNQ'
# print(restaurantId)

# logging.info('Loading image_caption_dataset')
# dataset = json.load(
#    open('../data/image_caption_dataset.json'))
#
# dataset_images = dataset['images']
#
# logging.info('Getting all photos for Mon Ami Gabi')
# dataset = []
# for item in dataset_images:
#     if item['business_id'] in [restaurantId]:
#         caption = (item['sentences'])[0]['raw'];
#         item['topic_info']=getTopicNumAndMatchWords(caption)
#         logging.info(item['topic_info'])
#         dataset.append(item)
#
# print(dataset)
# logging.info('Saving mon_ami_image_dataset')
# with open("../data/mon_ami_image_dataset.json", "w") as outfile:
#     dump_dict = dict()
#     dump_dict['images'] = dataset
#     json.dump(dump_dict, outfile)


###################Creating dataset : images for each word in topic
# logging.info('Loading image_caption_dataset')
# dataset_mon_ami_img = json.load(
#    open('../data/mon_ami_image_dataset.json'))
#
# dataset_mon_ami_img=dataset_mon_ami_img['images']
# dataset=[]
# recommended_photos={}
# logging.info('Opening reviews')
# reviews = extract_reviews('../data/MonAmiGabiTesting.pkl');
# for i in range(1,5):
#     topic = lda_model.show_topic(i)
#     # print(topic)
#     recommended_photos['text']=reviews[i]
#     # recommended_photos['review_topic_info']=topic
#     recommended_photos['review_topic_info']=getTopicNumAndMatchWords(reviews[i])
#     # print(recommended_photos['review_topic_info'])
#     logging.info(recommended_photos['review_topic_info'][2])
#     recommended_photos['images_info'] = []
#     for image in dataset_mon_ami_img:
#         pho_topic=image['topic_info'];
#         logging.info(recommended_photos['review_topic_info'][0])
#         logging.info(pho_topic[0])
#
#         logging.info('Checking topic words')
#         logging.info(pho_topic[2])
#         logging.info(recommended_photos['review_topic_info'][2])
#         for pho_topic_word in pho_topic[2]:
#             for rev_topic_word in recommended_photos['review_topic_info'][2]:
#                 logging.info(pho_topic_word)
#                 logging.info(rev_topic_word)
#                 if(rev_topic_word == pho_topic_word):
#                    try:
#                        recommended_photos['images_info'].extend([image['photoid']])
#                    except KeyError:
#                         recommended_photos['images_info']=[]
#                         recommended_photos['images_info'].extend([image['photoid']])
#         logging.info(recommended_photos['images_info'])
#     dataset.append(recommended_photos)
#     recommended_photos={}
#
# logging.info('Saving mon_ami_recomended_image_dataset')
# with open("../data/mon_ami_recomended_image_dataset2.json", "w") as outfile:
#     dump_dict = dict()
#     dump_dict['images'] = dataset
#     json.dump(dump_dict, outfile)

logging.info('Loading image_caption_dataset')
dataset_mon_ami_img = json.load(
   open("../data/mon_ami_recomended_image_dataset2.json"))

dataset=dataset_mon_ami_img['images']

for review in dataset:
    print("Review:")
    print(review['text'])
    print(review['review_topic_info'][1])
    print("Topic words:")
    print(review['review_topic_info'][2])
    print("Photos:")
    print(review['images_info'])
    # for topic_word in review['review_topic_info']:
    #     try:
    #         print(topic_word)
    #         print(topic_word[0])
    #         print(review[topic_word[0]])
    #     except KeyError:
    #         print();
