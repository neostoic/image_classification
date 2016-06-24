# -*- coding:utf-8 -*-
# This script takes as input the JSON file with the predicted labels/captions and stores the records in a sqlite db
# Adapted from Radim Rehurek's tutorial on Topics and transformations with Gensim, available at:
# https://radimrehurek.com/gensim/tut2.html
import json
import logging
import pickle
import shutil
import sqlite3

from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

adjectives = ['perfect', 'excellent', 'better', 'best', 'pretty', 'great', 'good', 'nice', 'delicious', 'little',
              'friendly', 'amazing', 'new', 'bad', 'old', 'loud', 'high', 'right']
verbs = ['like', 'love', 'think', 'wasnt', 'select', 'wait', 'come', 'came', 'ordered', 'got', 'know', 'order', 'said',
         'sit', 'taste', 'closed', 'try', 'went', 'seated', 'cut', 'eat', 'going', 'asked', 'took', 'seating']
common = ['food', 'day', 'people', 'place', 'meal', 'morning', 'view', 'restaurant', 'service', 'time', 'lunch',
          'price', 'prices', 'way', 'room', 'minutes', 'plus', 'reviews', 'manager', 'staff']
odd = ['le', 'la', 'las', 'est', 've', 'et', 'les', 'wasn', 'ayce', 'll', 'wall', 'walls', 'home', 'yelp', 'www',
       'http', 'com', 'oh', 'yes', 'kids']
more_stopwords = [];
more_stopwords.extend(adjectives)
more_stopwords.extend(verbs)
more_stopwords.extend(common)
more_stopwords.extend(odd)
more_stopwords.extend(STOPWORDS)


def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in more_stopwords]


def extract_reviews(dump_file):
    f = open(dump_file, 'rb')
    revs = []
    logging.info('Extracting reviews')
    while 1:
        try:
            rev = pickle.load(f)
            # print( rev )
            revs.append(rev)
        except EOFError:
            break
    return revs


def iter_rev(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    revs = extract_reviews(dump_file);
    for text in revs:
        tokens = tokenize(text);
        yield tokens


def createRestaurantDictionary(pklFileName, restaurantName):
    documents = extract_reviews('./data/' + pklFileName + '.pkl');
    restaurantName = restaurantName.lower().split();
    city = ['vegas']
    more_stopwords.extend(restaurantName)
    more_stopwords.extend(city)
    # print("Texts before STOPWORDS: ",documents)
    texts = []
    texts = [tokenize(document) for document in documents];
    print("Texts after STOPWORDS: ", texts);
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts];
    dictionaryW = corpora.Dictionary(texts);
    dictionaryW.save('./data/' + pklFileName + '.dict')  # store the dictionary, for future reference
    return dictionaryW


def createBowVector(document):
    bow_vector = id2word.doc2bow(tokenize(document))
    return bow_vector


def transform2lda(bow_vector):
    lda_vector = lda_model[bow_vector]
    return lda_vector


def getTopicNumAndMatchWords(document):
    lda_vector = transform2lda(createBowVector(document));
    top_topic = max(lda_vector, key=lambda item: item[1])
    top_topic_words = lda_model.show_topic(top_topic[0], topn=20)
    topic_document_words = []
    for word in tokenize(document):
        for topic_word in top_topic_words:
            if (word == topic_word[0]):
                topic_document_words.extend([word])
    return top_topic[0], top_topic[1], topic_document_words;


def copy_file(src, dest):
    try:
        shutil.copy(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)
    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


def get_top_topics():
    mon_ami_gabi_captions = json.load(open(MONAMIGABI_PHOTOS))
    mon_ami_gabi_reviews = json.load(open(MONAMIGABI_REVIEWS))

    for line in mon_ami_gabi_captions:
        line['topic'], line['weight'], line['top_words'] = getTopicNumAndMatchWords(line['caption'])
    json.dump(mon_ami_gabi_captions, open(images_topics, mode='w'))

    for line in mon_ami_gabi_reviews:
        line['topic'], line['weight'], top_words = getTopicNumAndMatchWords(line['text'])
        line['top_words'] = list(set(top_words))
    json.dump(mon_ami_gabi_reviews, open(reviews_topics, mode='w'))


def match_suggestions_to_captions():
    mon_ami_gabi_captions = json.load(open(images_topics))
    mon_ami_gabi_reviews = json.load(open(reviews_topics))

    for line in mon_ami_gabi_reviews:
        suggested_images = []
        if line['top_words']:
            for review_topword in line['top_words']:
                for caption in mon_ami_gabi_captions:
                    if caption['topic'] == line['topic']:
                        if review_topword in caption['top_words']:
                            suggested_images.append(caption['photo_id'])
                            if len(suggested_images) == 5:
                                break
                if len(suggested_images) == 5:
                    break
            if len(suggested_images) < 5:
                top_topic_words_tuple = lda_model.show_topic(line['topic'], topn=20)
                top_topic_words = [topic_word[0] for topic_word in top_topic_words_tuple]
                for top_word in line['top_words']:
                    top_topic_words.remove(top_word)
                for top_topic_word in top_topic_words:
                    for caption in mon_ami_gabi_captions:
                        if caption['topic'] == line['topic']:
                            if top_topic_word in caption['top_words']:
                                suggested_images.append(caption['photo_id'])
                                if len(suggested_images) == 5:
                                    break
                    if len(suggested_images) == 5:
                        break
        line['suggested_images'] = suggested_images
    json.dump(mon_ami_gabi_reviews, open(reviews_suggestions, mode='w'))


def build_database():
    # Load dataset and initialize DB
    dataset = json.load(open(reviews_suggestions))
    conn = sqlite3.connect(DB_FILE)

    # Suggested images
    # review_text
    # suggested_image1 ... suggested_image5
    # y review_top_words
    # Connect to DB
    c = conn.cursor()
    # Create table
    c.execute('''create table reviews
        (review_id text primary key,
        business_id text,
        review_text text,
        top_words text,
        suggested_image1 text,
        suggested_image2 text,
        suggested_image3 text,
        suggested_image4 text,
        suggested_image5 text,
        topic text
        )''')

    # {"review_id": "UpKdXO3jEElnqtsWjYxQ2w", "weight": 0.10000000000000002,
    #  "text": "Excellent food, great atmosphere, a bit noisy.  $$", "business_id": "4bEjOyTaDG24SY5TxsaUNQ",
    #  "top_words": [], "topic": 0, "suggested_images": []}

    # Loop through all the records on the JSON file and add the data to a query that adds a row at a time to the DB
    for row in dataset:
        if len(row['suggested_images']):
            data = []
            review_id = row['review_id']
            data.append(review_id)
            business_id = row['business_id']
            data.append(business_id)
            review_text = row['text']
            data.append(review_text)
            top_words = ', '.join(row['top_words'])
            data.append(top_words)
            try:
                suggested_image1 = row['suggested_images'][0]
            except IndexError:
                suggested_image1 = ''
            try:
                suggested_image2 = row['suggested_images'][1]
            except IndexError:
                suggested_image2 = ''
            try:
                suggested_image3 = row['suggested_images'][2]
            except IndexError:
                suggested_image3 = ''
            try:
                suggested_image4 = row['suggested_images'][3]
            except IndexError:
                suggested_image4 = ''
            try:
                suggested_image5 = row['suggested_images'][4]
            except IndexError:
                suggested_image5 = ''
            data.append(suggested_image1)
            data.append(suggested_image2)
            data.append(suggested_image3)
            data.append(suggested_image4)
            data.append(suggested_image5)
            topic = row['topic']
            data.append(topic)

            # Insert row to DB
            c.execute('insert into reviews values (?,?,?,?,?,?,?,?,?,?)', data)

    # Commit the changes to the DB
    conn.commit()
    c.close()


MONAMIGABI_PHOTOS = r"mon_ami_gabi_photos.json"
images_topics = r"mon_ami_gabi_photos_topics.json"
reviews_topics = r"mon_ami_gabi_reviews_topics.json"
reviews_suggestions = r"mon_ami_gabi_reviews_suggestions_2.json"
MONAMIGABI_REVIEWS = r"mon_ami_gabi_reviews.json"
PHOTOS_PATH = r'D:\Yelp\restaurant_photos/'
IMAGE_PATH = r'C:\Users\crobe\Google Drive\DataMiningGroup\Code\review_mapping\images'
# Load dataset and initialize DB
DB_FILE = r"C:\Users\crobe\Google Drive\DataMiningGroup\suggestions_monamigabi_2.db"
lda_model = models.LdaModel.load(r'MonAmiGabiTrainingLDAtopicModel.mm',
                                 mmap='r')
lda_model.print_topics(20);
logging.info('Assigning id2word')
id2word = lda_model.id2word;

get_top_topics()
match_suggestions_to_captions()
build_database()