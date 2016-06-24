# Create LDA model for Mon Ami Gabi reviews.
# Taken from Radim Rehurek's tutorial for topic modeling with Gensim
# Available at:
# http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html
from __future__ import print_function
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from gensim.utils import simple_preprocess
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from CreateBoWCaptions import createRestaurantDictionary,extract_reviews

fileName='MonAmiGabiTraining';
captionsPklFileName='./data/MonAmiGabiCapsTraining.pkl';
restaurantName='Mon Ami Gabi';
method = "captionWordsOnly";
#method= "stopwords"
documents=extract_reviews('./data/'+fileName+'.pkl');

#print("Texts before STOPWORDS: ",documents)
#Words without STOPWORDS found in the corresponding captions
#Using the dictionary processed with the corresponding captions
#The text from the captions gets the STOPWORDS removed through the createCaptionDictionary
try:
    captionsDict=corpora.Dictionary.load('./data/MonAmiGabiCapsTraining.dict');
except FileNotFoundError:
    createRestaurantDictionary(captionsPklFileName, restaurantName);

wordsInCaptions=[]
for idx in captionsDict:
    wordsInCaptions.extend([captionsDict[idx]])
    print('Added: '+captionsDict[idx])

removeWords=[]
texts=[]
if(method=="captionWordsOnly"):
    removeWords=wordsInCaptions;
else:
    removeWords=STOPWORDS;

texts = [[word for word in simple_preprocess(document) if word in removeWords] for document in documents];
print("Texts after STOPWORDS: ",texts);
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
       frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts];

#Dictionary for the reviews
dictionary = corpora.Dictionary(texts);
dictionary.save('./data/'+fileName+'.dict') # store the dictionary, for future reference
logging.info('Dictionary saved');

corpus = [dictionary.doc2bow(text) for text in texts];
corpora.MmCorpus.serialize('./data/'+fileName+'.mm', corpus) # store to disk, for later use
logging.info('Corpus saved')
#print(corpus)

# set the number of topics to look for
num = 10

print("Creating LDA") #Not really good online unless topics don't change too much
lda = gensim.models.LdaModel(corpus,id2word=dictionary,num_topics=num, passes=15)
lda.print_topics(num)

lda.save(fileName+'LDAtopicModel.mm');