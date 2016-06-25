# This script changes the JSON format so that it is compatible with the captioning algorithm,
# changing it to a dictionary.
# The format is very similar to the one used by MS COCO dataset. It also tokenizes each caption and splits the images
# based on which dataset they belong to: train/test/validation
import json
import shutil

import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.cross_validation import StratifiedKFold


def preprocess_db():
    tkn = TweetTokenizer()
    photos = pd.read_pickle(r'./data/restaurant_photos_with_labels.pkl')
    img_path = r'./data/restaurant_photos/'
    sentid = 1
    img_list = []

    # Split data in such a way that labels are evenly distributed between 6 folds
    skf = StratifiedKFold(photos['label'], n_folds=6)

    folds = []
    # Initialize all images to train dataset initially
    photos['split'] = ['train' for i in range(len(photos))]

    # Obtain the indices for the test and validation splits and change value appropriately
    for _, test_ix in skf:
        folds.append(test_ix)
    photos.split[folds[0]] = 'test'
    photos.split[folds[1]] = 'val'

    # Obtain the information from each picture and move the pictures to the appropriate dir. The images are renamed.
    for i, photo_id in enumerate(photos.photo_id):
        img_dict = dict()
        img_dict['sentids'] = [sentid]
        img_dict['business_id'] = photo_id.business_id[i]
        if photos.split[i] in ['train']:
            img_dict['filepath'] = u'train'
            img_dict['imgid'] = 0
            img_dict['split'] = u'train'
            shutil.copy(img_path + photo_id + '.jpg', './data/restaurant_photos_split/train/' + str(sentid).zfill(6) + '.jpg')
        elif photos.split[i] in ['test']:
            img_dict['filepath'] = u'test'
            img_dict['imgid'] = 0
            img_dict['split'] = u'test'
            shutil.copy(img_path + photo_id + '.jpg', './data/restaurant_photos_split/test/' + str(sentid).zfill(6) + '.jpg')
        else:
            img_dict['filepath'] = u'val'
            img_dict['imgid'] = 0
            img_dict['split'] = u'val'
            shutil.copy(img_path + photo_id + '.jpg', './data/restaurant_photos_split/val/' + str(sentid).zfill(6) + '.jpg')
        img_dict['label'] = photos.label[i]
        caption_dict = dict()
        if photos.caption[i]:
            # Tokenize the captions
            caption_dict['tokens'] = tkn.tokenize(photos.caption[i])
            caption_dict['raw'] = photos.caption[i]
        else:
            caption_dict['tokens'] = 'None'
            caption_dict['raw'] = 'None'
        caption_dict['imgid'] = 0
        caption_dict['sentid'] = sentid
        img_dict['sentences'] = [caption_dict]
        img_dict['photoid'] = sentid
        img_dict['yelpid'] = photo_id
        img_list.append(img_dict)
        sentid += 1

    # Store the new dataset as a JSON file
    with open("./data/image_caption_dataset.json", "w") as outfile:
        json.dump(img_list, outfile)
