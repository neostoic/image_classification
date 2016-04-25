import cPickle as pickle
import json

import pandas as pd

# Load the caption data
photos = pd.read_pickle(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_captions.pkl')
dataset = json.load(
    open(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\image_caption_dataset_no_business.json'))
dataset_images = dataset['images']

for item in dataset_images:
    item['business_id'] = photos.query('photo_id == "' + item['yelpid'] + '"')['business_id'].iloc[0]

with open("C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\image_caption_dataset.json", "w") as outfile:
    dump_dict = dict()
    dump_dict['images'] = dataset_images
    json.dump(dump_dict, outfile)

with open(
        r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\caption_results\image_caption_with_predictions_coco.pkl',
        'rb') as f:
    dataset_images = pickle.load(f)

for item in dataset_images:
    item['business_id'] = photos.query('photo_id == "' + item['yelpid'] + '"')['business_id'].iloc[0]

with open("C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\coco_caption_results.json", "w") as outfile:
    dump_dict = dict()
    dump_dict['images'] = dataset_images
    json.dump(dump_dict, outfile)

with open(
        r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\caption_results\image_caption_with_predictions_yelp.pkl',
        'rb') as f:
    dataset_images = pickle.load(f)

for item in dataset_images:
    item['business_id'] = photos.query('photo_id == "' + item['yelpid'] + '"')['business_id'].iloc[0]

with open("C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\yelp_caption_results.json", "w") as outfile:
    dump_dict = dict()
    dump_dict['images'] = dataset_images
    json.dump(dump_dict, outfile)
