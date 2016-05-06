# Helper script to add the business ID to the JSON files since this was not done initially
# This is required to reference restaurants/photos
import cPickle as pickle
import json

import pandas as pd

# Load the image data
photos = pd.read_pickle(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_captions.pkl')

# Load the image dataset where the business_id should be added
dataset = json.load(
    open(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\image_caption_dataset_no_business.json'))
dataset_images = dataset['images']

# Use the query to find the record with the same photo id and add the business id to the other dataset
for item in dataset_images:
    item['business_id'] = photos.query('photo_id == "' + item['yelpid'] + '"')['business_id'].iloc[0]

# Store the new dataset that includes the business_id
with open("C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\image_caption_dataset.json", "w") as outfile:
    # We use dump_dict to create the proper JSON structure
    dump_dict = dict()
    dump_dict['images'] = dataset_images
    json.dump(dump_dict, outfile)

# Load next dataset where the business id should be added
with open(
        r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\caption_results\image_caption_with_predictions_coco.pkl',
        'rb') as f:
    dataset_images = pickle.load(f)

# Use query to find the record with the same photo id and add the business id to the other dataset
for item in dataset_images:
    item['business_id'] = photos.query('photo_id == "' + item['yelpid'] + '"')['business_id'].iloc[0]

# Store the dataset
with open("C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\coco_caption_results.json", "w") as outfile:
    dump_dict = dict()
    dump_dict['images'] = dataset_images
    json.dump(dump_dict, outfile)

# Store the dataset
with open("C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\results_84_62.json", "w") as outfile:
    dump_dict = dict()
    dump_dict['images'] = dataset
    json.dump(dump_dict, outfile)

# Open dataset
with open(
        r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\caption_results\image_caption_with_predictions_yelp.pkl',
        'rb') as f:
    dataset_images = pickle.load(f)

# Use query to find the record with the same photo id and add the business id to the other dataset
for item in dataset_images:
    item['business_id'] = photos.query('photo_id == "' + item['yelpid'] + '"')['business_id'].iloc[0]

# Store the dataset
with open("C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\yelp_caption_results.json", "w") as outfile:
    dump_dict = dict()
    dump_dict['images'] = dataset_images
    json.dump(dump_dict, outfile)
