# -*- coding:utf-8 -*-
# This script takes as input the JSON file with the predicted labels/captions and stores the records in a sqlite db
import json
import sqlite3

#JSON_FILE = "C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\yelp_caption_results.json"
#DB_FILE = "C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\yelp_caption_results.db"
JSON_FILE = r"C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\results_84_62.json"
DB_FILE = r"C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\results_84_62.db"

# Load dataset and initialize DB
dataset = json.load(open(JSON_FILE))
conn = sqlite3.connect(DB_FILE)

# Connect to DB
c = conn.cursor()
# Create table
c.execute('''create table images
    (photo_id text primary key,
    yelp_id text,
    business_id text,
    file_path text,
    file_name text,
    split text,
    label text,
    predicted_label text,
    caption text,
    predicted_caption_1 text,
    predicted_caption_2 text,
    predicted_caption_3 text,
    predicted_caption_4 text,
    predicted_caption_5 text)''')

# Loop through all the records on the JSON file and add the data to a query that adds a row at a time to the DB
for row in dataset['images']:
    data = []
    photoid = row['photoid']
    data.append(photoid)
    yelpid = row['yelpid']
    data.append(yelpid)
    businessid = row['business_id']
    data.append(businessid)
    filepath = row['filepath']
    data.append(filepath)
    filename = row['filename']
    data.append(filename)
    split = row['split']
    data.append(split)
    label = row['label']
    data.append(label)
    predicted_label = row['predicted label']
    data.append(predicted_label)
    caption = row['sentences'][0]['raw']
    data.append(caption)
    predicted_0 = row['predicted caption'][0]
    data.append(predicted_0)
    predicted_1 = row['predicted caption'][1]
    data.append(predicted_1)
    predicted_2 = row['predicted caption'][2]
    data.append(predicted_2)
    predicted_3 = row['predicted caption'][3]
    data.append(predicted_3)
    predicted_4 = row['predicted caption'][4]
    data.append(predicted_4)

    # Insert row to DB
    c.execute('insert into images values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', data)

# Commit the changes to the DB
conn.commit()
c.close()
