# -*- coding:utf-8 -*-
# This script takes as input the JSON file with the predicted labels/captions and stores the records in a sqlite db
# Adapted from a gist from Nobuya Oshiro, taken from: https://gist.github.com/atsuya046/7957165
import json
import sqlite3

JSON_FILE = r"results_yelp_webpage.json"
DB_FILE = r"C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\results_yelp_webpage.db"

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
    caption_confidence_1 text,
    predicted_caption_2 text,
    caption_confidence_2 text,
    predicted_caption_3 text,
    caption_confidence_3 text,
    predicted_caption_4 text,
    caption_confidence_4 text,
    predicted_caption_5 text,
    caption_confidence_5 text
    )''')

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
    try:
        predicted_label = row['predicted label']
    except IndexError:
        predicted_label = 'not predicted'
    data.append(predicted_label)
    caption = row['sentences'][0]['raw']
    data.append(caption)
    try:
        predicted_0 = row['predicted caption'][0]
    except IndexError:
        predicted_0 = ''
    data.append(predicted_0)
    try:
        confidence_0 = str(row['confidence'][0])
    except IndexError:
        confidence_0 = ''
    data.append(confidence_0)
    try:
        predicted_1 = row['predicted caption'][1]
    except IndexError:
        predicted_1 = ''
    data.append(predicted_1)
    try:
        confidence_1 = str(row['confidence'][1])
    except IndexError:
        confidence_1 = ''
    data.append(confidence_1)
    try:
        predicted_2 = row['predicted caption'][2]
    except IndexError:
        predicted_2 = ''
    data.append(predicted_2)
    try:
        confidence_2 = str(row['confidence'][2])
    except IndexError:
        confidence_2 = ''
    data.append(confidence_2)
    try:
        predicted_3 = row['predicted caption'][3]
    except IndexError:
        predicted_3 = ''
    data.append(predicted_3)
    try:
        confidence_3 = str(row['confidence'][3])
    except IndexError:
        confidence_3 = ''
    data.append(confidence_3)
    try:
        predicted_4 = row['predicted caption'][4]
    except IndexError:
        predicted_4 = ''
    data.append(predicted_4)
    try:
        confidence_4 = str(row['confidence'][4])
    except IndexError:
        confidence_4 = ''
    data.append(confidence_4)
    # Insert row to DB
    c.execute('insert into images values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', data)

# Commit the changes to the DB
conn.commit()
c.close()
