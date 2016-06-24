from PIL import Image
from scipy import misc
import numpy as np
import pandas as pd
import gc

train_data = pd.read_pickle(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_labels_train.pkl')
test_data = pd.read_pickle(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_labels_test.pkl')
train_list = []
i = 0
print 'Read training data'
for image_name in train_data['photo_id']:
    img = Image.open(r'D:\Yelp\restaurant_photos\\' + image_name + '.jpg', 'r')
    img = np.asarray(img, dtype = 'float32') / 255.
    img = img.transpose(2,0,1)
    train_list.append(img)
    gc.collect()
train_data['image'] = train_list
train_data.to_pickle(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\train_labels_2.pkl')
print 'Pickled training data, read test data'
test_list = []
for image_name in test_data['photo_id']:
    img = Image.open(r'D:\Yelp\restaurant_photos\\' + image_name + '.jpg', 'r')
    img = np.asarray(img, dtype = 'float32') / 255.
    img = img.transpose(2,0,1)
    test_list.append(img)
    gc.collect()
test_data['image'] = test_list
test_data.to_pickle(r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\test_labels_2.pkl')
