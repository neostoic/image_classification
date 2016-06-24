from PIL import Image
from scipy import misc
import numpy as np
import pandas as pd
import gc

train_data = pd.read_pickle(r'../data/restaurant_photos_with_labels_train.pkl')
test_data = pd.read_pickle(r'../data/restaurant_photos_with_labels_test.pkl')
train_list = []
i = 0
print 'Read training data'
for image_name in train_data['photo_id']:
    image = Image.open(r'../data/restaurant_photos/' + image_name + '.jpg', 'r')
    #image_array = np.asarray(image, dtype='float32')
    #image_array = np.asarray(image)
    #image_array= np.rollaxis(image_array, 2, 0)
    image_array = np.asarray(image)
    train_list.append(image_array)
    del image
    gc.collect()
train_data['image'] = train_list
train_data.to_pickle(r'../data/train_labels_.pkl')
print 'Pickled training data, read test data'
test_list = []
for image_name in test_data['photo_id']:
    image = Image.open(r'../data/restaurant_photos/' + image_name + '.jpg', 'r')
    image_array = np.array(image)
    # image_array= np.rollaxis(image_array, 2, 0)
    del image
    test_list.append(image_array)
    gc.collect()
test_data['image'] = test_list
test_data.to_pickle(r'../data/test_labels_.pkl')
