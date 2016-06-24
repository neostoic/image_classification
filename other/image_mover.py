import pandas as pd
import shutil

photos = pd.read_pickle(r'D:\Yelp\restaurant_photos.pkl')
# business = pd.read_pickle(r'D:\Yelp\restaurant_business.pickle')
# reviews = pd.read_pickle(r'D:\Yelp\restaurant_reviews.pickle')

for idx, photo_id in enumerate(photos.photo_id):
    path_in = r'D:\Yelp\restaurant_photos\\' + photo_id + '.jpg'
    path_out = r'D:\Yelp\photos\\' + photos.label[idx] + r'\\' + photo_id + '.jpg'
    shutil.copy(path_in, path_out)
