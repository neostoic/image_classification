import pandas as pd
import shutil
photos = pd.read_pickle(r'D:\Yelp\restaurant_photos.pickle')
business = pd.read_pickle(r'D:\Yelp\restaurant_business.pickle')
reviews = pd.read_pickle(r'D:\Yelp\restaurant_reviews.pickle')

for id in photos.photo_id:
    path_in = 'D:\\Yelp\\2016_yelp_dataset_challenge_photos\\' + id + '.jpg'
    path_out = 'D:\\Yelp\\restaurant_photos\\' + id + '.jpg'
    shutil.copy (path_in, path_out)
	