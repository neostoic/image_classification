import pandas as pd
import shutil
import os


def move_images(path_photos):
    if not os.path.exists(r'./data/restaurant_photos/'):
        os.makedirs(r'./data/restaurant_photos/')
    photos = pd.read_pickle(r'./data/restaurant_photos.pkl')

    for idx, photo_id in enumerate(photos.photo_id):
        path_out = r'./data/restaurant_photos/' + photo_id + '.jpg'
        path_in = path_photos + r'/' + photo_id + '.jpg'
        shutil.copy(path_in, path_out)
