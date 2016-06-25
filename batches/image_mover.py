import pandas as pd
import shutil


def move_images(path_photos):
    photos = pd.read_pickle(r'./data/restaurant_photos.pkl')

    for idx, photo_id in enumerate(photos.photo_id):
        path_in = r'./data/restaurant_photos/' + photo_id + '.jpg'
        path_out = path_photos + photos.label[idx] + r'/' + photo_id + '.jpg'
        shutil.copy(path_in, path_out)
