# This script pickles the images taking as input the size and number of batches that will be dataset will be splitted into
import hickle as hkl
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps

PIXELS = [224, 299]
CLASSES = ['drink', 'food', 'inside', 'menu', 'outside']
LABELS = {cls: i for i, cls in enumerate(CLASSES)}

model_list = ['VGG_16', 'VGG_19', 'googlenet', 'inception_v3']


input_dir = r'C:\Users\crobe\Google Drive\DataMiningGroup\Datasets\restaurant_photos_with_labels_'
output_dir = r'D:\Yelp\datasets'

img_path = r'D:\Yelp\restaurant_photos\\'
id_label = 'photo_id'

batches = 24
input_files = []
output_files = []

# Initialize the list of input pickles
for k in range(1, batches):
    input_files.append(input_dir + 'train_batch' + str(k) + '.pkl')
input_files.append(input_dir + 'test.pkl')


def pickle_dataset(input_pkl, output_pkl, img_path, id_label, PIXELS):
    """
    This function takes the input pickle with text info and outputs a pickle with label and image only.
    """
    data = pd.read_pickle(input_pkl)
    dataset = {}

    # Initialize the arrays just for the first image
    iter_images = iter(data[id_label])
    first_image = next(iter_images)
    im = Image.open(img_path + first_image + '.jpg', 'r')
    # Resize image
    im = ImageOps.fit(im, (PIXELS, PIXELS), Image.ANTIALIAS)
    # Extract rgb channels
    im = (np.array(im))
    r = im[:, :, 0].flatten()
    g = im[:, :, 1].flatten()
    b = im[:, :, 2].flatten()

    # Store image in a numpy array
    img_list = np.array(list(r) + list(g) + list(b), dtype='uint8')
    img_list = img_list[np.newaxis, :]

    # Do the same for the rest of the images
    for img_name in iter_images:
        im = Image.open(img_path + img_name + '.jpg', 'r')
        im = ImageOps.fit(im, (PIXELS, PIXELS), Image.ANTIALIAS)
        im = (np.array(im))
        r = im[:, :, 0].flatten()
        g = im[:, :, 1].flatten()
        b = im[:, :, 2].flatten()

        img = np.array(list(r) + list(g) + list(b), dtype='uint8')
        # Stack vertically so that each row represents an image
        img_list = np.vstack((img_list, img[np.newaxis, :]))

    # Use hickle to compress the data (pickle didn't work for this data type
    hkl.dump(img_list, output_pkl + '_data.hpy', mode='w', compression='gzip')
    hkl.dump(data['label'], output_pkl + '_labels.hpy', mode='w')

    del img_list
    del data


# Pickle the datasets for different image sizes
for pix in PIXELS:
    output_files = []
    for k in range(1, batches):
        output_files.append(output_dir + '\\train_images_batch' + str(k) + '_' + str(pix))
    output_files.append(output_dir + '\\test_images_' + str(pix))
    for i, input_file in enumerate(input_files):
        print i
        pickle_dataset(input_file, output_files[i], img_path, id_label, pix)
