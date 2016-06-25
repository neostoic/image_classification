from multiprocessing import freeze_support
from batches.image_mover import move_images
from batches.pickle_train_test_data import create_train_test_data
from convnets.convnet import train
from captioning.captioning import captioning
from captioning.add_predicted_labels import get_labels_features
from captioning.preprocess_db import preprocess_db
from captioning.train_rnn import train_rnn

MODELS = ['VGG_16', 'VGG_19', 'googlenet', 'inception_v3']
#home/
#D:
#C:
# Paths that we must take into account!
#categories.pkl
#cnn_features = r'./data/image_caption_with_cnn_features.pkl'
#'results_yelp_webpage.json'
#'captions_yelp_nic_results_webpage.json'
# Check add business id

if __name__ == '__main__':
    freeze_support()
    # HARDCODED: CHANGE THIS TO THE PATH WHERE THE PHOTOS ARE DOWNLOADED:
    path_photos = r'D:/Yelp/Photos/'
    move_images(path_photos)
    create_train_test_data(folds=6)
    train('googlenet')
    preprocess_db()
    get_labels_features()
    train_rnn()
    captioning()

