from multiprocessing import freeze_support
from batches.pickle_train_test_data import create_train_test_data
from convnets.convnet import train
from captioning.add_predicted_labels import get_labels_features

MODELS = ['VGG_16', 'VGG_19', 'googlenet', 'inception_v3']

label_pickle = r'../data/restaurant_photos_with_labels.pkl'

# Paths that we must take into account!
img_path = r'../data/restaurant_photos/'
CLASSES = pickle.load(open(r'../data/categories.pkl', 'rb'))
model_path = r'../data/models/'
cnn_model = r'../data/googlenet_model.npz'
caption_dataset = r'../data/image_caption_dataset.json'
#split_photos = r'/home/rcamachobarranco/datasets/caption_dataset/{}/{}'
split_photos = r'../data/restaurant_photos_split/{}/{}'
labeled_dataset = r'../data/image_caption_dataset_labels.json'
cnn_features = r'../data/image_caption_with_cnn_features.pkl'
# Check add business id

if __name__ == '__main__':
    freeze_support()
    create_train_test_data(label_pickle, folds=6)
    train('googlenet')
    get_labels_features()
