# This script loads the data from the image pickles and preprocesses it so that the Keras models can use it.
import hickle as hkl
import numpy as np

MODELS = ['VGG_16', 'VGG_19', 'googlenet', 'inception_v3']


def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _load_batch(path, pixels, dtype='float64', model='VGG_16'):
    """
    load a batch of images
    """
    batch = hkl.load(path + '_data.hpy')
    if model in MODELS[0:1]:
        data = _preprocess_vgg(batch.astype(dtype), pixels)
    elif model in MODELS[2]:
        data = _preprocess_googlenet(batch.astype(dtype), pixels)
    elif model in MODELS[3]:
        data = _preprocess_inception_v3(batch.astype(dtype))
    else:
        raise ValueError
    labels = one_hot(hkl.load(path + '_labels.hpy'), n=5)  # convert labels to one-hot representation with 5 classes
    return data, labels.astype(dtype)


def _grayscale(a, pixels):
    """
    Convert image to grayscale
    """
    return a.reshape(a.shape[0], 3, pixels, pixels).mean(1).reshape(a.shape[0], -1)


def _preprocess_vgg(batch, pixels):
    """
    Preprocess the data if it is for VGG-16 model
    """
    rgb_mean = [123.68, 116.779, 103.939]
    img_size = pixels * pixels
    batch[:, 0:img_size] -= rgb_mean[0]
    batch[:, img_size:(2 * img_size)] -= rgb_mean[1]
    batch[:, (2 * img_size):(3 * img_size)] -= rgb_mean[2]
    return batch


def _preprocess_googlenet(batch, pixels):
    """
        Preprocess the data if it is for googlenet model
    """
    rgb_mean = np.asarray([123, 117, 104])
    img_size = pixels * pixels
    batch[:, 0:img_size] -= rgb_mean[0]
    batch[:, img_size:(2 * img_size)] -= rgb_mean[1]
    batch[:, (2 * img_size):(3 * img_size)] -= rgb_mean[2]
    return batch


def _preprocess_inception_v3(batch):
    """
        Preprocess the data if it is for inception-v3 model
    """
    data = (batch - 128) / 128.
    return data


def _preprocess_original(batch):
    """
        Preprocess the data if its for CIFAR10 model
        """
    return batch / 255.0  # scale between [0, 1]


def load_yelp_data(dtype='float64', grayscale=True, pixels=64, data_dir=r'D:\Yelp\datasets\\', batches=6, model='VGG_16'):
    """
    Loads the data and transforms it to numpy arrays divided by train and test data.
    """
    input_files = []

    for k in range(1, batches):
        input_files.append(data_dir + 'train_images_batch' + str(k) + '_' + str(pixels))
    input_files.append(data_dir + 'test_images_' + str(pixels))

    # train
    x_train = []
    y_train = []

    for k in xrange(0, batches - 1):
        x, t = _load_batch(input_files[k], pixels, dtype=dtype, model=model)
        x_train.append(x)
        y_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    x_train = x_train.reshape((x_train.shape[0], 3, pixels, pixels))

    # test
    x_test, y_test = _load_batch(input_files[batches - 1], pixels, dtype=dtype, model=model)
    x_test = x_test.reshape((x_test.shape[0], 3, pixels, pixels))

    if grayscale:
        x_train = _grayscale(x_train, pixels)
        x_test = _grayscale(x_test, pixels)

    return (x_train, y_train), (x_test, y_test)
