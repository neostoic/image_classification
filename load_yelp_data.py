import hickle as hkl
import numpy as np


def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def _load_batch(path, dtype='float64'):
    """
    load a batch of images
    """
    batch = hkl.load(path + '_data.hpy')
    data = batch / 255.0 # scale between [0, 1]
    labels = one_hot(hkl.load(path + '_labels.hpy'), n=5)  # convert labels to one-hot representation with 5 classes
    return data.astype(dtype), labels.astype(dtype)


def _grayscale(a, pixels):
    return a.reshape(a.shape[0], 3, pixels, pixels).mean(1).reshape(a.shape[0], -1)


def yelp_data(dtype='float64', grayscale=True, pixels=64, data_dir=r'D:\Yelp\datasets', batches=6):
    input_files = []

    for k in range(1, batches):
        input_files.append(data_dir + '\\train_images_batch' + str(k) + '_' + str(pixels))
    input_files.append(data_dir + '\\test_images_' + str(pixels))

    # train
    x_train = []
    y_train = []

    # TODO: change to 5 to get all batches
    for k in xrange(5):
        x, t = _load_batch(input_files[k], dtype=dtype)
        x_train.append(x)
        y_train.append(t)

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    x_train = x_train.reshape((x_train.shape[0], 3, pixels, pixels))

    # test
    x_test, y_test = _load_batch(input_files[batches - 1], dtype=dtype)
    x_test = x_test.reshape((x_test.shape[0], 3, pixels, pixels))
    
    if grayscale:
        x_train = _grayscale(x_train, pixels)
        x_test = _grayscale(x_test, pixels)

    return (x_train, y_train), (x_test, y_test)