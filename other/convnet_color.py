# CNN for image prediction with color images.
# Based on Sander Dieleman's Theano tutorial available at:
# https://github.com/benanne/theano-tutorial/blob/master/6_convnet.py
# See LICENSE file for more information regarding copyright.
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

from memload import load_yelp_data as load

pixels = 64

# load data
x_train, t_train, x_test, t_test = load.yelp_data(dtype=theano.config.floatX, grayscale=False, pixels=pixels)
labels_test = np.argmax(t_test, axis=1)

# reshape data
x_train = x_train.reshape((x_train.shape[0], 3, pixels, pixels))
x_test = x_test.reshape((x_test.shape[0], 3, pixels, pixels))

# define symbolic Theano variables
x = T.tensor4()
t = T.matrix()


# define model: neural network
def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))


def momentum(cost, params, learning_rate, momentum):
    grads = theano.grad(cost, params)
    updates = []

    for p, g in zip(params, grads):
        mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i - learning_rate * g
        updates.append((mparam_i, v))
        updates.append((p, p + v))

    return updates


def model(x, w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o):
    c1 = T.maximum(0, conv2d(x, w_c1) + b_c1.dimshuffle('x', 0, 'x', 'x'))
    p1 = max_pool_2d(c1, (3, 3))

    c2 = T.maximum(0, conv2d(p1, w_c2) + b_c2.dimshuffle('x', 0, 'x', 'x'))
    p2 = max_pool_2d(c2, (2, 2))

    p2_flat = p2.flatten(2)
    h3 = T.maximum(0, T.dot(p2_flat, w_h3) + b_h3)
    p_y_given_x = T.nnet.softmax(T.dot(h3, w_o) + b_o)
    return p_y_given_x

#w_c1 = init_weights((4, 3, 3, 3))
#b_c1 = init_weights((4,))
#w_c2 = init_weights((8, 4, 3, 3))
#b_c2 = init_weights((8,))
#w_h3 = init_weights((8 * 4 * 4, 100))
#b_h3 = init_weights((100,))
#w_o = init_weights((100, 5)) # Change to number of classes
#b_o = init_weights((5,))

w_c1 = init_weights((32, 1, 3, 3))
b_c1 = init_weights((32,))
w_c2 = init_weights((64, 32, 3, 3))
b_c2 = init_weights((64,))
w_h3 = init_weights((64 * 3 * 3, 625))
b_h3 = init_weights((625,))
w_o = init_weights((625, 5)) # Change to number of classes
b_o = init_weights((5,))

params = [w_c1, b_c1, w_c2, b_c2, w_h3, b_h3, w_o, b_o]

p_y_given_x = model(x, *params)
y = T.argmax(p_y_given_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(p_y_given_x, t))

updates = momentum(cost, params, learning_rate=0.01, momentum=0.9)

# compile theano functions
train = theano.function([x, t], cost, updates=updates)
predict = theano.function([x], y)

# train model
batch_size = 48

for i in range(50):
    print "iteration %d" % (i + 1)
    for start in range(0, len(x_train), batch_size):
        x_batch = x_train[start:start + batch_size]
        t_batch = t_train[start:start + batch_size]
        cost = train(x_batch, t_batch)

    predictions_test = predict(x_test)
    accuracy = np.mean(predictions_test == labels_test)
    print "accuracy: %.5f" % accuracy
    print
