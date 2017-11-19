import numpy as np
import math

STUDENT = {'name': 'Tomer Gill',
           'ID': '318459450'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def classifier_output(x, params):
    W, b = params[0]
    h = x.dot(W) + b
    probs = list()
    for (W, b) in params[1:]:
        probs.append(np.tanh(h))
        h = probs[-1].dot(W) + b
    probs.append(softmax(h))
    return probs


def predict(x, params):
    return np.argmax(classifier_output[-1](x, params))


def loss_and_gradients(x, y, params):
    all_probs = classifier_output(x, params)  # layers 1 to n and softmax of n
    probs = all_probs[-1]
    layer_probs = all_probs[:-2]
    loss = -1 * math.log(probs[y])

    W_n, b_n = params[-1]

    dl_dtanh = W_n.dot(probs) - W_n[:, y]

    gb_n = probs.copy()
    gb_n[y] -= 1

    gW_n = np.outer(layer_probs[-1], probs)
    temp = np.zeros_like(gW_n)
    temp[:, y] = layer_probs[-1]
    gW_n -= temp

    gradients = [(gW_n, gb_n)]

    layer = len(params) - 2
    r_params = reversed(params)
    for W, b in r_params[1:]:  # [(W1,b1),...,(W_n-1,b_n-1)]
        gb = 1 - np.square(layer_probs[layer].dot(W) + b)
        gW = layer_probs[layer].dot(gb)

        gradients.append((gW * dl_dtanh, gb * dl_dtanh))

        dl_dtanh = (gb * W).dot(dl_dtanh)
        layer -= 1

    return loss, gradients.reverse()

def uniform_init(dim1, dim2=0):
    epsilon = math.sqrt(6) / math.sqrt(dim1 + dim2)
    if dim2 == 0:
        return np.random.uniform(-1 * epsilon, epsilon, dim1)
    return np.random.uniform(-1 * epsilon, epsilon, [dim1, dim2])  # else


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for dim1, dim2 in zip(dims, dims[1:]):
        params.append((uniform_init(dim1, dim2), uniform_init(dim2)))
    return params
