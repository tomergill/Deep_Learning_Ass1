import numpy as np
import math

STUDENT = {'name': 'Tomer Gill',
           'ID': '318459450'}

def tanh(x):
    return np.divide(np.exp(x) + 1, np.expm1(x))

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def classifier_output(x, params):
    W, b = params[0], params[1]
    h = x.dot(W) + b
    for W, b in zip(params[2:], params[3:]):
        h = tanh(h).dot(W) + b
    probs = softmax(h)
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    # YOU CODE HERE
    return ...

def uniform_init(dim1, dim2=0):
    epsilon = math.sqrt(6) / math.sqrt(dim1 + dim2)
    if dim2 == 0:
        return np.random.uniform(-1 * epsilon, epsilon, dim1)
    return np.random.uniform(-1 * epsilon, epsilon, [dim1, dim2]) #else

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
        params.append(uniform_init(dim1,dim2))
        params.append(uniform_init(dim2))
    return params

