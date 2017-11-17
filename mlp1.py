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
    W1, b1, W2, b2 = params
    h1 = tanh(x.dot(W1) + b1)
    probs = softmax(h1.dot(W2) + b2)
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    # YOU CODE HERE
    W1, b1, W2, b2 = params
    probs = classifier_output(x, params)
    loss = -1 * math.log(probs[y])

    yvector = np.zeros_like(x)
    yvector[y] = 1

    def gtag(v):
        return 1 - tanh(v) ** 2

    #gWs
    a = [x]
    a.append(tanh(a[0].dot(W1)))
    a.append(tanh(a[1].dot(W2)))
    lower_delta = [a[2] - yvector]
    lower_delta.append((W1.dot(lower_delta[0])) * gtag(a[1]))
    lower_delta.reverse()

    delta = [a_i.dot(lower_delta[i]) for i, a_i in enumerate(a[1:])]

    return [loss, gb, delta]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    W1 = np.random.rand(in_dim, hid_dim)
    b1 = np.random.rand(hid_dim)
    W2 = np.random.rand(hid_dim, out_dim)
    b2 = np.random.rand(out_dim)
    params = [W1,b1,W2,b2]
    return params

