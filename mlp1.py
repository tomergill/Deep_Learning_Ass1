import numpy as np

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
    probs = classifier_output(x, params)
    return ...

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    W1 = np.zeros((in_dim, hid_dim))
    b1 = np.zeros(hid_dim)
    W2 = np.zeros((hid_dim, out_dim))
    b2 = np.zeros(out_dim)
    params = [W1,b1,W2,b2]
    return params

