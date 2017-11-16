import numpy as np

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    # YOU CODE HERE
    return ...

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    params = []
    return params

