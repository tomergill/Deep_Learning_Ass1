import loglinear as ll
import random
import utils as ut
import numpy as np

STUDENT = {'name': 'Tomer Gill',
           'ID': '318459450'}


def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    size = len(features)
    return np.array([(100 * features.count(bigram) / size) for bigram, i in ut.F2I.iteritems()])


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features)  # convert features to a vector.
        y = ut.L2I[label]  # convert the label to number if needed.
        if ll.predict(x, params) == y:
            good += 1
        else:
            bad += 1

    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = ut.L2I[label]           # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            for i, g in enumerate(grads):
                params[i] -= learning_rate * g

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    train_data = ut.TRAIN
    dev_data = ut.DEV
    num_iterations = 10
    learning_rate = 0.2
    in_dim = len(ut.F2I)
    out_dim = len(ut.L2I)


    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
