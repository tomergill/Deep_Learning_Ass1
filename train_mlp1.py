import mlp1 as m1
import random
import utils as ut
import numpy as np

STUDENT = {'name': 'Tomer Gill',
           'ID': '318459450'}


def feats_to_vec(features, F2I):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    size = len(features)
    return np.array([(100 * features.count(bigram) / size) for bigram, i in F2I.iteritems()])


def accuracy_on_dataset(dataset, params, F2I):
    good = bad = 0.0
    for label, features in dataset:
        x = feats_to_vec(features, F2I)  # convert features to a vector.
        y = ut.L2I[label]           # convert the label to number
        if m1.predict(x, params) == y:
            good += 1
        else:
            bad += 1

    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params, F2I = ut.F2I):
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
            x = feats_to_vec(features, F2I)  # convert features to a vector.
            y = ut.L2I[label]  # convert the label to number if needed.
            loss, grads = m1.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            for i, g in enumerate(grads):
                params[i] -= learning_rate * g

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params, F2I)
        dev_accuracy = accuracy_on_dataset(dev_data, params, F2I)
        print I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':
    test_bigrams = True

    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    num_iterations = 30
    learning_rate = 0.01
    hidden_dim = 60
    out_dim = len(ut.L2I)

    if test_bigrams:
        train_data = ut.TRAIN
        dev_data = ut.DEV
        in_dim = len(ut.F2I)

        params = m1.create_classifier(in_dim, hidden_dim, out_dim)
        trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    else:
        train_data, dev_data, F2I = ut.get_unigrams()
        in_dim = len(F2I)

        params = m1.create_classifier(in_dim, hidden_dim, out_dim)
        trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params, F2I)

    predictTest = False  # set wether the log-liniear should predict the test data and write it to the file

    if predictTest:
        I2L = {i: l for l, i in ut.L2I.iteritems()}
        print I2L
        test = open("test.pred", "w")
        for feature in ut.getTEST():
            lang = m1.predict(feats_to_vec(feature), trained_params)
            test.write(I2L[lang] + '\n')
            print I2L[lang]
        test.close()
