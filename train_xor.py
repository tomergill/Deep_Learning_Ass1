import random

import numpy

import mlp1 as m1
import xor_data as xr


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        if m1.predict(features, params) == label:
            good += 1
        else:
            bad += 1

    return good / (good + bad)


if __name__ == "__main__":

    num_iterations = 0
    acc = 0

    learning_rate = 0.1
    hidden_dim = 10
    train_data = [(l, numpy.array(f)) for l, f in xr.data]
    dev_data = list(train_data)
    in_dim = 2
    out_dim = 2

    params = m1.create_classifier(in_dim, hidden_dim, out_dim)

    print "itn train_l train_a dev_a"

    while acc < 1.0:
        num_iterations += 1
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            # x = feats_to_vec(features, F2I)  # convert features to a vector.
            # y = ut.L2I[label]  # convert the label to number if needed.
            loss, grads = m1.loss_and_gradients(features, label, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            for i, g in enumerate(grads):
                params[i] -= learning_rate * g

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        acc = accuracy_on_dataset(dev_data, params)
        print num_iterations, train_loss, train_accuracy, acc
