import numpy as np
import math

STUDENT = {'name': 'Tomer Gill',
           'ID': '318459450'}


# def tanh(x):
#     return np.divide(np.expm1(x), np.exp(x) + 1)


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
    h1 = np.tanh(x.dot(W1) + b1)
    probs = softmax(h1.dot(W2) + b2)
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    # YOU CODE HERE
    W1, b1, W2, b2 = params

    probs = classifier_output(x, params)
    loss = -1 * math.log(probs[y])

    def gtag(v):
        return 1 - np.tanh(v) ** 2

    """
    yvector = np.zeros_like(x)
    yvector[y] = 1

    

    # gWs
    a = [x]
    a.append(tanh(a[0].dot(W1)))
    a.append(tanh(a[1].dot(W2)))
    lower_delta = [a[2] - yvector]
    lower_delta.append((W1.dot(lower_delta[0])) * gtag(a[1]))
    lower_delta.reverse()

    delta = [a_i.dot(lower_delta[i]) for i, a_i in enumerate(a[1:])]
    """

    # gradients of W2 & b2

    # gb2
    gb2 = probs.copy()
    gb2[y] -= 1

    # gW2
    h = np.tanh(x.dot(W1) + b1)
    gW2 = np.outer(h, probs)
    gW2[:, y] -= h

    # gradients of W1 & b1
    dtanh_db1 = gtag(x.dot(W1) + b1)
    dl_dtanh = W2.dot(probs) - W2[:, y]

    # gb1
    gb1 = dl_dtanh * dtanh_db1

    # gW1
    gW1 = np.outer(x, (dl_dtanh * dtanh_db1))

    return loss, [gW1, gb1, gW2, gb2]


def uniform_init(dim1, dim2=0):
    epsilon = math.sqrt(6) / math.sqrt(dim1 + dim2)
    if dim2 == 0:
        return np.random.uniform(-1 * epsilon, epsilon, dim1)
    return np.random.uniform(-1 * epsilon, epsilon, [dim1, dim2])  # else


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    W1 = uniform_init(in_dim, hid_dim)
    b1 = uniform_init(hid_dim)
    W2 = uniform_init(hid_dim, out_dim)
    b2 = uniform_init(out_dim)
    params = [W1, b1, W2, b2]
    return params


if __name__ == '__main__':
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W1, b1, W2, b2 = create_classifier(3, 7, 9)


    def _loss_and_W2_grad(W2):
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W1, b1, W2, b2])
        return loss, grads[2]


    def _loss_and_W1_grad(W1):
        global b1
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W1, b1, W2, b2])
        return loss, grads[0]


    def _loss_and_b1_grad(b1):
        global W1
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W1, b1, W2, b2])
        return loss, grads[1]


    def _loss_and_b2_grad(b2):
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W1, b1, W2, b2])
        return loss, grads[3]


    for _ in xrange(10):
        W1 = np.random.randn(W1.shape[0], W1.shape[1])
        b1 = np.random.randn(b1.shape[0])
        W2 = np.random.randn(W2.shape[0], W2.shape[1])
        b2 = np.random.randn(b2.shape[0])
        loss, grads = loss_and_gradients(np.array([1, 2, 3]), 0, [W1, b1, W2, b2])

        gradient_check(_loss_and_W2_grad, W2)
        gradient_check(_loss_and_W1_grad, W1)
        gradient_check(_loss_and_b1_grad, b1)
        gradient_check(_loss_and_b2_grad, b2)
