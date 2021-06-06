import numpy as np

STUDENT = {'name': 'Israel Cohen', 'name2': 'Hananel Hadad',
           'ID': '205812290', 'ID2': '313369183'}


def oneHotEncoding(y, dim):
    encoded = [0 for j in range(dim)]
    encoded[y] = 1
    return np.array(encoded)


def softmax(x):
    """
    Return softmax of x
    """
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).reshape(-1, )


def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a log-linear classifier with given params on input x.
    """
    W, b = params
    return softmax(np.dot(W.T, x) + b)


def predict(x, params):
    """
    Return the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.

    params: a list of the form [(W, b)]
    W: matrix
    b: vector
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    y = oneHotEncoding(y, params[1].shape[0])  # (out_dim,)
    y_hat = classifier_output(x, params)  # (out_dim,)
    loss = cross_entropy(y_hat, y)
    gb = y_hat - y  # (out_dim,)
    y_hat = y_hat.reshape(1, -1)  # (1,outdim)
    y = y.reshape(1, -1)  # (1,outdim)
    x = np.array(x).reshape(-1, 1)  # (in_dim,1)
    gW = np.dot(x, (y_hat - y))  # (in_dim,outdim)
    return loss, [gW, gb]


def cross_entropy(y_hat, y):
    """
    return cross entropy loss of x's prediction, given real label y
    """

    return -np.sum(y * np.log(y_hat + 1e-12))  # the 1e-12 is to not do log(0)


def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W, b]


if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1, 2]))
    print(test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001, 1002]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array([0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001, -1002]))
    print(test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6

    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.

    from grad_check import gradient_check

    W, b = create_classifier(3, 4)


    def _loss_and_W_grad(W):
        global b
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W
        loss, grads = loss_and_gradients([1, 2, 3], 0, [W, b])
        return loss, grads[1]


    for _ in range(10):
        W = np.random.randn(W.shape[0], W.shape[1])  # 3X4
        b = np.random.randn(b.shape[0])  # 4
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)



