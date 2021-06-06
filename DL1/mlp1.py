import numpy as np
from loglinear import oneHotEncoding, softmax, cross_entropy



def zScore(x):
    """
    Doing z-score normalization
    """
    mean = np.mean(x)
    std = np.std(x)
    return np.array([(x[i]-mean)/std for i in range(len(x))])


def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a log-linear classifier with given params on input x.
    """
    x = np.array(x)
    x = x.reshape(-1, 1)                       # (in_dim,1)
    w1, b1, w2, b2 = params                    # (in_dim,h1) (h1,) (h1,out_dim) (out_dim,)
    h1 = np.tanh(np.dot(w1.T, x) + b1.reshape(-1, 1))  # (h1,1)
    h2 = np.dot(w2.T, h1) + b2.reshape(-1, 1)          # (output_dim, 1)
    return softmax(h2)


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # z=w1*x+b1     h1=tanh(z)    h2=w2*h1+b2     y_hat=softmax(h2)      loss=nll
    # dl/dh2=y_hat-y    dh2/db2=1   =>  dl/db2=y_hat-y
    #                   dh2/dw2=h1  =>  dl/dw2=(y_hat-y)*h1
    #                   dh2/dh1=w2  =>  dl/dh1=(y_hat-y)*w2
    #                                   dh1/dz=1-tan(z)^2
    #                                   dz/db1=1   =>  dl/db1=(y_hat-y)*w2*(1-tanh(z)^2)*1
    #                                   dz/dw1=x   =>  dl/dw1=(y_hat-y)*w2*(1-tanh(z)^2)*x
    #
    #
    #

    w1, b1, w2, b2 = params                         # (in_dim,h1) (h1,) (h1,out_dim) (out_dim,)
    x = np.array(x).reshape(-1, 1)                  # (in_dim,1)
    z = np.dot(w1.T, x) + b1.reshape(-1, 1)         # (h1,1)
    z = zScore(z)
    h1 = np.tanh(z)                                 # (h1,1)
    y = oneHotEncoding(y, params[3].shape[0])       # (out_dim,)
    y_hat = classifier_output(x, params)            # (out_dim,)
    loss = cross_entropy(y_hat, y)                  # scalar :-)
    gb2 = y_hat-y                                   # (out_dim,)
    gw2 = np.dot(h1, gb2.reshape(1, -1))            # (h1,out_dim)
    #      (h1,out_dim) dot (out_dim,1)   * (h1,1)
    gb1 = np.dot(w2,  gb2.reshape(-1, 1)) * (1 - np.square(np.tanh(z))).reshape(-1,1)       # (h1,1)
    gw1 = np.dot(x, gb1.T)                          # (in_dim,h1)
    return loss, [gw1, gb1, gw2, gb2]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    # dimensions:  (in_dim,hid_dim) (hid_dim,) (hid_dim,out_dim) (out_dim,)
    w1 = np.random.rand(in_dim, hid_dim)
    b1 = np.random.rand(hid_dim)
    w2 = np.random.rand(hid_dim, out_dim)
    b2 = np.random.rand(out_dim)

    # w1 = np.random.uniform(0, 1, in_dim * hid_dim).reshape(in_dim, hid_dim)
    # b1 = np.random.uniform(0, 1, hid_dim)
    # w2 = np.random.uniform(0, 1, hid_dim * out_dim).reshape(hid_dim, out_dim)
    # b2 = np.random.uniform(0, 1, out_dim)

    return [w1, b1, w2, b2]

