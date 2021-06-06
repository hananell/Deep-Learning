import numpy as np
from loglinear import oneHotEncoding, softmax, cross_entropy
from mlp1 import zScore


STUDENT = {'name': 'Israel Cohen', 'name2': 'Hananel Hadad',
           'ID': '205812290', 'ID2': '313369183'}


def tanh(x):
    """
    Normalize x, than do tanh
    """
    x = zScore(x)
    return np.tanh(x)


def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a log-linear classifier with given params on input x.
    """
    curInput = x.reshape(-1, 1)  # (in_dim,1)
    # for each layer, calculate next layer
    for i in range(0, len(params) - 1, 2):
        w, b = params[i], params[i + 1]  # (dim_cur,dim_next) (dim_next,)
        z = np.dot(w.T, curInput) + b.reshape(-1, 1)  # (next_dim,1)
        # if not final layer, do tanh and define as next input
        if i < len(params) - 1:
            h = tanh(zScore(z))
            curInput = h
    return softmax(z)


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # z1=w1*x+b1     h1=tanh(z)    z2=w2*h1+b2     y_hat=softmax(h2)      loss=nll
    # dl/dz2=y_hat-y    dz2/db2=1    =>  dl/db2=y_hat-y
    #                   dz2/dw2=h1   =>  dl/dw2=(y_hat-y)*h1
    #                   dz2/dh1=w2   =>  dl/dh1=(y_hat-y)*w2
    #                                    dh1/dz1=1-tanh(z1)^2
    #                                    dz/db1=1   =>  dl/db1=(y_hat-y)*w2*(1-tanh(z1)^2)*1
    #                                    dz/dw1=x   =>  dl/dw1=(y_hat-y)*w2*(1-tanh(z1)^2)*x
    #
    # dl/dbi=(y_hat-y)*[wj(1-tanh(zj-1) for j in range(i+1,zNum)]
    # dl/dwi=dl/dbi*hi-1 (or x if i==0)

    # extract weights
    ws = []
    bs = []
    for i in range(0, len(params) - 1, 2):
        ws.append(params[i])  # (next_dim,cur_dim)
        bs.append(params[i + 1])  # (next_dim,1)

    # calculate layers
    curInput = x.reshape(-1, 1)  # (in_dim,1)
    zs = []
    hs = []
    for i in range(len(ws)):
        #   (next_dim,cur_dim) dot (cur_dim,1)  +  (next_dim,1)
        zs.append(np.dot(ws[i].T, curInput) + bs[i].reshape(-1, 1))  # (next_dim,1)
        if i < len(ws):  # continue - make next hidden layer, unless this layer is the last
            hs.append(tanh(zs[i]))
            curInput = hs[i]

    # end net values
    y = oneHotEncoding(y, bs[-1].shape[0])  # (out_dim,)
    y_hat = softmax(hs[-1]).reshape(-1, )  # (out_dim,)
    loss = cross_entropy(y_hat, y)  # scalar :-)

    # calculate derivatives from end to start
    curDiv = y_hat - y  # (z_last,)
    bDivs = []
    wDivs = []
    for i in range(len(ws) - 1, -1, -1):
        bDivs.insert(0, curDiv.reshape(-1, ))  # (z_i,)
        if i == 0:  # get x instead of h[-1]
            #                      (z_i-1,1) dot (1,z_i)
            wDivs.insert(0, np.dot(x.reshape(-1, 1), curDiv.reshape(1, -1)))  # (z_i-1,z_i)
        else:
            #                (z_i-1,1) dot (1,z_i)
            wDivs.insert(0, np.dot(hs[i - 1], curDiv.reshape(1, -1)))  # (z_i-1,z_i)
        if i > 0:  # update curDiv *= wi(q-tanh(zi-1))
            #           (z_i-1,z_i) dot (z_i,1)          *   (z_i-1,1)
            curDiv = np.dot(ws[i], curDiv.reshape(-1, 1)) * (1 - np.square(tanh(zs[i - 1])))  # (z_i-1,1)

    # merge divs and return
    allDivs = []
    for i in range(len(zs)):
        allDivs.append(wDivs[i])
        allDivs.append(bDivs[i])

    return loss, allDivs


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
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(len(dims) - 1):
        w = np.random.rand(dims[i], dims[i + 1])
        b = np.random.rand(dims[i + 1])
        params.append(w)
        params.append(b)
    return params