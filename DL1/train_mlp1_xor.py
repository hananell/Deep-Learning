import numpy as np
import mlp1 as mp
import random
from collections import Counter
from xor_data import data as XORData


# Hyper params:
learning_rate = 0.1
num_iterations = 200
vocabLength = 2
hid_dim = 4


# read data from file
def read_data(fname):
    data = []
    for line in fname:
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    return data


def feats_to_vec(features, vocab):
    """
    return features because it's already good vector
    """
    return features


# make data, using given functions
def make_data():
    # loading the data:
    train = dev = test = XORData
    # make vocabulary - most common bigrams or unigrams in the training set
    fc = Counter()
    for l, feats in train:
        fc.update(feats)    # for each bigram, update counter
    vocabulary = np.array(list(set([x for x, c in fc.most_common(vocabLength)])))
    return train, dev, test, vocabulary


# split given string to its chars
def split(word):
    return [char for char in word]


def accuracy_on_dataset(dataset, params, vocab):
    """
    calculate accuracy of prediction on given dataset, using given params
    """
    good = bad = 0.0
    for label, features in dataset:
        y = L2I[label]
        x = feats_to_vec(features, vocab)
        pred = mp.predict(x, params)
        if pred == y:
            good += 1
        else:
            bad += 1
        pass
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
    for I in range(num_iterations):
        loss = 0.0
        random.shuffle(train_data)
        for label, features in train_data:
            # y is ID of label, x is vector representing features
            y = L2I[label]
            x = feats_to_vec(features, vocab)
            # calculate loss
            loss, grads = mp.loss_and_gradients(x, y, params)
            loss += loss
            # back prop - update gradients with SGD:
            params[0] = params[0] - learning_rate * grads[0]
            params[1] = params[1] - learning_rate * grads[1].reshape(-1,)
            params[2] = params[2] - learning_rate * grads[2]
            params[3] = params[3] - learning_rate * grads[3].reshape(-1, )
        # train_loss is average of one sample
        train_loss = loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params, vocab)
        dev_accuracy = accuracy_on_dataset(dev_data, params, vocab)
        # print results
        #print(I,'\t', train_loss,'\t', train_accuracy,'\t', dev_accuracy)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    # dimensions:  (in_dim,hid_dim) (hid_dim,) (hid_dim,out_dim) (out_dim,)
    w1 = np.random.uniform(0, 1, in_dim * hid_dim).reshape(in_dim, hid_dim)
    b1 = np.random.uniform(0, 1, hid_dim)
    w2 = np.random.uniform(0, 1, hid_dim * out_dim).reshape(hid_dim, out_dim)
    b2 = np.random.uniform(0, 1, out_dim)

    return [w1, b1, w2, b2]


if __name__ == '__main__':
    # data and vocab
    train_data, dev_data, test_data, vocab = make_data()
    # label strings to IDs
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train_data]))))}



    # initialize params, train them
    params = create_classifier(vocabLength, hid_dim, len(L2I))
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    # # search for good hid_dim
    # for i in range(100,400,100):
    #     hid_dim = i
    #     params = mp.create_classifier(vocabLength, hid_dim, len(L2I))
    #     trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params, vocab)

    # test
    # test()
