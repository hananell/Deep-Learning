import loglinear as ll
import random
from collections import Counter
import numpy as np



# Hyper params:
learning_rate = 0.03
num_iterations = 20
vocabLength = 784
bi_uni = "bi"        # "bi" for bigrams, "uni" for unigrams


# read data from file
def read_data(fname):
    data = []
    for line in fname:
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    return data


# make data, using given functions
def make_data():
    # loading the data:
    train_f = open('train', encoding="latin-1")
    dev_f = open('dev', encoding="latin-1")  # dev => validation
    test_f = open('test', encoding="latin-1")
    # make list of labels and the lines
    train = read_data(train_f)
    dev = read_data(dev_f)
    test = read_data(test_f)
    # close files
    train_f.close()
    dev_f.close()
    test_f.close()
    # make vocabulary - most common bigrams or unigrams in the training set
    if bi_uni == "bi":
        TRAIN = [(l, text_to_bigrams(t)) for l, t in train]
    elif bi_uni == "uni":
        TRAIN = [(l, text_to_unigrams(t)) for l, t in train]
    fc = Counter()
    for l, feats in TRAIN:
        fc.update(feats)    # for each bigram, update counter
    vocabulary = np.array(list(set([x for x, c in fc.most_common(vocabLength)])))
    return train, dev, test, vocabulary


# split given string to its chars
def split(word):
    return [char for char in word]


def feats_to_vec(features):
    """
    return vector representing given line.
    vec[i] = number of occurrences of vacab[i]
    """
    if bi_uni == "bi":
        bigrams = np.array(text_to_bigrams(features))
        vec = np.zeros(vocabLength)
        for i in range(vocabLength):
            occurr_indexes = np.where(bigrams == vocab[i])[0]
            vec[i] = len(occurr_indexes)
        return vec
    elif bi_uni == "uni":
        unigrams = np.array(text_to_unigrams(features))
        vec = np.zeros(vocabLength)
        for i in range(vocabLength):
            occurr_indexes = np.where(unigrams == vocab[i])[0]
            vec[i] = len(occurr_indexes)
        return vec


def accuracy_on_dataset(dataset, params):
    """
    calculate accuracy of prediction on given dataset, using given params
    """
    good = bad = 0.0
    for label, features in dataset:
        y = L2I[label]
        x = feats_to_vec(features)
        pred = ll.predict(x, params)
        if pred == y:
            good += 1
        else:
            bad += 1
        pass
    return good / (good + bad)


def text_to_bigrams(text):  # get 2 letters words:
    """
    return bigrams list of given text
    """
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


def text_to_unigrams(text):
    """
    return unigrams list of given text
    """
    return [c for c in text]


# train our classifier with given data and params,
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
            x = feats_to_vec(features)
            # calculate loss
            loss, grads = ll.loss_and_gradients(x, y, params)
            loss += loss
            # back prop - update gradients with SGD:
            params[0] = params[0] - learning_rate * grads[0]
            params[1] = params[1] - learning_rate * grads[1]
        # train_loss is average of one sample
        train_loss = loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        # print results
        # print(I,'\t', train_loss,'\t', train_accuracy,'\t', dev_accuracy)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


def test(test_data, vocab):
    f = open('test.pred', 'w')
    for label, features in test_data:
        x = feats_to_vec(features, vocab)
        pred = ll.predict(x, trained_params)
        language = list(L2I.keys())[list(L2I.values()).index(pred)]
        f.write(language + '\n')
    f.close()


if __name__ == '__main__':
    # data and vocab
    train_data, dev_data, test_data, vocab = make_data()
    # label strings to IDs
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train_data]))))}

    # initialize params, train them
    params = ll.create_classifier(vocabLength, len(L2I))
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    # test
    # test()
