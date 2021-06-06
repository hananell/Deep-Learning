import mlpn as mn
import numpy as np
from train_loglin import make_data, text_to_bigrams, text_to_unigrams, bi_uni
import random
import sys
from ast import literal_eval

STUDENT = {'name': 'Israel Cohen & HANANEL HADAD',
           'ID': '205812290 & 313369183'}

# Hyper params:
learning_rate = 0.02
num_iterations = 20
vocabLength = 784
dims = literal_eval(sys.argv[1])


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


# calculate accuracy of prediction on given dataset, using given params
def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        y = L2I[label]
        x = feats_to_vec(features)
        pred = mn.predict(x, params)
        if pred == y:
            good += 1
        else:
            bad += 1
        pass
    return good / (good + bad)



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
            loss, grads = mn.loss_and_gradients(x, y, params)
            loss += loss
            # back prop - update gradients with SGD:
            for i in range(len(params)):
                params[i] = params[i] - learning_rate * grads[i]
        # train_loss is average of one sample
        train_loss = loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        # print results
        print(I,'\t', train_loss,'\t', train_accuracy,'\t', dev_accuracy)
        #print(I, train_loss, train_accuracy, dev_accuracy)
    return params



if __name__ == '__main__':
    # data and vocab
    train_data, dev_data, test_data, vocab = make_data()
    # label strings to IDs
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train_data]))))}

    # initialize params, train them
    dims.insert(0, vocabLength)         # add input dim at first
    params = mn.create_classifier(dims)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
