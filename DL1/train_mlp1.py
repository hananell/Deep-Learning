import mlp1 as mp
import numpy as np
import random
from train_loglin import make_data, text_to_bigrams, text_to_unigrams, bi_uni



# Hyper params:
learning_rate = 0.05
num_iterations = 11
vocabLength = 784
hid_dim = 200

end = False


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
            x = feats_to_vec(features)
            # calculate loss
            loss, grads = mp.loss_and_gradients(x, y, params)
            loss += loss
            # back prop - update gradients with SGD:
            params[0] = params[0] - learning_rate * grads[0]
            params[1] = params[1] - learning_rate * grads[1].reshape(-1,)
            params[2] = params[2] - learning_rate * grads[2]
            params[3] = params[3] - learning_rate * grads[3].reshape(-1,)
        # train_loss is average of one sample
        train_loss = loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        # print results
        print(I, train_loss, train_accuracy, dev_accuracy)
        # print(I,'\t', train_loss,'\t', train_accuracy,'\t', dev_accuracy)
        # if dev_accuracy >= 0.86:
        #     global end
        #     end = True
        #     return params
    return params

def testPred():
    f = open("test.pred","w")
    for label, features in test_data:
        x = feats_to_vec(features)
        pred = mp.predict(x, trained_params)
        language = list(L2I.keys())[list(L2I.values()).index(pred)]
        f.write(language + '\n')
    f.close()

if __name__ == '__main__':
    # data and vocab
    train_data, dev_data, test_data, vocab = make_data()
    # label strings to IDs
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in train_data]))))}

    # initialize params, train them
    params = mp.create_classifier(vocabLength, hid_dim, len(L2I))
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    # # # search for good hid_dim and do test
    # for i in range(250, 440, 40):
    #     print("\n", i, ":")
    #     hid_dim = i
    #     params = mp.create_classifier(vocabLength, hid_dim, len(L2I))
    #     trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    #     if end:
    #         break
    # testPred()
