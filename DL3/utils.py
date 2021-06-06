from matplotlib import pyplot as plt
from bilstmTrain import epochs, seperator
import torch


# Creates vocab of given text (words or labels)
def createVocab(text, kind):
    dict = {}
    i = 0
    # add each token in text
    for sentence in text:
        for token in sentence:
            if token not in dict:
                dict[token.strip()] = i
                i += 1
    # add special tokens
    if kind == "words":
        # dict['PAD_BEGIN'] = len(dict)
        # dict['PAD_END'] = len(dict)
        dict['UNKNOWN'] = len(dict)  # for new words in dev and test.
    # if kind == "labels":
    #     dict['BEGIN_LABEL'] = len(dict)  # for PAD_BEGIN
    #     dict['END_LABEL'] = len(dict)    # for PAD_END. we don't need UNKNOWN_LABEL because we skip those words
    return dict


# Creates vocab of all chars in vocabW
def createVocabC(vocabW):
    vocabC = {}
    i = 0
    for word in list(vocabW.keys()):
        for char in word:
            if char not in vocabC:
                vocabC[char] = i
                i += 1
    vocabC['UNKNOWN'] = len(vocabC)     # for new chars in dev and test.

    return vocabC


# Reads train and dev sets
def readLabeledData(fileName, kind, vocabL={}):
    # variables to store the data. W for words, L for labels
    sentencesW = []  # one sentence - list of words
    sentencesL = []
    with open(fileName + "/" + kind) as f:
        for line in f:
            # reached start of sentence - add words/labels to sentenceW/sentenceL
            sentenceW = []
            sentenceL = []
            while line != '\n' and line:
                if kind == "dev" and line.split(seperator)[1].strip() not in vocabL:    # in dev, skip lines with unknown labels
                    continue
                sentenceW.append(line.split(seperator)[0])
                sentenceL.append(line.split(seperator)[1].strip())
                line = f.readline()
            # end of sentence - save sentenceW/sentenceL
            sentencesW.append(sentenceW.copy())
            sentencesL.append(sentenceL.copy())

    #padBeginEnd(sentencesW, sentencesL)

    return sentencesW, sentencesL


# Reads test set
def readTestData(trainFile):
    # variables to store the data. W for words, L for labels
    sentencesW = []
    with open(trainFile + "/test") as f:
        for line in f:
            # reached start of sentence - add words/labels to sentenceW/sentenceL
            sentenceW = []
            while line != '\n' and line:
                sentenceW.append(line.strip())
                line = f.readline()
            # end of sentence - save sentenceW/sentenceL
            sentencesW.append(sentenceW.copy())

    #padBeginEnd(sentencesW)
    return sentencesW


# Puts start and end tokens in each sentence
def padBeginEnd(sentencesW, sentencesL=None):
    # add tokens for start and end of sentence
    for i in range(len(sentencesW)):
        sentencesW[i].insert(0, 'PAD_BEGIN')
        sentencesW[i].append('PAD_END')
        # pad labels only if given (in train and dev but not in test)
        if sentencesL:
            sentencesL[i].insert(0, 'BEGIN_LABEL')
            sentencesL[i].append('END_LABEL')


# Encodes entire text (list of sentences) of words or of labels
def encodeText(text: list, vocab: dict):

    # Converts list of numbers to torch
    def torchi(listOfNums):
        return torch.tensor(listOfNums, dtype=torch.long)

    # Encodes single sentence
    def encodeSentence(sentence):
        return torchi([vocab[token] if token in vocab else vocab['UNKNOWN'] for token in sentence])  # 'else' activated only in dev/test words. we don't read lines with unrecognized labels

    return [encodeSentence(sentence) for sentence in text]


# Plots one measurement
def plotMeasurement(name, trainData, devData):
    epochsList = [i for i in range(epochs)]
    plt.figure()
    plt.title(name)
    plt.plot(epochsList, trainData, label="Train")
    plt.plot(epochsList, devData, label="Dev")
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.locator_params(axis="x", integer=True, tight=True)  # make x axis to display only whole number (iterations)
    plt.legend()
    plt.savefig(f"{name}.jpeg")


# Plot accuracies
def plotGraphs(accuracy_t, accuracy_v, losses_t, losses_v, name):
    plotMeasurement(f"Accuracy_{name}", accuracy_t, accuracy_v)
    plotMeasurement(f"Loss_{name}", losses_t, losses_v)