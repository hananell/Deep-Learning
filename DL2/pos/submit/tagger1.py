import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 8
seperator = '\t' if sys.argv[1] == "ner" else ' '
parts = [sys.argv[i] for i in range(2,len(sys.argv),1)]
# constants
winSize = 5
embedding_dim_word = 50
embedding_dim_char = 30
loss_f = CrossEntropyLoss()
max_norm = 2
# hyper parameters of model
hidden_dim = 128
lr = 0.002
batchSize = 1000
filtersNum = 3  # part 5


# Create vocabulary dict of words or labels
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
        dict['PAD_BEGIN'] = len(dict)
        dict['PAD_END'] = len(dict)
        dict['DEFAULT'] = len(dict)         # for new words in dev and test
    return dict


# Reads test set
def readTestData():
    # variables to store the data. W for words, L for labels
    sentencesW = []  # one sentence - list of words
    with open(str(sys.argv[1]) + "/test") as f:
        for line in f:
            line = line.lower()  # lower casing for part 3. not supposed to harm other parts
            # reached start of sentence - add words/labels to sentenceW/sentenceL
            sentenceW = []
            while line != '\n' and line:
                sentenceW.append(line.strip())
                line = f.readline()
            # end of sentence - save sentenceW/sentenceL
            sentencesW.append(sentenceW.copy())
    return sentencesW


# Reads train and dev sets
def readLabeledData(kind, vocabL={}):
    # variables to store the data. W for words, L for labels
    sentencesW = []  # one sentence - list of words
    sentencesL = []
    with open(str(sys.argv[1]) + "/" + kind) as f:
        for line in f:
            #line = line.lower()  # lower casing for part 3. not supposed to harm other parts
            # reached start of sentence - add words/labels to sentenceW/sentenceL
            sentenceW = []
            sentenceL = []
            while line != '\n' and line:
                if kind == "dev" and line.split(seperator)[1].strip() not in vocabL:    # in dev, skip lines with unknown labels
                    continue
                sentenceW.append(line.split(seperator)[0].lower())
                sentenceL.append(line.split(seperator)[1].strip())
                line = f.readline()
            # end of sentence - save sentenceW/sentenceL
            sentencesW.append(sentenceW.copy())
            sentencesL.append(sentenceL.copy())
    return sentencesW, sentencesL


# Returns list of all chars in given list of words
def allChars(wordsList):
    chars = []
    [chars.append(char) for word in wordsList for char in word if char not in chars]
    return chars


# Creates embedding with pre-trained vectors. part4
def createPreEmb():
    # create random weights for all words
    weights = torch.rand((len(vocabW), embedding_dim_word), dtype=torch.float32)
    # part 3 - pre-trained vectors:
    if "part3" in parts:
        #print("part3")
        # read the pre-trained
        preEmbs = np.loadtxt("wordVectors.txt")
        preWords = open("vocab.txt", "r").read().lower().split('\n')    # for lower-casing issue of part3. not supposed to harm other parts
        preWords.remove('')
        preWord2preEmb = {preWord: preEmb for preWord, preEmb in zip(preWords, preEmbs)}
        # for each word in vocabW, if we have pre-trained vector for it, put it instead of the random one
        for i in range(len(vocabW)):
            word = list(vocabW.keys())[list(vocabW.values()).index(i)]
            if word in preWords:
                weights[i] = torch.FloatTensor(preWord2preEmb[word])
    # part 4 - prefix and suffix:
    if "part4" in parts:
        #print("part4")
        # create dict from word in vocabW to its prefix and suffix, and dicts of prefixes and suffixes with their random vectors of embeddings
        # when we encounter same prefix twice, the second random vector will replace the first, that's OK
        word2presuff = {word: (word[:3], word[-3:]) for word in list(vocabW.keys())}
        pref2emb = {word[:3]: torch.rand(embedding_dim_word, dtype=torch.float32) for word in list(vocabW.keys())}
        suff2emb = {word[-3:]: torch.rand(embedding_dim_word, dtype=torch.float32) for word in list(vocabW.keys())}
        # update weights
        for i, word in enumerate(list(vocabW.keys())):
            weights[i] += pref2emb[word2presuff[word][0]] + suff2emb[word2presuff[word][1]]
    return weights


# The neural network
class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        # maps each label to an embedding_dim vector
        self.embeddings = nn.Embedding(len(vocabW), embedding_dim_word, norm_type=2, max_norm=max_norm).requires_grad_(True)
        # use pre-trained embedding
        if "part3" in parts or "part4" in parts:
            self.embeddings = nn.Embedding.from_pretrained(createPreEmb(), norm_type=2, max_norm=max_norm).requires_grad_(True)
        self.fc1 = nn.Linear(embedding_dim_word * winSize, hidden_dim)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, len(vocabL))
        self.dropout2 = nn.Dropout()

    def forward(self, x):                                 # (batchSize, winSize)
        x = self.embeddings(x)                            # (batchSize, winSize, embedSize)
        x = x.view((-1, winSize * embedding_dim_word))    # (batchSize, winSize * embedSize)
        x = self.dropout1(x)
        x = torch.tanh(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# Trains our model
def train(fiveGrams):
    modely.train()
    trainLoader = DataLoader(fiveGrams, batch_size=batchSize, shuffle=True)
    for contextWords, midLabel in trainLoader:
        optimizer.zero_grad()
        output = modely(contextWords)
        loss = loss_f(output, midLabel)
        loss.backward()
        optimizer.step()


# do validation
def validation(fiveGrams):
    modely.eval()
    correctTotal = 0
    lossVal = 0
    devLoader = DataLoader(fiveGrams, batch_size=batchSize, shuffle=True)
    with torch.no_grad():
        for contextWords, midLabel in devLoader:
            output = modely(contextWords)
            predLabelNum = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correctTotal += predLabelNum.eq(midLabel.view_as(predLabelNum)).cpu().sum().item()
            loss = loss_f(output, midLabel)
            lossVal += loss.item()
        accuracy = correctTotal / len(fiveGrams)
        lossVal /= len(devLoader)
    return accuracy, lossVal


# Converts list of numbers to torch
def torchi(listOfNums):
    return torch.tensor(listOfNums, dtype=torch.long)


# fiveGrams are 5 window words, and one mid labelNum
def createFiveGrams(textWords, textLabels, vocabW, vocabL):
    # start with words
    fiveGrams = []
    for sentenceWords, sentenceLabels in zip(textWords,textLabels):
        # for each sentence (list of words): add 2 padding at both sides
        sentenceWords[:0] = ['PAD_BEGIN', 'PAD_BEGIN']
        sentenceWords.extend(['PAD_END', 'PAD_END'])
        # for each label in original sentence, create fiveGram
        for i, word in enumerate(sentenceWords):
            if 2 <= i <= len(sentenceWords) - 3:
                fiveGrams.append(([sentenceWords[i - 2], sentenceWords[i - 1], sentenceWords[i], sentenceWords[i + 1], sentenceWords[i + 2]], vocabL[sentenceLabels[i-2]]))

    # encode words to wordNums
    fiveGramsWordNums = []
    for fivegram in fiveGrams:
        fiveGramsWordNums.append((torchi([vocabW[word] if word in vocabW else vocabW['DEFAULT'] for word in fivegram[0]])
                                  , torch.tensor(fivegram[1], dtype=torch.long)))
    return fiveGramsWordNums


# Returns label of given labelNum
def labelNum2Label(labelNum, vocabL):
    return list(vocabL.keys())[list(vocabL.values()).index(labelNum)]


# Create five-gram of five words
def createFiveGramsTest(textWords, vocabW):
    # start with words
    fiveGrams = []
    for sentenceWords in textWords:
        # for each sentence (list of words): add 2 padding at both sides
        sentenceWords[:0] = ['PAD_BEGIN', 'PAD_BEGIN']
        sentenceWords.extend(['PAD_END', 'PAD_END'])
        # for each label in original sentence, create fiveGram
        for i, word in enumerate(sentenceWords):
            if 2 <= i <= len(sentenceWords) - 3:
                fiveGrams.append([sentenceWords[i - 2], sentenceWords[i - 1], sentenceWords[i], sentenceWords[i + 1], sentenceWords[i + 2]])

    # encode words to wordNums
    fiveGramsWordNums = []
    for fivegram in fiveGrams:
        fiveGramsWordNums.append(torchi([vocabW[word] if word in vocabW else vocabW['DEFAULT'] for word in fivegram]))
    return fiveGramsWordNums


# load test data, write test result
def test(partNum):
    # get test data
    testData = readTestData()
    testFiveGrams = createFiveGramsTest(testData, vocabW)
    f = open("test" + partNum + "." + str(sys.argv[1]), 'w')
    with torch.no_grad():
        for contextWords in testFiveGrams:
            # prediction
            output = modely(contextWords)
            predLabelNum = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # write result
            f.write(labelNum2Label(predLabelNum, vocabL) + "\n")
            # new line at end of sentence
            if contextWords[-1] == vocabW['PAD_END'] and contextWords[-2] == vocabW['PAD_END']:
                f.write("\n")
    f.close()


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
def plotGraphs():
    plotMeasurement("Accuracy", accuracy_t, accuracy_v)
    plotMeasurement("Loss", losses_t, losses_v)


if __name__ == '__main__':
    # read train data, encode strings to numbers, create vocabs from word to encoding number,
    # create fiveGrams of 4 context labels and one mid label
    trainData = readLabeledData("train")
    vocabW, vocabL = createVocab(trainData[0], "words"), createVocab(trainData[1], "labels")
    trainFiveGrams = createFiveGrams(trainData[0], trainData[1], vocabW, vocabL)
    # get dev data
    devData = readLabeledData("dev", vocabL)
    devFiveGrams = createFiveGrams(devData[0], devData[1], vocabW, vocabL)
    # init model
    modely = MyModel()
    optimizer = torch.optim.Adam(modely.parameters(), lr=lr)

    # do train and validation
    accuracy_t = []
    losses_t = []
    accuracy_v = []
    losses_v = []
    for i in range(epochs):
        # do validation and train, save results and accuracy
        curAccuracy_v, curLoss_v = validation(devFiveGrams)
        accuracy_v.append(curAccuracy_v)
        losses_v.append(curLoss_v)
        curAccuracy_t, curLoss_t = validation(trainFiveGrams)
        accuracy_t.append(curAccuracy_t)
        losses_t.append(curLoss_t)
        train(trainFiveGrams)
        #print(curAccuracy_v)

    plotGraphs()
    test(parts[-1][-1])
