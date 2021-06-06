import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from math import sqrt
from tagger1 import torchi, createVocab, readLabeledData, readTestData, allChars, labelNum2Label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 1
seperator = '\t' if sys.argv[1] == "ner" else ' '
parts = [sys.argv[i] for i in range(2,len(sys.argv),1)]
# constants
winSize = 5
embDimW = 50
embDimC = 30
loss_f = CrossEntropyLoss()
max_norm = 2
# hyper parameters of model
hidden_dim = 128
lr = 0.002
batchSize = 1000
filtersNum = 200
wordLen = 20



def createPreEmb():
    # create random weights for all words
    weights = torch.rand((len(vocabW), embDimW), dtype=torch.float32)
    # part 3 - pre-trained vectors:
    if "part3" in parts:
        # read the pre-trained
        preEmbs = np.loadtxt("wordVectors.txt")
        preWords = open("vocab.txt", "r").read().lower().split('\n')
        preWords.remove('')
        preWord2preEmb = {preWord: preEmb for preWord, preEmb in zip(preWords, preEmbs)}
        # for each word in vocabW, if we have pre-trained vector for it, put it instead of the random one
        for i in range(len(vocabW)):
            word = list(vocabW.keys())[list(vocabW.values()).index(i)]
            if word in preWords:
                weights[i] = torch.FloatTensor(preWord2preEmb[word])
    # part 4 - prefix and suffix:
    if "part4" in parts:
        # create dict from word in vocabW to its prefix and suffix, and dicts of prefixes and suffixes with their random vectors of embeddings
        # when we encounter same prefix twice, the second random vector will replace the first, that's OK
        word2presuff = {word: (word[:3], word[-3:]) for word in list(vocabW.keys())}
        pref2emb = {word[:3]: torch.rand(embDimW, dtype=torch.float32) for word in list(vocabW.keys())}
        suff2emb = {word[-3:]: torch.rand(embDimW, dtype=torch.float32) for word in list(vocabW.keys())}
        # update weights
        for i, word in enumerate(list(vocabW.keys())):
            weights[i] += pref2emb[word2presuff[word][0]] + suff2emb[word2presuff[word][1]]
    return weights


# fiveGrams are 5 window words, and one mid labelNum
def createFiveGrams(textWords, textLabels):
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
    return fiveGrams

def createFiveGramsTest(textWords):
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
    return fiveGrams


def fiveGramsChars(contextW):
    fiveGramsCharNums = torch.zeros((len(contextW), wordLen), dtype=torch.long)
    for i,word in enumerate(contextW):
        curChars = [word[i] if i < len(word) else 'PAD_CHAR' for i in range(wordLen)]
        fiveGramsCharNums[i] = torchi([vocabC[char] if char in vocabC else vocabC['DEFAULT'] for char in curChars])
    return fiveGramsCharNums


def fiveGramsWords(contextW):
    return torchi([vocabW[word] if word in vocabW else vocabW['DEFAULT'] for word in contextW])



class MyDataset(Dataset):
    def __init__(self, fiveGrams):
        self.labels = [fiveGram[1] for fiveGram in fiveGrams]
        cotextWs = [fiveGram[0] for fiveGram in fiveGrams]
        self.words = [fiveGramsWords(contextW) for contextW in cotextWs]
        self.chars = [fiveGramsChars(contextW) for contextW in cotextWs]

    def __getitem__(self, ind):
        return (self.words[ind], self.chars[ind]), self.labels[ind]

    def __len__(self):
        return len(self.labels)


class MyDatasetTest(Dataset):
    def __init__(self, fiveGrams):
        self.words = [fiveGramsWords(contextW) for contextW in fiveGrams]
        self.chars = [fiveGramsChars(contextW) for contextW in fiveGrams]

    def __getitem__(self, ind):
        return self.words[ind], self.chars[ind]

    def __len__(self):
        return len(self.words)


def createEmb5():
    # find all chars and assign random vecs to them
    chars = list(vocabC.keys())
    limit = sqrt(3) / 30
    char2vec = {char: (-2 * limit) * torch.rand(30, dtype=torch.float32) + limit for char in chars}
    weights = torch.empty((len(chars), embDimC), dtype=torch.float32)
    for i in range(len(chars)):
        weights[i] = torch.FloatTensor(char2vec[chars[i]])
    return weights


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        # maps each label to an embedding_dim vector
        self.embeddings = nn.Embedding(len(vocabW), embDimW, norm_type=2, max_norm=max_norm).requires_grad_(True)
        # use pre-trained embedding
        if "part3" in parts or "part4" in parts:
            self.embeddings = nn.Embedding.from_pretrained(createPreEmb(), norm_type=2, max_norm=max_norm).requires_grad_(True)
        self.fc1 = nn.Linear(embDimW * winSize, hidden_dim)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, len(vocabL))
        self.dropout2 = nn.Dropout()

    def forward(self, x):
        x = self.embeddings(x).view((-1, winSize * embDimW))
        x = self.dropout1(x)
        x = torch.tanh(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Part5Model(MyModel):
    def __init__(self):
        super(Part5Model, self).__init__()
        self.embeddings_char = nn.Embedding.from_pretrained(createEmb5(), norm_type=2, max_norm=max_norm).requires_grad_(True)
        self.dropout0 = nn.Dropout()
        self.conv0 = nn.Conv1d(in_channels=embDimC, out_channels=filtersNum, kernel_size=(3,), padding=(2,))
        self.leaky0 = nn.LeakyReLU()
        self.maxPool0 = nn.MaxPool1d(kernel_size=(wordLen,))
        self.fc1 = nn.Linear((embDimW+embDimC) * winSize, hidden_dim)

    # given x is batch of wordNum * winSize
    def forward(self, x):
        words, chars = x                                # (batchSize,winSize) , (batchSize,winSize,wordLen)
        # chars
        chars = self.embeddings_char(chars)             # (batchSize, winSize, wordLen, embDimC)
        chars = torch.transpose(chars, 1,3)
        chars = torch.transpose(chars, 2,3)             # (batchSize, embDimC, winSize, wordLen)
        chars = torch.reshape(chars, (-1, embDimC, winSize*wordLen))     # (batchSize, embDimC, winSize*wordLen)
        chars = self.conv0(chars)                        # (batchSize, embDimC, winSize*wordLen+2)
        chars = self.maxPool0(chars)                     # (batchSize, embDimC, winSize)
        chars = torch.transpose(chars, 1,2)
        # words
        words = self.embeddings(words)                   # (batchSize, winSize, embDimW)
        # together
        x = torch.cat((words, chars), 2)                 # (batchSize, winSize, embDimW+embDimC)
        x = torch.reshape(x,(-1,winSize*(embDimW+embDimC)))              # (batchSize, winSize*(embDimW+embDimC))
        x = self.dropout1(x)
        x = torch.tanh(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
        #return F.log_softmax(x, dim=1)


# train model. return loss of training
def train(fiveGrams):
    modely.train()
    trainLoader = DataLoader(MyDataset(fiveGrams), batch_size=batchSize, shuffle=True)
    for contextWords, midLabel in trainLoader:
        optimizer.zero_grad()
        output = modely(contextWords)
        loss = loss_f(output, midLabel)
        loss.backward()
        optimizer.step()


def validation(fiveGrams):
    modely.eval()
    correctTotal = 0
    lossVal = 0
    devLoader = DataLoader(MyDataset(fiveGrams), batch_size=batchSize, shuffle=True)
    with torch.no_grad():
        for context, midLabel in devLoader:
            output = modely(context)
            predLabelNum = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correctTotal += predLabelNum.eq(midLabel.view_as(predLabelNum)).cpu().sum().item()
            loss = loss_f(output, midLabel)
            lossVal += loss.item()
        accuracy = correctTotal / len(fiveGrams)
        lossVal /= len(devLoader)
    return accuracy, lossVal


# load test data, write test result
def test():
    # get test data
    testData = readTestData()
    testFiveGrams = createFiveGramsTest(testData)
    testLoader = DataLoader(MyDatasetTest(testFiveGrams))
    f = open("test5." + str(sys.argv[1]), 'w')
    with torch.no_grad():
        for contextWords in testLoader:
            # prediction
            output = modely(contextWords)
            predLabelNum = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # write result
            f.write(labelNum2Label(predLabelNum, vocabL) + "\n")
            # new line after end of sentence
            if contextWords[0][0][-1].item() == vocabW['PAD_END'] and contextWords[0][0][-2].item() == vocabW['PAD_END']:
                f.write("\n")
    f.close()



def writeResult(contextBatch, predLabelNumsBatch, file):
    for context, predLabelNum in zip(contextBatch,predLabelNumsBatch):
        file.write(labelNum2Label[predLabelNum.item()] + "\n")
        if context[-1] == 'PAD_END' and context[-2] == 'PAD_END':
            file.write("\n")




def plotMeasurement(name, trainData, devData, epochos):
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
    plotMeasurement("Accuracy", accuracy_t, accuracy_v, epochs)
    plotMeasurement("Loss", losses_t, losses_v, epochs)


if __name__ == '__main__':
    # read train data, encode strings to numbers, create vocabs from word to encoding number,
    # create fiveGrams of 4 context labels and one mid label
    trainData = readLabeledData("train")
    vocabW, vocabL = createVocab(trainData[0], "words"), createVocab(trainData[1], "labels")
    trainFiveGrams = createFiveGrams(trainData[0], trainData[1])
    # get dev data
    devData = readLabeledData("dev", vocabL)
    devFiveGrams = createFiveGrams(devData[0], devData[1])

    # build dict from char to vector of uniform distribution in range (-limit,limit)
    charsList = allChars(list(vocabW.keys()))
    vocabC = {char: i for i,char in enumerate(charsList)}
    vocabC['DEFAULT'] = len(vocabC)
    vocabC['PAD_CHAR'] = len(vocabC)

    # init model
    modely = Part5Model()
    optimizer = torch.optim.Adam(modely.parameters(), lr=lr)

    # do train and validation
    accuracy_t = [];
    losses_t = []
    accuracy_v = [];
    losses_v = []
    for i in range(epochs):
        curAccuracy_v, curLoss_v = validation(devFiveGrams)
        accuracy_v.append(curAccuracy_v);
        losses_v.append(curLoss_v)
        curAccuracy_t, curLoss_t = validation(trainFiveGrams)
        #print(curAccuracy_v)
        accuracy_t.append(curAccuracy_t);
        losses_t.append(curLoss_t)
        train(trainFiveGrams)

    plotGraphs()
    test()
