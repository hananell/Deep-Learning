from gen_examples import modelData
from gen_evil import modelData_evil
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import random
import matplotlib.pyplot as plt
import time
import sys
from math import ceil

startTime = time.time()

# constants
encDict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, '1': 4, '2': 4, '3': 4, '4': 4, '5': 4, '6': 4, '7': 4, '8': 4, '9': 4, 'PAD': 5}
# hyper parameters
epochs = 3
lr = 0.01
emb_dim = 50
hid_dim = 20


# Returns random merge of two given lists
def randomMerge(a, b):
    return random.sample(a + b, len(a) + len(b))


# Encode samples to numbers, and add target pred
def myEncode(samples, kind):
    # pad each encoded sample till maxLen, that is roughly max len of all samples. and cut longer samples
    def pad(sample):
        while len(sample) < maxLen:
            sample.append('PAD')
        return sample[:maxLen]

    padded = [pad(sample) for sample in samples]
    # encode each character to variable/tensor/serial number (to be embedded). all numbers chars mapped to same serial number as they have exactly same meaning
    encoded = [[encDict[char] for char in sample] for sample in padded]
    encoded = [Variable(torch.tensor(sample, dtype=torch.long)) for sample in encoded]
    # add target 1 for pos, and 0 for neg
    target0 = Variable(torch.tensor(0, dtype=torch.float32)).reshape(1)
    target1 = Variable(torch.tensor(1, dtype=torch.float32)).reshape(1)
    targeted = [(sample, target0) if kind == "neg" else (sample, target1) for sample in encoded]
    return targeted


class experimentModel(nn.Module):
    def __init__(self):
        super(experimentModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(set(encDict.values())), embedding_dim=emb_dim, padding_idx=len(set(encDict.values())) - 1)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=1)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):  # (sampleLen,)
        emb = self.embedding(x)  # (sampleLen, embDim)
        hn, _ = self.lstm(emb.view(len(x), 1, -1))  # (sampleLen, batchSize=1, hidDim)
        out = self.fc(hn[-1][-1])  # (1,)
        return torch.sigmoid(out)


# Does train
def train():
    modely.train()
    for sample, target in trainData:
        optimizer.zero_grad()
        output = modely(sample)
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()


# does dev and returns loss and accuracy
def dev(data):
    modely.eval()
    correct = 0
    lossVal = 0
    with torch.no_grad():
        for sample, target in data:
            output = modely(sample)
            loss = loss_f(output, target)
            lossVal += loss.item()
            if torch.round(output.data[0]) == target.data[0]:
                correct += 1
    accuracy = correct / len(data)
    lossVal /= len(data)
    return accuracy, lossVal


def plotMeasurement(name, trainMeasure, devMeasure):
    epochsList = [i for i in range(epochs)]
    plt.figure()
    plt.title(name)
    plt.plot(epochsList, trainMeasure, label="Train")
    plt.plot(epochsList, devMeasure, label="Dev")
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.locator_params(axis="x", integer=True, tight=True)  # make x axis to display only whole number (iterations)
    plt.legend()
    plt.savefig(f"{name}.png")


# Plot accuracies
def plotGraphs():
    plotMeasurement("Loss", losses_t, losses_v)
    plotMeasurement("Accuracy", accuracy_t, accuracy_v)


def test():
    _, _, test_pos, test_neg = modelData()
    test_pos, test_neg = myEncode(test_pos, "pos"), myEncode(test_neg, "neg")
    testData = randomMerge(test_pos, test_neg)
    accuracy_test, loss_test = dev(testData)
    print("\n")
    print(accuracy_test, loss_test)


if __name__ == "__main__":
    # prepare data
    train_pos, train_neg, dev_pos, dev_neg = modelData()
    # train_pos, train_neg, dev_pos, dev_neg = modelData_evil("padded")
    if len(sys.argv) > 1:
        train_pos, train_neg, dev_pos, dev_neg = modelData_evil(sys.argv[1])
    maxLen = ceil(max([len(sample) for sample in train_pos + train_neg]) * 0.8)

    train_pos, train_neg, dev_pos, dev_neg = myEncode(train_pos, "pos"), myEncode(train_neg, "neg"), myEncode(dev_pos, "pos"), myEncode(dev_neg, "neg")
    trainData = randomMerge(train_pos, train_neg)
    devData = randomMerge(dev_pos, dev_neg)
    # prepare model
    modely = experimentModel()
    loss_f = nn.BCELoss()
    optimizer = optim.Adam(modely.parameters(), lr=lr)
    output = modely(train_pos[0][0])

    # prepare lists for results
    losses_t = []; losses_v = []; accuracy_t = []; accuracy_v = []

    for _ in range(epochs):
        # dev on devData and trainData
        curAccuracy_v, curLoss_v = dev(devData)
        accuracy_v.append(curAccuracy_v); losses_v.append(curLoss_v)
        curAccuracy_t, curLoss_t = dev(trainData)
        accuracy_t.append(curAccuracy_t); losses_t.append(curLoss_t)
        # training
        train()
        print(curAccuracy_v)

    # test()
    plotGraphs()
