import sys
import torch
from torch import nn
from utils import *

# constants
seperator = '\t' if sys.argv[2] == "ner" or sys.argv[3] == "ner" else ' '
loss_f = nn.CrossEntropyLoss()
# hyper parameters
lr = 0.001
epochs = 5
emb_dim = 50
hid_dim = 128


# Generates word representation, given as list of chars
class bRepresenter(nn.Module):
    def __init__(self, vocabCLength):
        super(bRepresenter, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabCLength, embedding_dim=emb_dim, norm_type=2, max_norm=2)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim)

    def forward(self, x):  # (sampleLen,)
        emb = self.embedding(x)  # (sampleLen, embDim)
        hn, _ = self.lstm(emb.view(len(x), 1, -1))  # (sampleLen, batchSize=1, hidDim)
        outB = hn[-1][0]  # (hidDim)
        return outB


class MyBilstm(nn.Module):
    def __init__(self):
        super(MyBilstm, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(self.createEmb(), norm_type=2, max_norm=2).requires_grad_(True)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hid_dim * 2, len(vocabL.values()))

    def forward(self, x):  # (sampleLen,)
        emb = self.embedding(x)  # (sampleLen, embDim)
        hn, _ = self.lstm(emb.view(len(x), 1, -1))  # (sampleLen, batchSize, hidDim*2)
        hn = self.dropout(hn)
        hn = torch.squeeze(hn)  # (sampleLen, hidDim*2)
        out = self.fc(hn)  # (len(vocabL.values()), )
        if len(x) == 1:  # the output must has 2 dimensions for loss calculation
            out = out.reshape(1, -1)
        return out

    def createEmb(self):
        weights = torch.rand((len(vocabW), emb_dim), dtype=torch.float32)

        # doing b in function, to call it in d too
        def weightsToReprB():
            # replace weights[i] with output of lstmC on the chars of word_i
            vocabC = createVocabC(vocabW)
            lstmC = bRepresenter(len(vocabC))
            for i in range(len(weights)):
                word_i = list(vocabW.keys())[i]
                chars_i = list(word_i)
                chars_i_encoded = torch.tensor([vocabC[char] if char in vocabC else vocabC['UNKNOWN'] for char in chars_i], dtype=torch.long)
                weights[i] = lstmC(chars_i_encoded).data

        if repr == 'a':
            # do nothing - remain with random embedding
            pass

        elif repr == 'b':
            weightsToReprB()

        elif repr == 'c':
            # add to weights[i] the embedding of prefix and suffix of word_i
            # create dict from word in vocabW to its prefix and suffix, and dicts of prefixes and suffixes with their random vectors of embeddings. When we encounter same prefix twice, the second random vector will replace the first, that's OK
            word2presuff = {word: (word[:3], word[-3:]) for word in list(vocabW.keys())}
            pref2emb = {word[:3]: torch.rand(emb_dim, dtype=torch.float32) for word in list(vocabW.keys())}
            suff2emb = {word[-3:]: torch.rand(emb_dim, dtype=torch.float32) for word in list(vocabW.keys())}
            # update weights
            for i, word in enumerate(list(vocabW.keys())):
                weights[i] += pref2emb[word2presuff[word][0]] + suff2emb[word2presuff[word][1]]

        elif repr == 'd':
            # make new random embedding for a, update weights to b, then weights[i] is result of fc layer on a+b
            weightsToReprB()
            reprA = torch.rand((len(vocabW), emb_dim), dtype=torch.float32)
            fcLayer = nn.Linear(emb_dim * 2, emb_dim)
            for i in range(len(weights)):
                weights[i] = fcLayer(torch.cat((reprA[i], weights[i])))
        else:
            raise Exception("not valid repr")
        return weights


# Does train
def train():
    modely.train()
    for i, (sample, target) in enumerate(zip(trainW, trainL)):
        # do dev every 500 samples
        if i % 500 == 0:
            curAccuracy_v, curLoss_v = dev(devW, devL, devWordsNum)
            accuracy_v.append(curAccuracy_v)
            losses_v.append(curLoss_v)
            modely.train()
        # train
        optimizer.zero_grad()
        output = modely(sample)
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()


# does dev and returns loss and accuracy
def dev(sentencesW, sentencesL, totalLength):
    modely.eval()
    correct = 0
    lossVal = 0
    with torch.no_grad():
        for sentenceW, sentenceL in zip(sentencesW, sentencesL):
            # prediction
            preds = modely(sentenceW)
            # loss
            loss = loss_f(preds, sentenceL)
            lossVal += loss.item()
            # accuracy
            predLabelNums = preds.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += predLabelNums.eq(sentenceL.view_as(predLabelNums)).cpu().sum().item()
    accuracy = correct / totalLength
    lossVal /= totalLength
    return accuracy, lossVal


def writeResults():
    f = open(f"results_{repr}_{trainFile}", 'w')
    f.write("Accuracy:\n")
    for ac in accuracy_v:
        f.write(str(ac) + "\n")
    f.write("\n\nLoss:\n")
    for l in losses_v:
        f.write(str(l) + "\n")
    f.close()


if __name__ == "__main__":
    repr, trainFile, modelFile = sys.argv[1], sys.argv[2], sys.argv[3]
    # read  data and create vocabs
    trainW, trainL = readLabeledData(trainFile, "train")
    vocabW, vocabL = createVocab(trainW, "words"), createVocab(trainL, "labels")
    wordNum2word = {wordNum: word for word, wordNum in vocabW.items()}
    labelNum2label = {labelNum: label for label, labelNum in vocabL.items()}
    devW, devL = readLabeledData(trainFile, "dev", vocabL)
    # pad sentences at the edges, count total words in each file, encode data to tensors
    # padBeginEnd(trainW, trainL)
    # padBeginEnd(devW, devL)
    trainWordsNum = sum(1 for sentence in trainW for word in sentence)
    devWordsNum = sum(1 for sentence in devW for word in sentence)
    trainW, trainL, devW, devL = encodeText(trainW, vocabW), encodeText(trainL, vocabL), encodeText(devW, vocabW), encodeText(devL, vocabL)

    # init model
    modely = MyBilstm()
    optimizer = torch.optim.Adam(modely.parameters(), lr=lr)

    # do train, save results of dev in these lists
    accuracy_v = []
    losses_v = []
    for i in range(epochs):
        print(f"epoch {i}")
        train()

    # plotGraphs(accuracy_t, accuracy_v, losses_t, losses_v, f"{modelFile[:6]}_{trainFile}")
    torch.save(modely, modelFile)
    writeResults()
