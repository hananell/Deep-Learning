import sys
from utils import *
from bilstmTrain import MyBilstm
import torch


# load test data, write test result
def test():
    # get test data
    testW = readTestData(inputFile)
    testWEnc = encodeText(testW, vocabW)
    testWrite = open("test4." + inputFile, 'w')
    with torch.no_grad():
        for sentenceWEnc, sentenceW in zip(testWEnc, testW):
            # prediction whole sentence
            output = modely(sentenceWEnc)
            predLabelNumsTensor = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            predLabelNums = [predLabelNumsTensor[i].item() for i in range(len(sentenceW))]
            # write whole sentence
            [testWrite.write(sentenceW[i] + " " + labelNum2label[predLabelNums[i]] + "\n") for i in range(len(sentenceW))]
            testWrite.write("\n")
    testWrite.close()


if __name__ == "__main__":
    repr, modelFile, inputFile = sys.argv[1], sys.argv[2], sys.argv[3]
    modely = torch.load(modelFile)
    trainW, trainL = readLabeledData(inputFile, "train")
    vocabW, vocabL = createVocab(trainW, "words"), createVocab(trainL, "labels")
    labelNum2label = {labelNum: label for label, labelNum in vocabL.items()}
    test()