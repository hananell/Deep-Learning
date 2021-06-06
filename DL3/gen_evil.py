import random
from random import randint
from math import floor, ceil

LETTERS = ('a', 'b', 'c', 'd', '1')


# Returns list that is good or bad sample
def generatePalindrome(kind):
    # initialize empty list/sample of random size
    sampleLen = randint(1, 40)
    sample = ['empty'] * sampleLen
    # put values to make polyndrom
    for i in range(ceil(sampleLen / 2)):
        sample[i] = sample[-i - 1] = random.choice(LETTERS)
    # if neg - change one random letter
    if kind == "neg":
        changeInd = randint(0, sampleLen - 1)
        sample[changeInd] = random.choice([letter for letter in LETTERS if letter != sample[changeInd]])
    return sample


def generate10NoA(kind):
    # Chek if 'a' exist in first 10 letters
    def isPos(sample):
        for letter in sample[:10]:
            if letter == 'a':
                return True
        return False

    # initialize random sample
    sampleLen = randint(1, 40)
    sample = [random.choice(LETTERS) for i in range(sampleLen)]
    # neg - replace all 'a' at first 10 indexes with another characters
    if kind == "neg":
        for i in range(min(10, sampleLen)):
            if sample[i] == 'a':
                sample[i] = random.choice([letter for letter in LETTERS if letter != 'a'])
    # pos - make sure we have 'a' in first 10 indexes
    else:
        while not isPos(sample):
            sample = [random.choice(LETTERS) for i in range(sampleLen)]
    return sample


def generateQuarterNoA(kind):
    # Chek if 'a' exist in first 10 letters
    def isPos(sample):
        for letter in sample[:ceil(len(sample) / 4)]:
            if letter == 'a':
                return True
        return False

    # initialize random sample
    sampleLen = randint(1, 40)
    sample = [random.choice(LETTERS) for i in range(sampleLen)]
    # neg - replace all 'a' at first 10 indexes with another characters
    if kind == "neg":
        for i in range(ceil(len(sample) / 4)):
            if sample[i] == 'a':
                sample[i] = random.choice([letter for letter in LETTERS if letter != 'a'])
    # pos - make sure we have 'a' in first 10 indexes
    else:
        while not isPos(sample):
            sample = [random.choice(LETTERS) for i in range(sampleLen)]
    return sample


# Pos is with '1' at the start and end. experiment for part 3, to decide how safe it is to pad sentences
def generatePadded(kind):
    sampleLen = randint(1, 40)
    sample = [random.choice([l for l in LETTERS if l != '1']) for i in range(sampleLen)]
    if kind == "pos":
        sample.insert(0,'1')
        sample.append('1')
    return sample


# Returns sets of samples for training and testing a model
def modelData_evil(method):
    # define data generator
    func = None
    if method == "Palindrome":
        func = generatePalindrome
    elif method == "10NoA":
        func = generate10NoA
    elif method == "QuarterNoA":
        func = generateQuarterNoA
    elif method == "padded":
        func = generatePadded
    else:
        raise Exception("not a method name. correct names: Palindrome, 10NoA, QuarterNoA")

    # generate data
    train_pos = [func("pos") for _ in range(500)]
    train_neg = [func("neg") for _ in range(500)]
    test_pos = [func("pos") for _ in range(100)]
    test_neg = [func("neg") for _ in range(100)]
    return train_pos, train_neg, test_pos, test_neg
