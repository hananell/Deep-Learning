from random import randint

POSITIVE = ('n', 'a', 'n', 'b', 'n', 'c', 'n', 'd', 'n')
NEGATIVE = ('n', 'a', 'n', 'c', 'n', 'b', 'n', 'd', 'n')
characterLength = 5


# Returns list that is good or bad sample
def generateSamlpe(kind):
    # inner func - return list of given letter, with 1-5 occurrences
    def generateCharacter(letter):
        if letter == 'n':
            return [str(randint(1, 9)) for _ in range(randint(1, characterLength))]
        else:
            return [letter for _ in range(randint(1, characterLength))]

    # initiate sample, and append letter representation for each letter
    sample = []
    if kind == "pos":
        [sample.extend(generateCharacter(letter)) for letter in POSITIVE]
    elif kind == "neg":
        [sample.extend(generateCharacter(letter)) for letter in NEGATIVE]
    return sample


# write 500 good or bad samples to file
def writeSamples(kind):
    samples = [generateSamlpe(kind) for _ in range(500)]
    f = open(kind + "_examples", 'w')
    for sample in samples:
        [f.write(letter) for letter in sample]
        f.write('\n')
    f.close()


if __name__ == "__main__":
    writeSamples("pos")
    writeSamples("neg")
