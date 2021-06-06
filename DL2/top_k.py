import numpy as np
from scipy import spatial


# Return similarity between two given vectors
def similarity(a,b):
    return 1 - spatial.distance.cosine(a, b)


# Return k most similar words relative to given word
def most_similar(word, k):
    # calculate similarities to all words
    similarity2word = {similarity(word2emb[word], word2emb[other]):other for other in words}
    # sort and take first k - highest similarity
    max_similarities = list(sorted(similarity2word.keys(), reverse=True))[:k]
    # return each word of high similarity
    return [(similarity2word[high_similarity], high_similarity) for high_similarity in max_similarities]



if __name__ == '__main__':
    vecs = np.loadtxt("wordVectors.txt")
    words = open("vocab.txt", "r").read().split('\n')
    words.remove('')
    word2emb = {word: emb for word, emb in zip(words, vecs)}
    for word in ['dog', 'england', 'john', 'explode', 'office']:
        print(word, most_similar(word, 5))