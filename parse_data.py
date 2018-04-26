import re

def read(path):
    with open(path) as inFile:
        data = [s.encode('ascii', errors='ignore').decode().strip().split('\t') for s in inFile]
    return list(filter(lambda x: len(x) == 2, data))

def getCharSet():
    with open('charset.txt') as inFile:
        return inFile.read().split('\n')

num_re = re.compile(r'[0-9]+')
def tokenize(d):
    d = d.split(" ")
    d = map(lambda s: s.lower(), d)
    d = filter(lambda s: s != '', d)
    d = map(lambda s: 'NUM' if num_re.match(s) else s, d)
    return list(d)

def getVocab():
    with open('vocab.txt') as inFile:
        return inFile.read().split('\n')

def readTokens(path):
    return [[tokenize(s) for s in d] for d in read(path)]


if __name__ == '__main__':
    charSet = set()
    vocab = {}
    data = read('data/train.tsv.nopunc.tsv')
    for x in data:
        for c in x[0]:
            charSet.add(c)
        for c in x[1]:
            charSet.add(c)
        for w in tokenize(x[0]):
            vocab[w] = vocab.get(w,0) + 1
        for w in tokenize(x[1]):
            vocab[w] = vocab.get(w,0) + 1

    print(sorted(charSet))
    print(len(charSet))
    vocab = sorted(filter(lambda x: vocab[x] > 40, vocab), key = lambda x: -vocab[x])
    print(len(vocab))
    with open('charset.txt', 'w') as outFile:
        for c in sorted(charSet):
            outFile.write(c+'\n')

    with open('vocab.txt', 'w') as outFile:
        for w in vocab:
            outFile.write(w + '\n')
