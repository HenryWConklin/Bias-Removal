
def read(path):
    with open(path) as inFile:
        data = [s.encode('ascii', errors='ignore').decode().strip().split('\t') for s in inFile]
    return data

def getCharSet():
    with open('charset.txt') as inFile:
        return set(inFile.read().split('\n'))

if __name__ == '__main__':
    charSet = set()
    for x in read('data/train.tsv'):
        print(x[0])
        print(x[1])
        print()
        for c in x[0]:
            charSet.add(c)
        for c in x[1]:
            charSet.add(c)
    print(sorted(charSet))
    print(len(charSet))
    with open('charset.txt', 'w') as outFile:
        for c in sorted(charSet):
            outFile.write(c+'\n')

