import sys
import string

if __name__ == '__main__':
    with open(sys.argv[1]) as inFile:
        for line in inFile:
            line = line.split('\t')
            # Filter out punctuation
            line = map(lambda s: ''.join(map(lambda c: ' ' if c in string.punctuation else c, s)), line)
            for i in range(10):
                line = map(lambda s: s.replace('  ', ' ').strip(), line)
            line = list(line)
            if line[0] != line[1]:
                print("\t".join(line))
