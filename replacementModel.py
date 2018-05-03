# Transform given text through the replacement model

import ngram_replace
import parse_data
import pickle
import sys

if len(sys.argv) != 2:
    print("Usage: replacementModel.py <modelPath>")
    sys.exit(1)

with open(sys.argv[1], 'rb') as inFile:
    model = pickle.load(inFile)

for line in sys.stdin:
    line = parse_data.tokenize(line.strip())
    res = ngram_replace.doReplaceFreqReplace([line], model, 5)
    print(' '.join(res[0]))