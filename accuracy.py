# Computes accuracy between predicted and ground truth sentences

import sys
import difflib
import parse_data
if len(sys.argv) != 3:
    print("Usage: accuracy.py <pred> <gt>")

diffacc = 0
pairacc = 0
totSents = 0
totWords = 0
with open(sys.argv[1]) as pred:
    with open(sys.argv[2]) as gt:
        for pl, gl in zip(pred, gt):
            pl = parse_data.tokenize(pl)
            gl = parse_data.tokenize(gl)
            diffacc += difflib.SequenceMatcher(None, pl, gl).ratio()
            totSents += 1

            padlen = max(len(pl), len(gl))
            pl = pl + ['NULL' for i in range(padlen - len(pl))]
            gl = gl + ['NULL' for i in range(padlen - len(gl))]
            for wp, wg in zip(pl,gl):
                if wp == wg:
                    pairacc += 1
                totWords += 1

print("Diff Accuracy: {:f}".format(diffacc/totSents))
print("Pair Accuracy: {:f}".format(pairacc/totWords))
