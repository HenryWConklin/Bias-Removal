# Group lines from three files for comparision
# Used for comparing input, ground truth, and predicted outputs

import sys

if len(sys.argv) != 4:
    print("Usage: tridiff.py <f1> <f2> <f3>")
    print("Prints each line from the files in sequence, grouped together to standard out")

with open(sys.argv[1]) as f1:
    with open(sys.argv[2]) as f2:
        with open(sys.argv[3]) as f3:
            for l1, l2, l3 in zip(f1,f2,f3):
                print(l1.strip())
                print(l2.strip())
                print(l3.strip())
                print()