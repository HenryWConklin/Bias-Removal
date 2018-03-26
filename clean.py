import sys

# Usage takes one argument, file name. Prints cleaned data to stdout
if __name__=="__main__":
    with open(sys.argv[1]) as inFile:
        for line in inFile:
            # 0:title, 1:rev no, 2:revision text contains {{NPOV}} tag, 3:edit comment contains "POV", 4:editor ID, 5:revision size {minor, major, unknown}, 6:string before, 7:string after, 8:sentence before, 9:sentence after
            cols = line.strip().split('\t')
            cols = [x.replace('[', '').replace(']','') for x in cols]
            if cols[8] != cols[9] and cols[3] == 'true':
                print('\t'.join(cols[8:]))



