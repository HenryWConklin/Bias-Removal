import parse_data
import difflib
import sys
import os
import pickle

def build_ngrams(inDocs, outDocs, n):
    igrams = []
    ograms = []
    for di, do in zip(inDocs, outDocs):
        if len(di) < len(do):
            di = di + ['NULL' for i in range(len(do) - len(di))]
        elif len(do) < len(di):
            do = do + ['NULL' for i in range(len(di) - len(do))]
        for i in range(len(di) - n + 1):
            igrams.append(di[i:i+n])
            ograms.append(do[i:i+n])

    return igrams, ograms

def countPairs(igrams, ograms):
    counts = {}
    for ig, og in zip(igrams, ograms):
        ig = ' '.join(ig)
        og = ' '.join(og)
        if ig not in counts:
            counts[ig] = {}

        counts[ig][og] = counts[ig].get(og, 0) + 1

    return counts

# Count common replacements
def countReplacements(toks):
    cts = {}
    i=0
    for di, do in toks:
        sys.stdout.write('\r{:d}'.format(i))
        i+=1
        diff = difflib.ndiff(di, do)

        prevWord = None
        addChain = []
        delChain = []
        for x in diff:
            prefix = x[0]
            text = x[2:]
            if prefix == ' ':
                addText = ' '.join(addChain)

                # If just an insertion, no deleted words
                # Treat it as a replacement of adjacent words with more text
                if len(addChain) > 0 and len(delChain) == 0:
                    if prevWord is not None:
                        if prevWord not in cts:
                            cts[prevWord] = {}
                        addBeforeText = prevWord + ' ' + addText
                        cts[prevWord][addBeforeText] = cts[prevWord].get(addBeforeText, 0) + 1

                    if text not in cts:
                        cts[text] = {}
                    addAfterText = addText + ' ' + text
                    cts[text][addAfterText] = cts[text].get(addAfterText, 0) + 1
                # Otherwise, it is a replacement if either chain is non-empty
                elif len(addChain) > 0 or len(delChain) > 0:
                    delText = ' '.join(delChain)
                    if delText not in cts:
                        cts[delText] = {}
                    cts[delText][addText] = cts[delText].get(addText, 0) + 1

                addChain = []
                delChain = []
                prevWord = text
            elif prefix == '+':
                addChain.append(text)
            elif prefix == '-':
                delChain.append(text)

        # A replacement at the end of the document
        addText = ' '.join(addChain)
        delText = ' '.join(delChain)
        if len(addChain) > 0 and len(delChain) == 0:
            if prevWord is not None:
                if prevWord not in cts:
                    cts[prevWord] = {}
                addBeforeText = prevWord + ' ' + addText
                cts[prevWord][addBeforeText] = cts[prevWord].get(addBeforeText, 0) + 1
        elif len(addChain) > 0 or len(delChain) > 0:
            if delText not in cts:
                cts[delText] = {}
            cts[delText][addText] = cts[delText].get(addText, 0) + 1

    print()
    return cts

def ngramTermFreq(toks, n=3):
    tf = {}
    it=0
    for di, do in toks:
        sys.stdout.write('\r{:d}'.format(it))
        it+=1
        for i in range(1, n+1):
            igrams, ograms = build_ngrams([di], [do], i)
            for w in igrams:
                w = ' '.join(w)
                tf[w] = tf.get(w,0) + 1
    print()
    return tf



def relativeReplaceFreq(toks, thresh=10):
    cts = countReplacements(toks)
    tf = ngramTermFreq(toks, 2)
    res = {w: {k: v/tf[w] for k,v in cts[w].items()} for w in cts if tf.get(w,0) > thresh}
    return res



def doReplace(grams, cts):
    igrams, ograms = grams
    res = []
    for x in igrams:
        x = ' '.join(x)
        if x in cts:
            bestRep = max(cts[x], key=lambda y: cts[x][y])
            if sum(cts[x].values()) > 30:
                res += bestRep.split(' ')
            else:
                res.append([x])
        else:
            res.append([x])
    return res

def doReplaceFreqReplace(docs, cts, n=3, thresh = 0.1):
    res = []
    cts = {w: max(cts[w].items(), key=lambda x: x[1]) for w in cts}
    for i,d in enumerate(docs):
        sys.stdout.write('\r{:d}'.format(i))
        rd = []
        j = 0
        while j < len(d):
            rep = False
            for k in range(1, n+1):
                if j+k <= len(d):
                    w = ' '.join(d[j:j+k])
                    if w in cts and cts[w][1] >= thresh:
                        rd += cts[w][0].split(' ')
                        j+=k
                        rep=True
                        break
            if not rep:
                rd.append(d[j])
                j+=1
        res.append(rd)
    print()
    return res


def termFreq(docs):
    cts = {}
    for di, do in docs:
        for w in di:
            cts[w] = cts.get(w,0) + 1
    return cts


def loadOrElse(path, func):
    if os.path.exists(path):
        with open(path, 'rb') as inFile:
            res = pickle.load(inFile)
    else:
        res = func()
        with open(path, 'wb') as outFile:
            pickle.dump(res, outFile)
    return res

if __name__ == '__main__':
    N = 5
    # toks = parse_data.readTokens('data/train.uniq.tsv')
    print('Reading data')
    toks = loadOrElse('toks.pkl', lambda: parse_data.readTokens('/home/henry/Downloads/npov-edits/5gram-edits-train.uniq.tsv'))

    # cts = countReplacements(toks)
    print('Doing counts')
    cts = loadOrElse('counts.pkl', lambda: countReplacements(toks))
    tf = loadOrElse('termFreq{:d}.pkl'.format(N), lambda: ngramTermFreq(toks, N))
    print(sorted(tf, key=lambda x: -tf[x])[:50])
    cts = {w: {k: v/tf[w] for k,v in cts[w].items()} for w in cts if tf.get(w,0) > 10}
    for k in sorted(cts, key=lambda x: max(cts[x].values()))[-20:]:
        reps = sorted(cts[k], key=lambda x: -cts[k][x])
        if reps[0] != k and reps[1] != 1:
            print(k, [(x, cts[k][x]) for x in reps[:20]])

    print("Doing replacements")
    testToks = parse_data.readTokens('data/dev.tsv.nopunc.tsv')
    res = doReplaceFreqReplace(map(lambda x: x[0], testToks), cts, N, 0.3)
    # res = doReplace((igrams,ograms), cts)
    print("Scoring")
    sumRep = 0
    sumNothing = 0
    tot = 0
    for r, t in zip(res, testToks):
        dr = r
        di, do = t
        # if di != dr:
        #     print(di)
        #     print(dr)
        #     print(do)
        #     print()
        sumRep += difflib.SequenceMatcher(None, dr, do).ratio()
        sumNothing += difflib.SequenceMatcher(None, di, do).ratio()


    print(sumRep/len(testToks))
    print(sumNothing/len(testToks))



