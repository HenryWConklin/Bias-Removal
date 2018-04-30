import parse_data


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

def doReplace(grams, cts):
    igrams, ograms = grams
    res = []
    for x in igrams:
        x = ' '.join(x)
        if x in cts:
            bestRep = max(cts[x], key=lambda y: cts[x][y])
            if sum(cts[x].values()) > 30:
                res.append(bestRep.split(' '))
            else:
                res.append([x])
        else:
            res.append([x])
    return res



if __name__ == '__main__':
    N = 1
    toks = parse_data.readTokens('data/train.tsv.nopunc.tsv')

    cts = countPairs(*build_ngrams(*zip(*toks), N))
    for k in sorted(cts, key=lambda x: sum(cts[x].values())):
        reps = sorted(cts[k], key=lambda x: -cts[k][x])
        if reps[0] != k:
            print(k, [(x, cts[k][x]) for x in reps[:20]])

    testToks = parse_data.readTokens('data/train.tsv.nopunc.tsv')
    igrams, ograms = build_ngrams(*zip(*testToks), N)
    res = doReplace((igrams,ograms), cts)
    ncorr = 0
    nnothing = 0
    for x,y,o in zip(res, ograms, igrams):
        if x == y:
            ncorr += 1
        if o == y:
            nnothing += 1
    print(' '.join(map(lambda x: '-'.join(x), res)))
    print(ncorr/len(res))
    print(nnothing/len(res))



