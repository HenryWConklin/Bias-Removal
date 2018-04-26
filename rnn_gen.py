from keras.models import Model, load_model
import numpy as np
import rnn_train
import sys
import parse_data

def apply(model, text):
    text = parse_data.tokenize(text)
    enc_text = rnn_train.wordIndexTransform([text], [len(text)+100])
    out = model.predict(enc_text)
    return trunc(invOneHot(out))

def invOneHot(out):
    outS = ''
    for x in out:
        cInd = np.argmax(x)
        outS = outS + ' ' + rnn_train.invWordMap[cInd]
    return outS[1:]

def trunc(s):
    return s[:s.find('END')].strip()

if __name__ == '__main__':
    model = load_model('rnn.h5')
    # data = parse_data.read('data/test.tsv.nopunc.tsv')
    # data = filter(lambda x: len(x) == 2, data)
    # for before, after in data:
    #         before_enc = rnn_train.indexTransform([before], [len(before) + 100])
    #         out = model.predict(before_enc)
    #         out_dec = trunc(invOneHot(out))
    #         if out_dec != before:
    #             print(out_dec)
    for line in sys.stdin:
        print(apply(model, line.strip()))

