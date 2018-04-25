from keras.models import Model, load_model
import numpy as np
import rnn_train
import sys

def apply(model, text):
    enc_text = rnn_train.oneHotTransform([text], model.layers[0].input_shape[1])
    out = model.predict(enc_text)
    outS = ''
    for x in out[0]:
        cInd = np.argmax(x)
        outS = outS + rnn_train.invCharMap[cInd]
    return outS

if __name__ == '__main__':
    model = load_model('rnn.h5')
    for line in sys.stdin:
        print(apply(model, line.strip()))

