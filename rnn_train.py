from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Softmax, RepeatVector, Bidirectional, Embedding,Dropout
from keras.callbacks import TensorBoard
import numpy as np
import parse_data

charSet = ['.'] + parse_data.getCharSet()
charMap = {c:i for i,c in enumerate(charSet)}
invCharMap = {i:c for i,c in enumerate(charSet)}

N_CHARS = len(charSet)


def build_bidirectional_model():
    model = Sequential()
    model.add(Embedding(N_CHARS, 128))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(N_CHARS, activation='softmax')))

    return model


def build_forward_model(max_time):
    encoder_input = Input(shape=(max_time, N_CHARS), name='encoder-in')
    encoder = LSTM(128, return_sequences=True)(encoder_input)
    encoder = LSTM(128, return_sequences=True)(encoder)

    # decoder_in = RepeatVector(max_time)(encoder)
    decoder = LSTM(128, return_sequences=True)(encoder)
    decoder_output = TimeDistributed(Dense(N_CHARS, activation='softmax'))(decoder)

    model = Model(encoder_input, decoder_output)

    return model

def oneHotTransform(docs, pad_to):
    vecs = [np.zeros((max(len(d), pad), N_CHARS)) for d,pad in zip(docs, pad_to)]
    for i,d in enumerate(docs):
        for t,c in enumerate(d):
            # one-hot embedding
            vecs[i][t][charMap[c]] = 1
        for t in range(vecs[i].shape[0] - len(d)):
            # Pad with zeros
            vecs[i][t+len(d)][0] = 1
    return vecs

def indexTransform(docs, pad_to):
    vecs = [np.zeros(max(len(d), pad)) for d, pad in zip(docs, pad_to)]
    for i,d in enumerate(docs):
        for t,c in enumerate(d):
            vecs[i][t] = charMap[c]
    return vecs

def buildData(path, maxrows=-1):
    data = parse_data.read(path)
    if maxrows > 0:
        data = data[:maxrows]
    in_data, out_data = zip(*data)

    pad_len = [((max(len(d1), len(d2)) // 100 + 1) * 100) for d1,d2 in zip(in_data, out_data)]

    in_vecs = indexTransform(in_data, pad_len)
    out_vecs = oneHotTransform(out_data, pad_len)

    # Bin into gropus by length, in intervals of 100. Hopefully helps with runtime and weird behavior wrt padding
    buckets = [list(filter(lambda x: i*100 < x[0].shape[0] <= (i+1)*100, zip(in_vecs, out_vecs))) for i in range(5)]
    buckets = [list(zip(*x)) for x in buckets]
    buckets = [(np.array(x), np.array(y)) for x,y in buckets]
    return buckets

def trainModel(path, model=None, epochs=3):
    buckets = buildData(path)

    if model is None:
        model = build_bidirectional_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    for i in range(epochs):
        for in_vecs, out_vecs in buckets:
            model.fit(in_vecs, out_vecs, batch_size=64, epochs=1, validation_split=0.1)
    model.save('rnn.h5')
    return model


if __name__ == '__main__':
    model = load_model('rnn.h5')
    for i in range(30):
        model = trainModel('data/train.tsv.nopunc.tsv', model, 3)



