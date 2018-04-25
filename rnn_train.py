from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Softmax, RepeatVector
from keras.callbacks import TensorBoard
import numpy as np
import parse_data

charSet = parse_data.getCharSet() | {'[', ']'}
charMap = {c:i for i,c in enumerate(charSet)}
invCharMap = {i:c for i,c in enumerate(charSet)}

N_CHARS = len(charSet)

def build_model(max_time):
    encoder_input = Input(shape=(max_time, N_CHARS), name='encoder-in')
    encoder = LSTM(128, return_sequences=True)(encoder_input)
    encoder = LSTM(128, return_sequences=True)(encoder)

    # decoder_in = RepeatVector(max_time)(encoder)
    decoder = LSTM(128, return_sequences=True)(encoder)
    decoder_output = TimeDistributed(Dense(N_CHARS, activation='softmax'))(decoder)

    model = Model(encoder_input, decoder_output)

    return model

def oneHotTransform(docs, docLen):
    vecs = np.zeros((len(docs), docLen, N_CHARS))
    for i,d in enumerate(docs):
        for t,c in enumerate(d):
            # one-hot embedding
            vecs[i][t][charMap[c]] = 1
    return vecs


def buildData(path, maxrows=-1):
    data = parse_data.read(path)
    if maxrows > 0:
        data = data[:maxrows]
    in_data, out_data = zip(*data)
    # square brackets won't appear normally, use as start/end chars
    out_data = ['[' + s + ']' for s in out_data]

    max_in_len = max([len(s) for s in in_data])
    max_out_len = max([len(s) for s in out_data])
    max_len = max(max_in_len, max_out_len)

    in_vecs = oneHotTransform(in_data, max_len)
    out_vecs = oneHotTransform(out_data, max_len)

    return in_vecs, out_vecs, max_len

def trainModel(path, model=None):
    tb = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True, write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None)
    in_vecs, out_vecs, maxlen = buildData(path)
    if model is None:
        model = build_model(maxlen)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit(in_vecs, out_vecs, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tb])
    model.save('rnn.h5')


if __name__ == '__main__':
    trainModel('data/train.tsv')

