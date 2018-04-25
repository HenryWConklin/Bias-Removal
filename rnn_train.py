from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Softmax
import numpy as np
import parse_data

charSet = parse_data.getCharSet() | {'[', ']'}
charMap = {c:i for i,c in enumerate(charSet)}
invCharMap = {i:c for i,c in enumerate(charSet)}

N_CHARS = len(charSet)

def build_model():
    encoder_input = Input(shape=(None, N_CHARS))
    encoder = LSTM(256, return_state=True)
    encoder_output, state_h, state_c = encoder(encoder_input)

    decoder_inputs = Input(shape=(None, N_CHARS))
    decoder = LSTM(256, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=[state_h, state_c])
    decoder_dense = Dense(N_CHARS, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)


    model = Model([encoder_input, decoder_inputs], decoder_outputs)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model

def buildData(path):
    data = parse_data.read(path)
    in_data, out_data = zip(*data)
    # square brackets won't appear normally, use as start/end chars
    out_data = ['[' + s + ']' for s in out_data]

    max_in_len = max([len(s) for s in in_data])
    max_out_len = max([len(s) for s in out_data])

    in_vecs = np.zeros((len(in_data), max_in_len, N_CHARS), dtype=np.float32)
    out_state_vecs = np.zeros((len(out_data), max_out_len, N_CHARS), dtype=np.float32)
    out_vecs = np.zeros((len(out_data), max_out_len, N_CHARS), dtype=np.float32)

    for i,d in enumerate(in_data):
        for t,c in enumerate(d):
            # one-hot embedding
            in_vecs[i][t][charMap[c]] = 1
    for i,d in enumerate(out_data):
        for t,c in enumerate(d):
            # one-hot embedding
            out_state_vecs[i][t][charMap[c]] = 1
            if t > 0:
                # shift output back by one, gets all of the history outputs next char
                out_vecs[i][t-1][charMap[c]] = 1

    return in_vecs, out_state_vecs, out_vecs

def trainModel(path):
    in_vecs, out_state_vecs, out_vecs = buildData(path)
    model = build_model()
    model.fit([in_vecs, out_state_vecs], out_vecs, batch_size=32, epochs=10, validation_split=0.1)
    model.save('rnn.h5')


if __name__ == '__main__':
    trainModel('data/train.tsv')

