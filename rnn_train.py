from keras.models import Model, Sequential, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Softmax, RepeatVector, Bidirectional, Embedding,Dropout, Conv1D, MaxPool1D, GRU, Masking
from keras.callbacks import TensorBoard, LambdaCallback, ModelCheckpoint
import numpy as np
import parse_data
import rnn_gen
import gc

charSet = ['.'] + parse_data.getCharSet()
charMap = {c:i for i,c in enumerate(charSet)}
invCharMap = {i:c for i,c in enumerate(charSet)}

N_CHARS = len(charSet)

vocab = ['END', 'UNK'] + parse_data.getVocab()
wordMap = {c:i for i,c in enumerate(vocab)}
invWordMap = {i:c for i,c in enumerate(vocab)}
N_WORDS = len(vocab)

def build_conv_model():
    model = Sequential()
    model.add(Conv1D(256, 7, padding='same', activation='relu', input_shape=(None, N_CHARS)))
    model.add(Conv1D(256, 7, padding='same', activation='relu'))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(N_CHARS, activation='softmax')))
    return model

def build_conv_rnn_model(max_len=None):
    model = Sequential()
    model.add(Conv1D(256, 7, padding='same', activation='relu', input_shape=(None, N_CHARS)))
    model.add(Conv1D(256, 7, padding='same', activation='relu'))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(Bidirectional(GRU(512, return_sequences=True, recurrent_activation='sigmoid', reset_after=True)))
    model.add(Dropout(0.5))
    model.add(GRU(512, recurrent_activation='sigmoid', reset_after=True))
    model.add(RepeatVector(max_len))
    model.add(GRU(512, return_sequences=True, recurrent_activation='sigmoid', reset_after=True))
    model.add(Dropout(0.5))
    model.add(GRU(512, return_sequences=True, recurrent_activation='sigmoid', reset_after=True))
    model.add(TimeDistributed(Dense(N_CHARS, activation='softmax')))
    return model

def build_replace_model():
    model = Sequential()
    model.add(Embedding(N_WORDS, 128))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dense(N_WORDS, activation='softmax')))
    return model


def build_bidirectional_model():
    model = Sequential()
    model.add(Embedding(N_WORDS, 512))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(N_WORDS, activation='softmax')))

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

def oneHotTransform(docs, pad_to, zero_pad=True):
    vecs = [np.zeros((max(len(d), pad), N_CHARS)) for d,pad in zip(docs, pad_to)]
    for i,d in enumerate(docs):
        for t,c in enumerate(d):
            # one-hot embedding
            vecs[i][t][charMap[c]] = 1
        if not zero_pad:
            for t in range(pad_to[i]- len(d)):
                vecs[i][t+len(d)][0] = 1
    return vecs

def wordOneHotTransform(docs, pad_to):
    vecs = [np.zeros((max(len(d), pad), N_WORDS)) for d,pad in zip(docs, pad_to)]
    for i,d in enumerate(docs):
        for t,c in enumerate(d):
            # one-hot embedding
            vecs[i][t][wordMap.get(c, 1)] = 1
    return vecs

def indexTransform(docs, pad_to):
    vecs = [np.zeros(max(len(d), pad)) for d, pad in zip(docs, pad_to)]
    for i,d in enumerate(docs):
        for t,c in enumerate(d):
            vecs[i][t] = charMap[c]
    return vecs

def wordIndexTransform(docs, pad_to):
    vecs = [np.zeros(max(len(d), pad), dtype=np.float32) for d, pad in zip(docs, pad_to)]
    for i,d in enumerate(docs):
        for t,c in enumerate(d):
            vecs[i][t] = wordMap.get(c,1)
    return vecs

# takes [n_documents, doc_size, one_hot], transforms to [x, n, one_hot] so,
def ngramTransform(vecs, n):
    ngrams = []
    for d in vecs:
        for i in range(0, len(d) - n+1, n//2):
            ngrams.append(d[i:i+n])
    return ngrams


def buildData(path, maxrows=-1):
    data = parse_data.read(path)
    if maxrows > 0:
        data = data[:maxrows]
    in_data, out_data = zip(*data)

    # print(max(len(d) for d in in_data))
    # pad_len = [((max(len(d1), len(d2)) // 100 + 1) * 100) for d1,d2 in zip(in_data, out_data)]
    # pad_len = max([len(d) for d in in_data] + [len(d) for d in out_data])
    # pad_len = [pad_len for i in range(len(in_data))]

    in_vecs = oneHotTransform(in_data, [len(d) for d in out_data], True)
    out_vecs = oneHotTransform(out_data, [len(d) for d in in_data], True)
    del in_data
    del out_data
    del data
    gc.collect()

    in_vecs = ngramTransform(in_vecs, 20)
    out_vecs = ngramTransform(out_vecs, 20)

    # Bin into gropus by length, in intervals of 100. Hopefully helps with runtime and weird behavior wrt padding
    # buckets = [list(filter(lambda x: i*100 < x[0].shape[0] <= (i+1)*100, zip(in_vecs, out_vecs))) for i in range(5)]
    # buckets = [list(zip(*x)) for x in buckets]
    # buckets = [(np.array(x), np.array(y)) for x,y in buckets]
    # return buckets

    return np.array(in_vecs), np.array(out_vecs)

def trainModel(path, model=None, epochs=3):
    in_vecs, out_vecs = buildData(path)
    gc.collect()
    if model is None:
        model = build_conv_rnn_model(out_vecs.shape[1])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.output_shape)
    print('"' +rnn_gen.apply(model, 'I am the best') + '"')
    exampleCallback = LambdaCallback(on_epoch_end=lambda x,y: print('"' +rnn_gen.apply(model, 'I am the best') + '"'))
    checkpointCallback = ModelCheckpoint('rnn.{epoch:02d}-{val_loss:.2f}')
    model.fit(in_vecs[:-2000], out_vecs[:-2000], shuffle=True, batch_size=128, epochs=epochs, validation_data=(in_vecs[-2000:], out_vecs[-2000:]), callbacks=[exampleCallback, checkpointCallback])
    return model


if __name__ == '__main__':
    #trainModel('data/train.tsv.nopunc.tsv', model=load_model('rnn.h5'), epochs=200)
    trainModel('data/train.tsv.nopunc.tsv', epochs=100)



