import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import argparse
import generate as g
import codecs
import os
import io
from model import make_model, MODEL_NAME
input_file = "input.txt"


def main(args):
    with codecs.open(os.path.join(args.data_dir, input_file), "r", encoding='utf-8') as f:
        raw_text = f.read()

    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)


    seq_length = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("Total Patterns: ", n_patterns)
    X = np.reshape(dataX, (n_patterns, seq_length, 1))

    X = X / float(n_vocab)

    y = np_utils.to_categorical(dataY)
    model = make_model(X, y)
    # define the checkpoint
    filepath = os.path.join('models', "snapshot_"+MODEL_NAME+"_{epoch:03d}_{loss:.4f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
    g.sample(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_dir', type=str, help='data directory containing input.txt')

    args = parser.parse_args()

    main(args)
