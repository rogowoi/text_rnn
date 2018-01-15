import fnmatch
import os
import numpy as np
import io
import pickle


def get_data(data_dir):
    data_file = os.path.join(data_dir, 'input.txt')
    text = io.open(data_file, encoding='utf-8').read()
    print('corpus length:', len(text))
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    with open(os.path.join(data_dir, 'charIndices.pkl'), 'wb') as f:
        pickle.dump(char_indices, f)
    with open(os.path.join(data_dir, 'indicesChar.pkl'), 'wb') as f:
        pickle.dump(indices_char, f)
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return x, y, chars


def find_file(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result[0]