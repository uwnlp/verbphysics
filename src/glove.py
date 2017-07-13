'''
Get GloVe distance between words.

This code adapted from
https://github.com/stanfordnlp/GloVe/blob/master/eval/python/distance.py
'''

# IMPORTS
# -----------------------------------------------------------------------------

# Builtins
import cPickle as pickle

# 3rd party
import numpy as np


# CONSTANTS
# -----------------------------------------------------------------------------

# 300d constants
MODEL_FILE_300D = 'data/glove/glove.6B.300d.txt'
MODEL_FILE_300D_NORM = 'data/glove/glove.6B.300d-weights-norm.npy'
VOCAB_FILE = 'data/glove/glove.6B.vocab.pickle'


# CLASSES
# -----------------------------------------------------------------------------

class Glove(object):
    '''
    Once loaded (g = Glove()), can get word vector using

        g.w[g.vocab[word], :]

    For example,

        g.w[g.vocab['fish'], :]

    This vector will have unit length (l2 norm).
    '''

    def __init__(self, model_file_norm=MODEL_FILE_300D_NORM, vocab_file=VOCAB_FILE):
        '''
        Load normalized vectors and vocab from a cache. Use Glove.convert(...)
        to make.

        Args:
            model_file_norm (str)
            vocab_file (str)
        '''

        w_norm, vocab = self.load_npy(model_file_norm, vocab_file)

        # use
        self.w = w_norm
        self.vocab = vocab

    def load_npy(self, model_file_norm, vocab_file):
        w_norm = np.load(model_file_norm)
        with open(vocab_file, 'r') as f:
            vocab = pickle.load(f)
        return w_norm, vocab

    @staticmethod
    def convert(in_model_file=MODEL_FILE_300D,
                out_model_file_norm=MODEL_FILE_300D_NORM,
                out_vocab_file=VOCAB_FILE):
        '''
        Takes a raw model file and saves to disk (a) a vocab file that indexes
        the model np.ndarray (w matrix), (b) a normalized model file {str: int}
        (vocab).

        Args:
            in_model_file (str): Path to original (downloaded) GloVe file.
            model_file_norm (str): Path to write norm weights to.
            vocab_file (str): Path to write vocab to.
        '''
        vocab, vectors = {}, {}
        with open(in_model_file, 'r') as f:
            i = 0
            for line in f:
                vals = line.rstrip().split(' ')
                word = vals[0]
                vocab[word] = i
                vectors[word] = [float(x) for x in vals[1:]]
                i += 1
        vocab_size = len(vocab)
        vector_dim = len(vectors['the'])  # yay
        w = np.zeros((vocab_size, vector_dim))
        for word, v in vectors.iteritems():
            if word == '<unk>':
                continue
            w[vocab[word], :] = v

        # normalize each word vector to unit length (l2 norm)
        w_norm = np.zeros(w.shape)
        d = (np.sum(w ** 2, 1) ** (0.5))
        w_norm = (w.T / d).T

        # save
        np.save(out_model_file_norm, w_norm)
        with open(out_vocab_file, 'w') as f:
            pickle.dump(vocab, f)

        # NOTE: This left here for your convenience if you decide to adapt this
        # code.
        # return w_norm, vocab

    def distance(self, target, queries):
        '''
        Args:
            target  str
            queries [str]

        Returns:
            np.array of length len(queries): distance (from 1 (close) to 0 (I
                think) (far)) of each word in queries to target according to w
        '''
        res = np.zeros(len(queries))
        if target not in self.vocab:
            return res
        vec_result = self.w[self.vocab[target], :]  # indexes self.w; don't mutate!
        vec_norm = np.zeros(vec_result.shape)
        d = (np.sum(vec_result ** 2,) ** (0.5))
        vec_norm = (vec_result.T / d).T
        dist = np.dot(self.w, vec_norm.T)

        # compute dist for each query
        for i, q in enumerate(queries):
            res[i] = dist[self.vocab[q]] if q in self.vocab else 0.0
        return res


if __name__ == '__main__':
    Glove.convert()
