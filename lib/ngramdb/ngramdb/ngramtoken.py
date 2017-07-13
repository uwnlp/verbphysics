import itertools
from collections import defaultdict, Counter


class Ngram(object):
    def __init__(self, tokens,
                 nid=None, freq=None):
        self.tokens = tokens
        self.headpositions = [t.headposition for t in self.tokens]

        try:
            for t in self.tokens:
                if t.headposition > -1:
                    t.head = self.tokens[t.headposition]
                    t.head.children.append(t)

                else:
                    t.depth = 0

            for t in self.tokens:
                if t.depth is None:
                    depth = 0
                    current = t
                    while current.depth != 0 and depth < len(self.tokens):
                        depth += 1
                        current = current.head

                    if depth >= len(self.tokens):
                        raise IndexError

                    t.depth = depth

            self.height = max(t.depth for t in self.tokens if t is not None)

        except IndexError:
            pass

        self.nid = nid
        self.freq = freq

    @property
    def postags(self):
        return [t.postag for t in self.tokens]

    @property
    def deprels(self):
        return [t.deprel for t in self.tokens]

    @property
    def words(self):
        return [t.surface for t in self.tokens]

    def __repr__(self):
        kwargs = ["=".join((k, repr(v))) for k, v in self.__dict__.items()
                  if v is not None and k != "tokens"]
        return "Ngram({}, {})".format(self.tokens, ', '.join(kwargs))

    def __str__(self):
        return self.rawstring + " (freq: {})".format(self.freq)

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, key):
        return self.tokens[key]

    def __setitem__(self, key, value):
        raise TypeError("Can't replace Token in Ngram!")

    @property
    def surface(self):
        return ' '.join(w for w in self.words)

    @property
    def rawstring(self):
        return ' '.join(t.rawprint for t in self.tokens)


class Token(object):
    def __init__(self, surface,
                 position=None, postag=None, deprel=None, headposition=None,
                 freq=None):

        self.surface = surface

        self.position = position

        self.postag = postag
        self.deprel = deprel

        self.headposition = headposition if headposition != -1 else None
        self.head = None

        self.children = []

        self.depth = None

        self.freq = freq

    def __repr__(self):
        kwargs = ["{}={}".format(k, v) for k, v in self.__dict__.items()
                  if v is not None and k not in ("surface", "head")]
        return "Token({}, {})".format(repr(self.surface), ', '.join(kwargs))

    def __str__(self):
        return self.rawprint

    @property
    def rawprint(self):
        try:
            return '{}/{}/{}/{}'.format(
                self.surface, self.postag, self.deprel, self.headposition)
        except:
            return '{}/{}/{}/{}'.format(
                self.surface, self.postag, self.deprel, 0)


def ngrams_from_tupledict(tuples):
    def keyfunc(x):
        return x['nid']
    results = []

    for key, group in itertools.groupby(tuples, keyfunc):
        group = list(group)

        tokens = [Token(t['surface'], t['position']-1, t['postag'],
                  t['deprel'], headposition=t['headposition']-1)
                  for t in group]

        ngram_freq = None if 'freq' not in group[0] else group[0]['freq']

        ngram = Ngram(tokens, key, freq=ngram_freq)

        # filter out the TRASH :(
        # TODO: why is there TRASH
        # if [t.position for t in ngram] == list(range(len(ngram))) \
        #    and all(t.headposition < len(ngram) for t in ngram):
        results.append(ngram)

    return results


def tokens_from_tupledict(tuples, collapse=True):
    def keyfunc(x):
        return x['surface']

    results = []
    if collapse:
        tuples.sort(key=lambda x: (x['surface'], -x['freq']
                    if 'freq' in x else 0))
        for key, group in itertools.groupby(tuples, keyfunc):
            g = list(group)
            if 'freq' in g[0]:
                freq = sum(t['freq'] for t in g)
            results.append(Token(key, freq=freq))

        results.sort(key=lambda t: t.freq if t.freq is not None else 0,
                     reverse=True)

    else:
        for t in tuples:
            results.append(Token(t['surface'], t['tid'], t['postag'],
                           t['deprel'], freq=t['freq']))

    return results
