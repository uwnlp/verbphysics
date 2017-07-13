"""
Functions for querying ngramdb and managing local cache(s) of query results.

Author: mbforbes
"""

# IMPORTS
# ------------------------------------------------------------------------------

from __future__ import division

# builtins
import code  # code.interact(local=dict(globals(), **locals()))
from collections import Counter
import cPickle as pickle
import glob
import math
import os
import sys

# 3rd party
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm

# local
from ngramdb import NgramDb
from ngramdb.util import pprint_ngram_list


# CONSTANTS
# ------------------------------------------------------------------------------

# ngramdb cache info
CACHE_SPREAD_DIR = 'data/ngramdb/queries/'
CACHE_SPREAD_EXT = '.cache'
PMI_CACHE_FN = 'data/ngramdb/pmi/pmi.cache'

QUIT = 'q'

# Wordnet stuff
# The following synset names are synsets given by 'abstraction'
ABSTRACT_SS_NAMES = [
    'abstraction.n.01',
    'abstraction.n.02',
    'abstraction.n.03',
    'abstraction.n.04',
    'abstractedness.n.01',
    'abstraction.n.06',
]
ABSTRACT_SS = [wn.synset(x) for x in ABSTRACT_SS_NAMES]

# Verb endings
SUBS = ['_d', '_p', '_dp', '_op', '_od']

# For PMI and use in system.

# We deal with people separately because we assume that all nominal subjects
# refer to the same physical-propertied object (roughly the same size, weight,
# etc.) Here show all nouns that we assume refer to a person. (I only saw 'man'
# but I'm adding more in case others show up.) We'll remove all of these and
# include only HUMAN_NOUN.
PERSON_NOUNS = ['man', 'woman', 'he', 'she', 'I', 'you', 'human', 'person']

# The replacement for all PERSON_NOUNS.
HUMAN_NOUN = 'PERSON'


# TOP LEVEL FUNCTIONS
# ------------------------------------------------------------------------------

def attr_filter(attr, val):
    """
    General filter constructor: ensures obj's attr == val. Example attrs:
        - 'deprel'
        - 'postag'

    Takes:
        attr (str)
        val  (str)

    Returns:
        f(obj) -> bool
    """
    return lambda o: o.__dict__[attr] == val


def passes_filters(token, filters):
    """
    Returns whether token passes all filters.

    Takes:
        token   (Token)
        filters ([f(Token) -> bool])

    """
    for f in filters:
        if not f(token):
            return False
    return True


def filter_count_n(ngrams, fs):
    """
    Args:
        ngrams ([Ngram] (I think))
        fs ([[filter]]): Critical: *LIST* of filter lists.

    Returns:
        Counter[tuple(str)]
    """
    c = Counter()
    for idx, ng in enumerate(ngrams):
        # Grab words via filters.
        wlists = []
        for f in fs:
            w = [x for x in ng if passes_filters(x, f)]
            wlists.append(w)

        # Don't add this ngram if any word has multiple matches.
        mul = False
        for w in wlists:
            if len(w) != 1:
                mul = True
                break
        if mul:
            continue

        # Flatten the list
        ws = [l[0] for l in wlists]

        # Check the positions are increasing.
        noninc = False
        for i in range(0, len(ws) - 1):
            if ws[i].position >= ws[i+1].position:
                noninc = True
                break
        if noninc:
            continue

        # Turn into a tuple (to be a key of the Counter).
        tup = tuple([w.surface for w in ws])
        c[tup] += ng.freq
    return c


def is_abstract(noun):
    """
    Try to (heuristically) filter abstract nouns.

    Args:
        noun (str|unicode)
    """
    assert type(noun) in [str, unicode], 'bad noun type: %r. noun: %s' % (
            type(noun), noun)
    noun_ss = wn.synsets(noun, pos=wn.NOUN)

    # if wordnet doesn't know about the noun, let it fly
    if len(noun_ss) == 0:
        return False

    # Checking just the first (most common?) synset for the noun, but checking
    # all hypernym paths for that synset, and all paths must be clean of any
    # abstraction.

    # previously, looped over all with:
    #     for n_ss in noun_ss:
    # but that proved not restrictive enough
    n_ss = noun_ss[0]
    paths = n_ss.hypernym_paths()
    # all ps must pass
    paths_good = True
    for p in paths:
        for a in ABSTRACT_SS:
            if a in p:
                paths_good = False
                break
        if not paths_good:
            break
    if paths_good:
        # debug
        # print paths
        return False
    return True


def filter_abstract_from_counter(c):
    """
    Try to (heuristically) filter abstract nouns.

    Args:
        c (Counter) Frequency counts of nouns
    """
    for n in c.keys():
        if is_abstract(n):
            del c[n]


# CLASSES
# ------------------------------------------------------------------------------

class SizeQueryN(object):
    """Keys to the n-obj cache"""

    def __init__(self, query, raw_f_list):
        """
        Takes:
            query  (NGramDBQuery)
            raw_f_list ([[(str, str)]])
        """
        self.query = query
        self.raw_f_list = raw_f_list

    def __eq__(self, other):
        return (self.query == other.query and
            self.raw_f_list == other.raw_f_list)

    def __hash__(self):
        f_hash = hash(tuple([tuple(rf) for rf in self.raw_f_list]))
        return hash((hash(self.query), f_hash))


class DBWrapper(object):
    def __init__(self, cxn_id, fn_n):
        """
        Args:
            cxn_id (str) connection ID for the DB
            fn_n (str) directory of spread cache n-obj cache
        """
        self.db = NgramDb(cxn_id)
        self.fn_n = fn_n
        self.cache_n = {}

    def load_caches(self):
        if os.path.isdir(self.fn_n):
            # Turning off for now; should have logging framework.
            # print '<loading cache N>'
            files = glob.glob(self.fn_n + '*' + CACHE_SPREAD_EXT)
            # hashes = [os.path.split(f)[1].rstrip(CACHE_SPREAD_EXT) for f in files]
            for f in tqdm(files):
                k, v = self.load_spreadfile(f)
                self.cache_n[k] = v
            # Turning off for now; should have logging framework.
            # print '<cache N: %d queries>' % (len(self.cache_n.keys()))

    def print_cache_stats(self):
        """
        Prints aggregated verb stats for cache.
        """
        n = 100
        c = Counter()
        for k, v in self.cache_n.iteritems():
            w = [w for w in k.query.words if w is not None]
            w = w[0]
            c[w] += sum(v.values())
        print '%d most common verbs:' & (n)
        for k,v in c.most_common(n):
            print '\t', k, '\t', v

    def write_caches(self):
        """
        Writes (all) caches to files.
        """
        print '<writing caches>'
        self.write_spread(self.cache_n, CACHE_SPREAD_DIR)

    def write_kv(self, fn, k, v):
        """
        Writes a single k, v to file name fn.

        Args:
            fn (str)
            k (Object)
            v (Object)
        """
        with open(fn, 'w') as f:
            pickle.dump(k, f)
            pickle.dump(v, f)

    def load_spreadfile(self, fn):
        """
        utility to load a single spreadfile and return its contents as k, v

        Args:
            fn (str)
        """
        with open(fn, 'r') as f:
            k = pickle.load(f)
            v = pickle.load(f)
        return k, v

    def check_cache_files(self, d):
        """
        Checks d to find any malformed cache entries. Does not require cache to
        be loaded in advance.

        Args:
            d (str): Directory.
        """
        files = glob.glob(d + '*' + CACHE_SPREAD_EXT)
        n_good, n_bad, n_total = 0, 0, 0
        for f in files:
            good = True
            try:
                k, v = self.load_spreadfile(f)
                if type(k) != SizeQueryN:
                    good = False
                    print 'ERRR: cache file key is not type SizeQueryN; is %r (%s)' % (type(k), f)
                if type(v) != Counter:
                    good = False
                    print 'ERRR: cache file value is not type Counter; is %r (%s)' % (type(v), f)
            except:
                good = False
                print 'ERRR: Problem loading cache file (%s)' % (f)
            if good:
                n_good += 1
            else:
                n_bad += 1
            n_total += 1
        if n_good + n_bad != n_total:
            print 'ERRR: cache checking error: good (%d) + bad (%d) != total (%d)' % (n_good, n_bad, n_total)
        print '%d/%d/%d good/bad/total' % (n_good, n_bad, n_total)

    def write_spread_item(self, k, v, d):
        """
        Writes a single item (k, v) to d.

        Args:
            k (SizeQueryN): Cache key.
            v (Counter): Cache value.
            d (str): Directory.

        Returns:
            bool: True if the item was written, False if it was found and
                  didn't need to be written.
        """
        orig = str(hash(k))
        written, done = False, False
        postfix, postfix_n = '', 0
        while not done:
            fn = os.path.join(d, orig) + postfix + CACHE_SPREAD_EXT

            # no collision! just write.
            if not os.path.isfile(fn):
                self.write_kv(fn, k, v)
                written = True
                done = True
            else:
                # existing key, existing value
                ek, ev = self.load_spreadfile(fn)
                if ek == k and ev == v:
                    # best case scenario: collision and it's what we want!
                    # no need to write.
                    done = True
                else:
                    # a collision AND it's not what we want. try the next
                    # one.
                    postfix_n += 1
                    postfix = '-%d' % (postfix_n)
        return written

    def write_spread(self, c, d):
        """
        Lazily writes out c across d, only writing what is necessary.

        Args:
            c (dict): Cache.
            d (str): Directory.
        """
        # tracking
        n_total, n_written = len(c.keys()), 0

        # write me maybe
        for k,v in c.iteritems():
            written = self.write_spread_item(k, v, d)
            n_written = n_written + 1 if written else n_written
        print '<%d/%d/%d written/skipped/total>' % (n_written, n_total - n_written, n_total)

    def run(self, sq, fetch=False):
        """
        Cache-aware query runner.

        If fetch=True, always fetches the result and returns it as the second
        value. Otherwise, the second value returned will be None.

        Takes:
            sq (SizeQueryN)

        Returns:
            Counter[tuple(str)], (result|None)
        """
        # Cache getting layer.
        cache = self.cache_n
        in_cache = sq in cache.keys()
        if (not fetch) and in_cache:
            # Turning off for now; should have logging framework.
            # print '<cache hit>'
            return cache[sq], None
        if fetch:
            print '<skipping cache>'
        else:
            print '<cache miss>'

        # Running layer.
        res = self.db.run_query(sq.query)
        f_list = []
        for raw_fs in sq.raw_f_list:
            fs = [attr_filter(f[0], f[1]) for f in raw_fs]
            f_list.append(fs)
        count = filter_count_n(res, f_list)

        # Cache writing layer.
        if not in_cache:
            cache[sq] = count
            # In case things go wrong: always write the cache(s) after a
            # successful query. We only need to write this one query.
            self.write_spread_item(sq, count, self.fn_n)

        # Returning layer. We could just always return res, but this will keep
        # memory freer and make exploration more explicit.
        if fetch:
            return count, res
        return count, None


class Data(object):
    """
    This is the API for how to interact with the data programmatically.

    This class aims to ease the transition of this code base along three axes:

    -   (a) interactive      --> programmatic
    -   (b) database-focused --> data-focused
    -   (c) CLI-focused      --> API-focused
    """

    def __init__(self, w=None):
        """
        NOTE(mbforbes): Main consideration is whether I want this to be a
        lighter-weight init than loading the DB.

        Args:
            w (DBWrapper, optional): Default is None, in which case one is
                loaded from the cache. If w is provided, it should have the
                caches loaded already.

        """
        if w is None:
            w = DBWrapper('max data API', CACHE_SPREAD_DIR)
            w.load_caches()
        self._w = w

        # This is also maybe used below---init once.
        self.lmtz = WordNetLemmatizer()

    def get_queries_for_verb(self, verb):
        """
        Gets queries, noun indexes, and preposition indexes for a verb.

        Args:
            verb (str): Verb to get queries for.

        Returns:
            (
                SizeQueryN,  -- queries
                [[int]],     -- corresponding noun indexes
                [[int]],     -- corresponding preposition indexes
            )
        """
        qs = [
            # e.g. I threw the ball
            # saving: obj
            SizeQueryN(
                # Query
                self._w.db.create_query(
                    words=[None, verb, None],
                    postags=['PRP', 'VBD', 'NN|NNS'],
                    deprels=['nsubj', None, 'dobj']
                ),
                # List of filter lists.
                [
                    [('deprel', 'nsubj')],
                    [('deprel', 'dobj')],
                ]
            ),
            # e.g. I walked into the room
            # saving: prp, obj
            SizeQueryN(
                # Query
                self._w.db.create_query(
                    words=[None, verb, None, None],
                    postags=['PRP', 'VBD', 'IN', 'NN|NNS'],
                    deprels=['nsubj', None, None, 'pobj']
                ),
                # List of filter lists.
                [
                    [('deprel', 'nsubj')],
                    [('postag', 'IN')],
                    [('deprel', 'pobj')],
                ]
            ),
            # e.g. I put it (in / inside / on / under) the cupboard
            # e.g. I put it over the cupboard
            # saving: obj1, prp, obj
            SizeQueryN(
                # Query
                self._w.db.create_query(
                    words=[None, verb, None, None, None],
                    postags=['PRP', 'VBD', 'NN|NNS', 'IN', 'NN|NNS'],
                    deprels=['nsubj', None, 'dobj', None, 'pobj']
                ),
                # List of filter lists.
                [
                    [('deprel', 'nsubj')],
                    [('deprel', 'dobj')],
                    [('postag', 'IN')],
                    [('deprel', 'pobj')],
                ]
            ),
            # ------------------------------------------------------------------
            # META NOTES:
            #    The following two are pretty interesting, but they give
            #    absolute info rather than relative info. (Philosophically, a
            #    sentence is actually relative to "normal" experiences, but it's
            #    much harder to figure this reference frame out than with a
            #    direct comparison.)
            #
            #    Thus, it's probably OK to put these on the back-burner for now.
            # ------------------------------------------------------------------
            # # e.g. the plane flew
            # # saving: obj
            # SizeQueryN(
            #     # Query
            #     self._w.db.create_query(
            #         words=[None, verb],
            #         postags=['NN|NNS', 'VBD'],
            #         deprels=['nsubj', 'ROOT']
            #     ),
            #     # List of filter lists.
            #     [
            #         [('deprel', 'nsubj')],
            #     ]
            # ),
            # # e.g. the plane flew by
            # # saving: obj, prep
            # SizeQueryN(
            #     # Query
            #     self._w.db.create_query(
            #         words=[None, verb, None],
            #         postags=['NN|NNS', 'VBD', 'IN'],
            #         deprels=['nsubj', 'ROOT', None]
            #     ),
            #     # List of filter lists.
            #     [
            #         [('deprel', 'nsubj')],
            #         [('postag', 'IN')],
            #     ]
            # ),

            # ------------------------------------------------------------------
            # META NOTES:
            #    The following two are good analogs to the nsubj being PRP; if
            #    we talk about objects (nouns) doing things to other objects
            #    (nouns), this should capture that.
            #
            #    So, these, I think, should be kept.
            # ------------------------------------------------------------------
            # e.g. the plane flew by the blimp
            # saving: obj, prep, obj
            SizeQueryN(
                # Query
                self._w.db.create_query(
                    words=[None, verb, None, None],
                    postags=['NN|NNS', 'VBD', 'IN', 'NN|NNS'],
                    deprels=['nsubj', 'ROOT', None, 'pobj']
                ),
                # List of filter lists.
                [
                    [('deprel', 'nsubj')],
                    [('postag', 'IN')],
                    [('deprel', 'pobj')],
                ]
            ),
            # e.g. the boot squashed the bug
            # saving: obj, prep, obj
            SizeQueryN(
                # Query
                self._w.db.create_query(
                    words=[None, verb, None],
                    postags=['NN|NNS', 'VBD', 'NN|NNS'],
                    deprels=['nsubj', 'ROOT', 'dobj']
                ),
                # List of filter lists.
                [
                    [('deprel', 'nsubj')],
                    [('deprel', 'dobj')],
                ]
            ),

            # ------------------------------------------------------------------
            # META NOTES:
            #    The following query is interesting, but it's a three-way
            #    comparison, which will take extra work to integrate.
            #
            #    Thus: back-burner for now.
            # ------------------------------------------------------------------
            # # e.g. the man squashed the bug with his shoe
            # # saving: obj, prep, obj
            # SizeQueryN(
            #     # Query
            #     self._w.db.create_query(
            #         words=[None, verb, None, None, None],
            #         postags=['NN|NNS', 'VBD', 'NN|NNS', 'IN', 'NN|NNS'],
            #         deprels=['nsubj', 'ROOT', 'dobj', None, 'pobj']
            #     ),
            #     # List of filter lists.
            #     [
            #         [('deprel', 'nsubj')],
            #         [('deprel', 'dobj')],
            #         [('postag', 'IN')],
            #         [('deprel', 'pobj')],
            #     ]
            # ),
        ]
        noun_idxes = [
            [1],
            [2],
            [1, 3],
            # [0],
            # [0],
            [0, 2],
            [0, 1],
            # [0, 1, 3],
        ]
        prep_idxes  = [
            [],
            [1],
            [2],
            # [],
            # [1],
            [1],
            [],
            # [2],
        ]
        return qs, noun_idxes, prep_idxes

    def get_freq_nouns(self, v, s, p, cutoff=1000):
        """
        Get frequent nouns for v_sub occurring at or above cutoff.

        Args:
            v (str): Verb
            s (str): Sub ('_d', '_p', etc.)
            p (str|None): Preposition (or None if sub doesn't use one)
            cutoff (int): Frequency cutoff below which nouns will not be
                returned.

        Returns:
            [str] | [(str, str)]
        """
        c, n_idxes = self._get_cache_res_prep(v, s, p)
        fc = Counter()
        for surface, count in c.iteritems():
            il = list(surface)
            # NOTE: Assuming at most 2 nouns per query.
            o = il[n_idxes[0]] if len(n_idxes) == 1 else (il[n_idxes[0]], il[n_idxes[1]])
            fc[o] += count
        res = []
        for o, freq in fc.most_common():
            if freq < cutoff:
                break
            res += [o]
        return res

    def get_prep_freqs_agg(self, v):
        """
        Gets the counter of prepositions for v, aggregating across all of its
        subs (and their query results).

        Args:
            v (str) verb

        Returns:
            Counter
        """
        # gotta iterate over the various subs

        cs, _, p_idxes = self._get_cache_res_verb(v)
        # aggregating over different subs
        ac = Counter()
        for i, c in enumerate(cs):
            # Only aggregate prepositions if we saved any
            ps = p_idxes[i]
            if len(ps) == 0:
                continue
            # Consider most general case that we could have n prepositions
            # saved.
            for p in ps:
                for item, count in c.iteritems():
                    prep = list(item)[p]
                    ac[prep] += count
        return ac

    def get_prep_freqs(self, v):
        """
        Gets a counter of prepositions for v for each of its subs.

        Args:
            v (str) verb
            sub (str) one of ['p', 'dp', 'op']

        Returns:
            {str -> Counter(str)} --- {sub -> Counter(prep)}
        """
        cs, _, p_idxes = self._get_cache_res_verb(v)
        res = {}
        for i, c in enumerate(cs):
            # Only count prepositions any exist in this particular query
            ps = p_idxes[i]
            if len(ps) == 0:
                continue

            sub = SUBS[i]
            res[sub] = Counter()
            # Consider most general case that we could have n prepositions
            # saved.
            for p in ps:
                for item, count in c.iteritems():
                    prep = list(item)[p]
                    res[sub][prep] += count
        return res

    def get_top_nouns(self, v, s, p, filter_abstract, lemmatize):
        """
        Gets the counter for nouns in v, s, p.

        Args:
            v (str): Verb
            s (str): Sub
            p (str): Prep
            filter_abstract (bool): Whether to filter abstract nouns out of the
                returned list.
            lemmatize (bool): Whether to compress nouns into their lemmatized
                form before returning.
        Returns:
            Counter[str]
        """
        c, n_idxes = self._get_cache_res_prep(v, s, p)

        # aggregating
        ac = Counter()
        for i in n_idxes:
            for surface, count in c.iteritems():
                noun = list(surface)[i]
                ac[noun] += count

        # debugging
        # print '... <before filter abstract>'
        # code.interact(local=dict(globals(), **locals()))

        # Maybe filter
        if filter_abstract:
            filter_abstract_from_counter(ac)

        # debugging
        # print '... <after filter abstract>'
        # code.interact(local=dict(globals(), **locals()))

        # Maybe lemmatize
        if lemmatize:
            self.compress_lemmas_in_counter(ac)

        return ac

    def compress_lemmas_in_counter(self, c):
        """
        Compresses forms to their lemma in (keys of) c (by adding counts).

        Args:
            c (Counter)

        Modifies c in-place.
        """
        for k in c.keys():
            l = self.lmtz.lemmatize(k)
            if l != k:
                c[l] += c[k]
                del c[k]

    def get_verb_freq(self, v):
        """
        Get frequency statistics for verb.

        Args:
            v (str): Verbn

        Returns:
            int: sum occurrences of verb in currently active queries.
        """
        cs, _, _ = self._get_cache_res_verb(v)
        total = 0
        for c in cs:
            total += sum(c.values())
        return total

    def _get_cache_res_verb(self, v):
        """
        Gets verb cache result for all subs.

        Args:
            v (str): Verb

        Returns (each list in the returned tuple is of length `len(SUBS)`:
            (
                [Counter], -- results for each sub
                [[int]],   -- noun indexes for each sub
                [[int]],   -- preposition indexes for each sub
            )
        """
        qs, noun_idxes, prep_idxes = self.get_queries_for_verb(v)
        cs = []
        for q in qs:
            c, _ = self._w.run(q)
            cs += [c]
        return cs, noun_idxes, prep_idxes

    def _get_cache_res_sub(self, v, s):
        """
        Returns the cached result for v_sub along with the indexes of the nouns.
        Helper function for querying APIs.

        Args:
            v (str): Verb
            s (str): Sub

        Returns:
            Counter[tuple(str)]
            [int]: Noun indexes
            [int]: Prep indexes
        """
        # TODO(mbforbes): Should consolidate w/ constant in system.py. Then
        # again, much code (e.g. query code) assumes just these exist...

        # Figure out what we're looking for
        idx = SUBS.index(s)

        # Get all data
        qs, noun_idxes, prep_idxes = self.get_queries_for_verb(v)

        # Return the piece we want from each
        c, _ = self._w.run(qs[idx])
        n = noun_idxes[idx]
        p = prep_idxes[idx]
        return c, n, p

    def _get_cache_res_prep(self, v, s, p):
        """
        Args:
            v (str): Verb
            s (str): Sub
            p (str): Prep

        Returns:
            Counter[tuple(str)]
            [int]: Noun indexes
        """
        c, nidxes, pidxes = self._get_cache_res_sub(v, s)

        # If we're looking at a sub without a prep, we can just return directly.
        if p is None and len(pidxes) > 0 or p is not None and len(pidxes) == 0:
            assert False, 'verb %s sub %s has prep mismatch: prep %r, idxes %r' % (v, s, p, pidxes)
        if p is None and len(pidxes) == 0:
            return c, nidxes

        # Else, we select only results which match the preposition.
        res = Counter()
        # NOTE: Assuming that there's at most one preposition per query as
        # that's how things are currently structured. Change the argument above
        # to be a [str] (and Turk new data to match) if you want to allow this.
        for surface, cnt in c.iteritems():
            pidx = pidxes[0]
            if list(surface)[pidx] != p:
                continue
            res[surface] += cnt
        return res, nidxes


class PMI(object):

    def __init__(self):
        """
        Must have run PMI.compute() first.
        """
        with open(PMI_CACHE_FN, 'r') as f:
            self.frame_counter = pickle.load(f)
            self.frame_total = pickle.load(f)
            self.nounpair_counter = pickle.load(f)
            self.nounpair_total = pickle.load(f)
            self.joint_counter = pickle.load(f)
            self.joint_total = pickle.load(f)

    def query(self, frame, nounpair):
        """
        For the given frame (verb_sub[_prep]) and nounpair (noun1, noun2),
        determines the PMI score between them (according to the ngramdb data)
        and returns it.

        Args:
            frame (string)              verb_sub[_prep]
            nounpair (string, string)   (noun1, noun2)

        Return:
            float: The PMI between
        """
        res = self._get_pmi(frame, nounpair)
        # if HUMAN_NOUN in nounpair and res >= 0:
        #     print 'GOT HUMAN PMI:', frame, nounpair, res
        return res

    def _get_pmi(self, frame, nounpair):
        """
        Refactoring query() to allow for multiple trials if desired. See doc
        there.

        Args:
            frame (string)
            nounpair (string, string)

        Returns:
            float
        """
        joint = (frame, nounpair)
        # log(joint / (x*y)) = log(joint) - log(x*y) = log(joint) - log(x) - log(y)
        #
        # for any prob p_x, p_x = count(x) / total(x)
        #
        # log(p_x) = log(count(x) / total(x)) = log(count(x)) - log(total(x))

        # sanity checking
        if self.joint_counter[joint] == 0 or self.frame_counter[frame] == 0 or \
                self.nounpair_counter[nounpair] == 0:
            return float('-inf')

        lj = math.log(self.joint_counter[joint]) - math.log(self.joint_total)
        lf = math.log(self.frame_counter[frame]) - math.log(self.frame_total)
        lnp = math.log(self.nounpair_counter[nounpair]) - math.log(self.nounpair_total)
        return lj - lf - lnp

    @staticmethod
    def compute():
        """
        Computing PMI and saving
        """
        # get caches
        w = DBWrapper('max print verb info', CACHE_SPREAD_DIR)
        w.load_caches()
        api = Data(w)

        # load verbs
        print '[pmi] loading verbs...'
        basedir = 'data/turk/hardcore/'
        # NOTE: This is fine because this isn't the corpus we are training /
        # testing on --- these are just the verbs that we might encounter, so
        # we're precomputing PMI for everything we might want and caching it.
        # (i.e. not actually touching test data here).
        fnames = ['train.txt', 'dev.txt', 'test.txt']
        verbs = []
        for fname in fnames:
            with open(basedir + fname, 'r') as f:
                verbs += [v.strip() for v in f.readlines()]

        # there are our variables for aggregating pmi counts
        frame_counter = Counter()
        nounpair_counter = Counter()
        joint_counter = Counter()

        print '[pmi] counting query results...'
        for verb in tqdm(verbs):
            counters, noun_idxes_lst, prep_idxes_lst = api._get_cache_res_verb(verb)
            assert len(SUBS) == len(counters), 'Should get 1 set of results back for each frame type (sub)'
            for i in range(len(SUBS)):
                counter = counters[i]
                noun_idxes = noun_idxes_lst[i]
                prep_idxes = prep_idxes_lst[i]

                for surface_forms, freq in counter.iteritems():
                    # Compute frame
                    frame = verb + SUBS[i]
                    if len(prep_idxes) > 0:
                        # NOTE: assuming at most 1 prep for now
                        frame += '_' + surface_forms[prep_idxes[0]]

                    # Compute noun pair
                    # NOTE: Assuming 1 or 2 nouns
                    if len(noun_idxes) == 1:
                        nouns = [HUMAN_NOUN, surface_forms[noun_idxes[0]]]
                    else:
                        nouns = [surface_forms[noun_idxes[0]], surface_forms[noun_idxes[1]]]

                    for j in range(2):
                        if nouns[j] in PERSON_NOUNS:
                            nouns[j] = HUMAN_NOUN

                    nounpair = tuple(nouns)

                    frame_counter[frame] += freq
                    nounpair_counter[nounpair] += freq
                    joint_counter[(frame, nounpair)] += freq

        # easy to get actual probs so why not
        frame_total = sum(frame_counter.values())
        nounpair_total = sum(nounpair_counter.values())
        joint_total = sum(joint_counter.values())

        with open(PMI_CACHE_FN, 'w') as f:
            pickle.dump(frame_counter, f)
            pickle.dump(frame_total, f)
            pickle.dump(nounpair_counter, f)
            pickle.dump(nounpair_total, f)
            pickle.dump(joint_counter, f)
            pickle.dump(joint_total, f)


# TOP-LEVEL COMMAND-LINE FUNCS
# ------------------------------------------------------------------------------

def explore():
    w = DBWrapper('max explore', CACHE_SPREAD_DIR)
    w.load_caches()
    api = Data(w)
    code.interact(local=dict(globals(), **locals()))


def run_for_verb(verb, w=None):
    """
    Run queries for a verb.

    Args:
        verb (str): Verb to run queries for.
        w (DBWrapper, optional): Default is None, in which case one is loaded
            from the cache. If w is provided, it should have the caches loaded
            already.
    """
    print '<running for verb "%s">' % (verb)
    if w is None:
        w = DBWrapper('max verb run', CACHE_SPREAD_DIR)
        w.load_caches()
    api = Data(w)
    qs, _, _ = api.get_queries_for_verb(verb)
    for q in qs:
        w.run(q)


def run_for_file(fname):
    """
    Run queries for verbs in a file.

    Args:
        fname (str)
    """
    # Prep the DB.
    w = DBWrapper('max file run', CACHE_SPREAD_DIR)
    w.load_caches()

    # Read the verbs.
    with open(fname) as f:
        lines = f.readlines()
    verbs = [line.strip() for line in lines]

    # Run the queries.
    for v in verbs:
        run_for_verb(v, w)


def ping():
    """
    Really just want to see if the databse is actually up.
    """
    w = DBWrapper('max ping', CACHE_SPREAD_DIR)
    w.load_caches()
    res = w.db.create_and_run_query(
        words=['cat', None, 'dog'],
        postags=['NN', 'VBD', 'NN'],
        deprels=['nsubj', None, 'dobj'],
    )
    pprint_ngram_list(res[:10])


def check_cache():
    """
    Checks the cache files on disk.
    """
    w = DBWrapper('max check cache', CACHE_SPREAD_DIR)
    w.check_cache_files(CACHE_SPREAD_DIR)


def print_cache_stats():
    """
    Prints (verb) stats of cache.
    """
    w = DBWrapper('max print cache', CACHE_SPREAD_DIR)
    w.load_caches()
    w.print_cache_stats()


def interact_verb_info():
    """
    Prints info about verbs.

    Specifically, prints cached verb data in sets that would be partitioned
    into nodes in the factor graph.
    """
    w = DBWrapper('max interact verb info', CACHE_SPREAD_DIR)
    w.load_caches()
    api = Data(w)

    while True:
        verb = raw_input('enter a verb (%s to quit): ' % (QUIT))
        if verb == QUIT:
            break
        _print_single_verb_info(w, api, verb)


def print_top_preps_interact():
    """
    Pretty much this yeah.
    """
    w = DBWrapper('max interact verb preps', CACHE_SPREAD_DIR)
    w.load_caches()
    api = Data(w)

    while True:
        verb = raw_input('enter a verb (%s to quit): ' % (QUIT))
        if verb == QUIT:
            break
        preps = api.get_prep_freqs_agg(verb)
        for p, count in preps.most_common(20):
            print '%d\t%s' % (count, p)

def print_verb_info(verbs):
    """
    Prints info about verbs.

    Args:
        verbs ([str])
    """
    w = DBWrapper('max print verb info', CACHE_SPREAD_DIR)
    w.load_caches()
    api = Data(w)
    for verb in verbs:
        _print_single_verb_info(w, api, verb)


def _print_single_verb_info(w, api, verb):
    """
    Args:
        w (DBWrapper)
        api (API)
        verb (str)
    """
    qs, _, _ = api.get_queries_for_verb(verb)
    descs = [
        '(1) PRP   %s dobj' % (verb),
        '(2) PRP   %s      IN pobj' % (verb),
        '(3) PRP   %s dobj IN pobj' % (verb),
        # '(4) NN(S) %s' % (verb),
        # '(5) NN(S) %s      IN' % (verb),
        '(6) NN(S) %s      IN pobj' % (verb),
        '(7) NN(S) %s dobj' % (verb),
        # '(8) NN(S) %s dobj IN pobj' % (verb),
    ]
    # These index the noun positions in the queries. They are defined assuming
    # the PRPs---where applicable---have already been stripped off. Because of
    # this, we don't use the noun indexes returned from the
    # get_*queries_for_verb functions.
    idxes = [
        [0],
        [1],
        [0,2],
        # [0],
        # [0],
        [0,2],
        [0,1],
        # [0,2,4],
    ]

    # Sanity check
    if len(qs) != len(descs) or len(qs) != len(idxes):
        print 'ERRR: Code out-of-date: qs, descs, idxes must match query #s.'
        return

    for i, q in enumerate(qs):
        noun_idxes = idxes[i]
        c, _ = w.run(q)
        print descs[i]

        # maybe compress PRPs
        compress = True  # Change to false to not compress.
        if compress and i < 3:
            # compress PRP
            cp = Counter()
            for item, count in c.iteritems():
                minus_prp = tuple(list(item)[1:])
                cp[minus_prp] += count
        else:
            # no PRP; can't compress (or just disabled)
            cp = c

        # NOTE: In other functions, abstract nouns may be filtered. May want to
        # enable here.
        # filter_abstract_from_counter(cp, noun_idxes)
        for r, f in cp.most_common(20):
            print '\t %s\t %d' % (r, f)


def compute_pmi():
    PMI.compute()


def query_pmi():
    pmi = PMI()
    print 'Entering interactive python shell'
    print 'Query pmi with `pmi.query(frame, nounpair)`'
    print 'Example:'
    print ">>> pmi.query('looked_op_as', ('children', 'friend'))"
    print pmi.query('looked_op_as', ('children', 'friend'))
    code.interact(local=dict(globals(), **locals()))


def main():
    """
    NOTE: To any reader of this code, apologizes for the hacked-together
    command line parsing. I should have just used `argparse`. If you're seeing
    this and care, open an issue and send a PR and we'll make this better :-)
    """
    # sanity checking
    if len(sys.argv) < 2:
        print 'USAGE: python data.py --command [args]'
        print 'Possible commands:'
        print '\t', '--explore \t\t run iterative exploration'
        print '\t', '--ping \t\t\t ping the Myria DB'
        print '\t', '--check-cache \t\t check local cache for soundness'
        print '\t', '--print-cache-stats \t print (verb) cache stats'
        print '\t', '--interact-verb-info \t print cached verb info (interact)'
        print '\t', '--query-pmi \t\t querys precomputed PMI results'
        print '\t', '--compute-pmi \t\t computes PMI over ngramdb w/ turked verbs'
        print '\t', '--print-verb-info <verb1> [<more_verbs>]\t print cached verb info (verb(s) provided)'
        print '\t', '--print-top-preps-interact\t print cached info on top preps for verbs (interact)'
        print '\t', '--verb <verb> \t\t run query and cache results for <verb>'
        print '\t', '--file <file> \t\t run queries for all words in <file>'
        return 1

    if sys.argv[1] == '--explore':
        explore()
    elif sys.argv[1] == '--ping':
        ping()
    elif sys.argv[1] == '--check-cache':
        check_cache()
    elif sys.argv[1] == '--print-cache-stats':
        print_cache_stats()
    elif sys.argv[1] == '--interact-verb-info':
        interact_verb_info()
    elif sys.argv[1] == '--compute-pmi':
        compute_pmi()
    elif sys.argv[1] == '--query-pmi':
        query_pmi()
    elif sys.argv[1] == '--print-top-preps-interact':
        print_top_preps_interact()
    elif sys.argv[1] == '--print-verb-info':
        if len(sys.argv) < 3:
            print 'ERRR: Command "--print-verb-info" requires at least one verb'
            return 1
        print_verb_info(sys.argv[2:])
    elif sys.argv[1] == '--verb':
        if len(sys.argv) < 3:
            print 'ERRR: Command "--verb" requires a verb'
            return 1
        run_for_verb(sys.argv[2])
    elif sys.argv[1] == '--file':
        if len(sys.argv) < 3:
            print 'ERRR: Command "--file" requires a filename'
            return 1
        run_for_file(sys.argv[2])
    else:
        print 'ERRR: Command "%s" unrecognized' % (sys.argv[1])
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
