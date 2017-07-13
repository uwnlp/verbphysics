"""
Settings is for running experiments with different parameters. Supports
stuff like auto grid search and logging (yes, logging!).

TODO:
    - [ ] sanity check passed experiments to be of type 'list'. If passing a
          single setting that happens to be iterable it will happily iterate
          through, e.g., all characters of a string.

author: mbforbes
"""

# IMPORTS
# -----------------------------------------------------------------------------

# builtins
import code  # code.interact(local=dict(globals(), **locals()))
from itertools import product
import logging

# 3rd party
import numpy as np
from tabulate import tabulate


# TOP-LEVEL FUNCTIONS
# -----------------------------------------------------------------------------

def cell_massage(val):
    """
    Preprocessing values to ensure that they can fit well in the cell of a
    printed table.

    Args:
        val

    Returns:
        val (or something)
    """
    # tabulate appears to sometimes work for bools and sometimes not. So I'm
    # doing this so that it always works.
    if type(val) is bool:
        return 'True' if val else 'False'
    # tabulate TOTALLY doesn't handle numpy arrays as cell entries.
    if type(val) is np.ndarray:
        return ', '.join([str(row) for row in val])
    # default
    return val


# CLASSES
# -----------------------------------------------------------------------------

class Settings(object):
    """
    Class for trying all (exponentially many) combinations of all parameter
        settings. Must call next() before each trial run.

    New features:

    - [x]   np.ndarray aligned printing

    - [x]   Print settings that aren't changing at the top. If they're default,
            note them as so.

            Each iteration, note only the thing that is changing.

            Integrate with results. Output in a table format with the stuff that
            is changing.

            Example:

                Settings that aren't changing:

                   foo: 0.5 (default)
                barbar: 0.7 (default)
                   baz: 0.9

                (.. experiments run here ...)

                la -> |  0.5  |  0.7  |  0.9
                ------+-------+-------+------
                      |   98% |  30%  |  40%

                2D for 2 varied. TODO: For > 2, multiple tables?

                TODO: Use pandas for this?
    """
    # Class vars as constants for keys

    # Used with iterators to tell when to stop.
    NothingLeft = object()

    Eval = 'eval'
    GloveVerbSimThresh = 'glove-verb-sim-thresh'
    GloveNounSimThresh = 'glove-noun-sim-thresh'
    Attrs = 'attrs'
    VerbSimPot = 'verb-sim-pot'
    NounEqPot = 'noun-eq-pot'
    NounSimPot = 'noun-sim-pot'
    NounSimRevPot = 'noun-sim-rev-pot'
    MaxNounsPerFrame = 'max-nouns-per-frame'
    FilterAbstract = 'filter-abstract'
    GTBiggerPot = 'gt-bigger-pot'
    GTSmallerPot = 'gt-smaller-pot'
    GTEqPot = 'gt-eq-pot'
    AgreementNeeded = 'agreement-needed'
    SelPrefMethod = 'sel-pref-method'
    SelPrefFreqCutoff = 'sel-pref-freq-cutoff'
    SelPrefPMICutoff = 'sel-pref-pmi-cutoff'
    SelPrefPot = 'sel-pref-pot'
    NormalizeLBP = 'normalize-lbp'
    LBPMaxIters = 'lbp-max-iters'
    IncludeVerbSimFactors = 'include-verb-sim-factors'
    IncludeNounSimFactors = 'include-noun-sim-factors'
    IncludeSelPrefFactors = 'include-sel-pref-factors'
    IncludeInfWithinverbSimframeFactors = 'include-inf-withinverb-simframe-factors'
    WithinverbSimframePot = 'withinverb-simframe-pot'
    IncludeXgraph = 'include-xgraph'
    XgraphTuples = 'xgraph-tuples'
    XgraphPot = 'xgraph-pot'
    MaxSeeds = 'max-seeds'
    RawNounsFilename = 'raw-nouns-filename'
    EvalNounsFilename = 'eval-nouns-filename'
    Lemmatize = 'lemmatize'
    SelPrefMinFreqForPMI = 'sel-pref-min-freq-for-pmi'
    IncludeNgramDBNouns = 'include-ngramdb-nouns'
    IncludeGoldNounpairs = 'include-gold-nounpairs'
    GoldNounpairAgreementNeeded = 'gold-nounpair-agreement-needed'
    GoldNounpairGreaterPot = 'gold-nounpair-greater-pot'
    GoldNounpairLesserPot = 'gold-nounpair-lesser-pot'
    GoldNounpairEqPot = 'gold-nounpair-eq-pot'
    AddRemainderAsNonseeds = 'add-remainder-as-nonseeds'
    FrameSeedMethod = 'frame-seed-method'
    NounpairSeedMethod = 'nounpair-seed-method'
    SelPrefPotMethod = 'selpref-pot-method'
    SelPrefEmbFilename = 'selpref-emb-filename'
    ObjpairSplit = 'objpair-split'
    FrameSplit = 'frame-split'

    # Class vars in all caps as constants for vals
    EVAL_DEV = 'dev'
    EVAL_TEST = 'test'

    SEL_PREF_FREQ = 'freq'
    SEL_PREF_PMI = 'pmi'

    POTENTIAL_METHOD_HARDCODED = 'hardcoded'
    POTENTIAL_METHOD_TRAINED = 'trained'
    POTENTIAL_METHOD_BOTH = 'both'

    # digging into more detail here for selpref
    SEL_PREF_HARDCODED = 'hardcoded'
    SEL_PREF_MLE = 'mle'
    SEL_PREF_EMB = 'emb'

    # unary potentials
    POT_UNARY_MEDIUM_BIGGER = np.array([0.7, 0.2, 0.1])
    POT_UNARY_MEDIUM_SMALLER = np.array([0.2, 0.7, 0.1])
    POT_UNARY_MEDIUM_EQ = np.array([0.15, 0.15, 0.7])

    POT_UNARY_STRONG_BIGGER = np.array([0.9, 0.07, 0.03])
    POT_UNARY_STRONG_SMALLER = np.array([0.07, 0.9, 0.03])
    POT_UNARY_STRONG_EQ = np.array([0.05, 0.05, 0.9])

    # binary potentials
    POT_BINARY_MEDIUM_SIM = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.15, 0.15, 0.7],
    ])
    POT_BINARY_MEDIUM_REV = np.array([
        [0.2, 0.7, 0.1],
        [0.7, 0.2, 0.1],
        [0.15, 0.15, 0.7],
    ])

    POT_BINARY_STRONG_SIM = np.array([
        [0.9, 0.07, 0.03],
        [0.07, 0.9, 0.03],
        [0.05, 0.05, 0.9],
    ])
    POT_BINARY_STRONG_REV = np.array([
        [0.07, 0.9, 0.03],
        [0.9, 0.07, 0.03],
        [0.05, 0.05, 0.9],
    ])

    @staticmethod
    def _get_default_map():
        return {
            Settings.Eval: Settings.EVAL_DEV,
            Settings.Attrs: ['size', 'weight', 'verb-speed', 'hardness', 'rigidness'],
            Settings.MaxSeeds: -1,  # -1 means no limit
            Settings.GloveVerbSimThresh: 0.5,
            Settings.GloveNounSimThresh: 0.45,
            Settings.VerbSimPot: Settings.POT_BINARY_MEDIUM_SIM,
            Settings.NounEqPot: Settings.POT_UNARY_MEDIUM_EQ,
            Settings.NounSimPot: Settings.POT_BINARY_MEDIUM_SIM,
            Settings.NounSimRevPot: Settings.POT_BINARY_MEDIUM_REV,
            Settings.MaxNounsPerFrame: 1,
            Settings.FilterAbstract: True,
            Settings.GTBiggerPot: Settings.POT_UNARY_MEDIUM_BIGGER,
            Settings.GTSmallerPot: Settings.POT_UNARY_MEDIUM_SMALLER,
            Settings.GTEqPot: Settings.POT_UNARY_MEDIUM_EQ,
            Settings.AgreementNeeded: 2,
            Settings.SelPrefFreqCutoff: 1000,
            Settings.SelPrefMinFreqForPMI: 1,
            Settings.SelPrefPMICutoff: 4.0,
            Settings.SelPrefMethod: Settings.SEL_PREF_PMI,
            Settings.SelPrefPot: Settings.POT_BINARY_MEDIUM_SIM,
            Settings.NormalizeLBP: True,
            Settings.LBPMaxIters: 20,
            Settings.IncludeSelPrefFactors: True,
            Settings.IncludeXgraph: True,
            Settings.IncludeVerbSimFactors: True,
            Settings.IncludeNounSimFactors: True,
            Settings.IncludeInfWithinverbSimframeFactors: True,
            Settings.WithinverbSimframePot: Settings.POT_BINARY_MEDIUM_SIM,
            Settings.XgraphTuples: [
                ('size', 'weight'),
                ('size', 'hardness'),
                ('weight', 'hardness'),
            ],
            Settings.XgraphPot: Settings.POT_BINARY_MEDIUM_SIM,
            Settings.RawNounsFilename: '',
            Settings.EvalNounsFilename: '',
            Settings.Lemmatize: True,
            Settings.IncludeNgramDBNouns: False,
            Settings.IncludeGoldNounpairs: True,
            Settings.GoldNounpairAgreementNeeded: 2,
            Settings.GoldNounpairGreaterPot: Settings.POT_UNARY_MEDIUM_BIGGER,
            Settings.GoldNounpairLesserPot: Settings.POT_UNARY_MEDIUM_SMALLER,
            Settings.GoldNounpairEqPot: Settings.POT_UNARY_MEDIUM_EQ,
            Settings.AddRemainderAsNonseeds: True,
            Settings.FrameSeedMethod: Settings.POTENTIAL_METHOD_BOTH,
            Settings.NounpairSeedMethod: Settings.POTENTIAL_METHOD_BOTH,
            Settings.SelPrefPotMethod: Settings.SEL_PREF_HARDCODED,
            Settings.SelPrefEmbFilename: '',
            Settings.ObjpairSplit: 20,
            Settings.FrameSplit: 5,
        }

    def __init__(self, logger=None):
        """
        Sets dict with default settings.

        Settings to do:
        - [x] constants above
        - [x] number of nounsp
        - [x] Potentials (bigger, smaller, eq)
        - [x] Agreement needed (x/3)
        - [x] Verb sim fac pots
        - [x] Noun sim fac pots
        - [x] Sel pref pots
        - [x] Sel pref cutoff
        - [x] whether to normalize in lbp
        - [x] max n iterations to run lbp for
        - [x] which factors to add
        - [x] whether to filter abstract nouns
        - [x] check out data.py settings
        - [x] check rest of this file
        """
        # Some admin
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

        # Default values
        self._params = Settings._get_default_map()
        self.param_keys = []
        self.param_iterator = None

    def get(self, key):
        return self._params[key]

    def _setup_trial(self, trial_keys):
        """
        Tracks which configs vary (are "trial" keys).

        Args:
            trial_keys ([str])
        """
        self.default_keys = set(self._get_default_map().keys()) - set(trial_keys)
        self.trial_keys = trial_keys
        self.trial_num = 0
        self.trial_log = {}
        self.trial_results = {}
        self.trial_results_all_keys = []

    def trial_sequence(self, params):
        """
        Sets up a trial to try the specified ranges of parameter values in
        sequence (holding all other parameters to their defaults and varying
        only one at a time).

        Args:
            params ({Settings.KEY: [list of values to try]})
        """
        self._setup_trial(params.keys())

        # This implementation is kind of gross because it's bolted onto how the
        # trial_product was designed. We really want to iterate over both keys
        # and values and just set what we want. But I'm too lazy to learn about
        # how iterators work in python. So we just use all the keys.
        dm = self._get_default_map()
        keys = dm.keys()
        vals = [dm[k] for k in keys]
        trials = []
        for k, v in params.iteritems():
            kidx = keys.index(k)
            for val in v:
                trial = vals[:]
                trial[kidx] = val
                trials += [tuple(trial)]
        self.param_keys = keys
        self.param_iterator = iter(trials)

    def trial_product(self, params):
        """
        Sets up a trial to try the product (all exponentially many
        combinations) of the specified ranges of parameter values.

        Args: params ({Settings.KEY: [list of values to try]})
        """
        self._setup_trial(params.keys())

        param_keys = []
        param_vals = []
        for k,v in params.iteritems():
            param_keys += [k]
            param_vals += [v]

        # self.current_indices = [-1 for _ in range(len(param_keys))]
        self.param_keys = param_keys
        self.param_iterator = product(*param_vals)

    def next(self):
        """
        Move on to the next parameter setting combination.

        Returns:
            bool Whether there's anything left
        """
        next_params = next(self.param_iterator, Settings.NothingLeft)
        if next_params is Settings.NothingLeft:
            return False
        assert len(next_params) == len(self.param_keys)

        self.trial_num += 1
        self.trial_log[self.trial_num] = {}
        self.trial_results[self.trial_num] = {}
        for i, k in enumerate(self.param_keys):
            self._params[k] = next_params[i]
            self.trial_log[self.trial_num][k] = self._params[k]
        return True

    def add_result(self, key, val):
        """
        Adds result in form of key: val *to currently running trial*.

        Args:
            key (any hashable)
            val (any)
        """
        if key not in self.trial_results_all_keys:
            self.trial_results_all_keys.append(key)
        self.trial_results[self.trial_num][key] = val

    def log_results(self):
        """
        Logs results. Call after trials have finished.

        First logs the config that didn't change.

        Then logs a table of the experiments run and any results that were
        added.
        """
        self.logger.info('Static config (defaults):')
        full_dm = self._get_default_map()
        pure_dm = {k: cell_massage(v) for k,v in full_dm.iteritems() if k in self.default_keys}
        list_pure_dm = [list(item) for item in pure_dm.iteritems()]
        for line in tabulate(list_pure_dm, tablefmt="fancy_grid").split('\n'):
            self.logger.info(line)

        self.logger.info('Trial configs:')
        rows = []
        for i in sorted(self.trial_log.keys()):
            row = {}
            # settings
            for tk in self.trial_keys:
                row[tk] = cell_massage(self.trial_log[i][tk])
            # ... then results
            for rk in self.trial_results_all_keys:
                val = '---'
                if rk in self.trial_results[i]:
                    val = cell_massage(self.trial_results[i][rk])
                row[rk] = val
            rows.append(row)

        # TODO: use ordereddict and set key order so table headers go settings
        # and then results.
        # headers = self.trial_keys + self.trial_results_all_keys
        for line in tabulate(rows, headers="keys", tablefmt="fancy_grid").split('\n'):
            self.logger.info(line)

    def debug_log_config(self):
        """
        Dumps full config to debug log.
        """
        self.logger.debug('Settings:')
        for k,v in self._params.iteritems():
            self.logger.debug('%(key)25s: %(val)s' % {'key': k, 'val': v})
