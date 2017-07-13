"""
data_turked is for turked gold annotations of *FRAMES*. Often imported as 'td'
as in 'TurkedData'.

author: mbforbes
"""

# IMPORTS
# -----------------------------------------------------------------------------

# builtins
import logging
import os

# 3rd party
import pandas as pd


# CONSTANTS
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

LABEL_GREATER = 1
LABEL_EQ = 0
LABEL_LESSER = -1
LABEL_UNK = -42

# The processed file with frame data.
PROCESSED_FILE = 'data/verbphysics/action-frames/action-frames.csv'

# Directories where we saved train/dev/test splits.
SPLIT_DIR_5 = 'data/verbphysics/action-frames/train-5/'
SPLIT_DIR_20 = 'data/verbphysics/action-frames/train-20/'


# CLASSES
# -----------------------------------------------------------------------------

class TurkedData(object):
    """
    Methods for loading / working with the turked data format, which is a dict
    of the form (henceforth known as TurkedDict):

            {
                'size': [
                    (verb_sub[_prep]_1, np.ndarray),
                    (verb_sub[_prep]_2, np.ndarray),
                    ...
                ],
                'weight': [
                    ...
                ],
                ...
            }
    """

    @staticmethod
    def load(fn, agreement_needed, bigger_pot, smaller_pot, eq_pot):
        """
        Loads Turked data using specified settings.

        NOTE(mbforbes): Could do different potentials for stronger agreement.

        Args:
            fn (str): Pandas converted CSV file. To generate this, run
                `notebooks/verb_process.ipynb`.

            agreement_needed (int): number out of 3 of agreement needed before
                using a turked data point

            bigger_pot (np.ndarray of 1x3) Potentials for data with "bigger" GT.

            smaller_pot (np.ndarray of 1x3) Potentials for data with "smaller"
            GT.

            eq_pot (np.ndarray of 1x3) Potentials for data with "equal" GT.

        Returns:
            TurkedDict

        """
        label_pot_map = {
            LABEL_GREATER: bigger_pot,
            LABEL_EQ: eq_pot,
            LABEL_LESSER: smaller_pot,
        }

        # select attributes to load
        attrs = ['size', 'weight', 'verb-speed', 'hardness', 'rigidness']

        # load up
        df = pd.read_csv(fn)

        res = {}
        for attr in attrs:
            col_ag = attr + '-agree'
            col_maj = attr + '-maj'
            tuples = []

            # Pick only rows that agree on a non-UNK result.
            data = df[(df[col_ag] >= agreement_needed) & (df[col_maj] != LABEL_UNK)]

            # NOTE(mbforbes): We could have two variants of the potentials, one
            # for unanimous agreement, and a less strong one for 2/3 agreement.
            # This uses one for all.
            for _, row in data.iterrows():
                v, s, p = row['verb'], row['sub'], row['prep']
                name = TurkedData.vsp_to_str(v, s, p)
                pot = label_pot_map[row[col_maj]]
                tuples += [(name, pot)]

            # Save this attribute's tuples
            res[attr] = tuples
        return res

    @staticmethod
    def load_raw(partition, attr, agreement_needed, fn=PROCESSED_FILE, split_dir=SPLIT_DIR_5):
        """
        Loads up `attr` data of `partition` of csv file `fn`, filtering out
        those with agreement < agreement_needed and those with majority of 'unk'.

        Returns list where each item is:

            (verb, preposition|None, gold_label, one-hot vector of frame type, v_s_p)

        Where gold_label is one of:

            LABEL_GREATER (1)
            LABEL_EQ (0)
            LABEL_LESSER (-1)

        Frame type is a 5-d one-hot vector of frame type:

            [_d, _p, _od, _op, _dp]

        And v_s_p is the string representation of the full frame, i.e.,

            verb_sub_preposition

        Args:
            partition (str) one of {'train', 'dev', 'test'}
            attr (str)
            agreement_needed (int)
            fn (str) location of the processed csv data file
            d (str, default=SPLIT_DIR_5) directory where we find train / dev /
                test verb splits

        Returns:
            [(str, str|None, int, [int], str)]
        """
        # load up verbs in partition list
        with open(os.path.join(split_dir, partition + '.txt')) as f:
            verbs = set([line.strip() for line in f.readlines()])

        # load full data
        df = pd.read_csv(fn)

        # filter agreement_needed for attr (and out maj unks). Wanted to filter
        # verbs here but can't broadcast the 'in set' operation, I guess.
        filtered = df[
            (df[attr + '-agree'] >= agreement_needed) &
            (df[attr + '-maj'] != LABEL_UNK)]

        # sub -> one hot vector. This could be represented more concisely (e.g.,
        # the index to one-hot), but this is clearer to look at.
        sub_to_onehot = {
            '_d':  [1, 0, 0, 0, 0],
            '_p':  [0, 1, 0, 0, 0],
            '_od': [0, 0, 1, 0, 0],
            '_op': [0, 0, 0, 1, 0],
            '_dp': [0, 0, 0, 0, 1],
        }

        # create results, filtering out verbs not in partition
        res = []
        for _, row in filtered.iterrows():
            if row['verb'] not in verbs:
                continue
            res.append((
                row['verb'],
                row['prep'] if not pd.isnull(row['prep']) else None,
                row[attr + '-maj'],
                sub_to_onehot[row['sub']],
                TurkedData.vsp_to_str(row['verb'], row['sub'], row['prep']),
            ))
        return res

    @staticmethod
    def n_verbs(data):
        """
        Counts the number of unique verbs in data.

        Args:
            data (single attribute list (element) of TurkedDict)

        Returns:
            int
        """
        verbs = []
        for d in data:
            node = d[0]
            v, _, _ = TurkedData.str_to_vsp(node)
            verbs += [v]
        return len(set(verbs))

    @staticmethod
    def train_dev_test_split(data, d):
        """
        Splits data into train, dev, test sections.

        Args:
            data (single attribute list (element) of TurkedDict)

            d (str) Directory in which we find train.txt, dev.txt, test.txt
                files, which are just one verb per line verb list files.

        Returns:
            3x tuple, each is a list just like the input data arg.
        """
        # Load up verb lists
        verb_splits = ['train', 'dev', 'test']
        verb_map = {}
        for s in verb_splits:
            with open(os.path.join(d, s + '.txt')) as f:
                for v in f.readlines():
                    verb_map[v.strip()] = s

        # Split data
        data_splits = {
            'train': [],
            'dev': [],
            'test': [],
        }
        for datum in data:
            v, _, _ = TurkedData.str_to_vsp(datum[0])
            split = verb_map[v]
            data_splits[split] += [datum]

        # Log info
        total_frames = sum([len(frames) for _, frames in data_splits.iteritems()])
        logger.debug('Data splits:')
        for split in verb_splits:
            frames = data_splits[split]
            logger.debug('\t%s: %d frames (%0.2f%%) (%d verbs)', split, len(frames), float(len(frames) * 100) / total_frames, TurkedData.n_verbs(frames))

        return data_splits['train'], data_splits['dev'], data_splits['test']

    @staticmethod
    def str_to_vsp(node):
        """
        Args:
            str

        Returns:
            str      (verb),
            str      (_sub) (yes, includes the '_'),
            str|None (prep) (or None if sub has no prep)
        """
        pieces = node.split('_')
        if len(pieces) == 2:
            # No prep
            return pieces[0], '_' + pieces[1], None
        elif len(pieces) == 3:
            # Has a prep
            return pieces[0], '_' + pieces[1], pieces[2]
        else:
            assert False, 'Malformed node string: %r' % (node)

    @staticmethod
    def vsp_to_str(v, s, p):
        """
        Args:
            v (str) verb
            s (str) sub (_p, _d, etc.)
            p (str) preposition

        Returns:
            str
        """
        res = v + s
        if not pd.isnull(p):
            res += '_' + p
        return res
