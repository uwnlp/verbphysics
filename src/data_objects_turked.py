"""
Loading up the (processed) turked *OBJECT PAIR* data.

author: mbforbes
"""

# IMPORTS
# -----------------------------------------------------------------------------

# stdlib
import logging
import sys

# 3rd party
import pandas as pd


# CONSTANTS
# -----------------------------------------------------------------------------

DIR_5 = 'data/verbphysics/objects/train-5/'
DIR_20 = 'data/verbphysics/objects/train-20/'

# The attr names are inconsistent in my code. This translates from external
# (e.g. AttrGraph) names to internal (turked object pairs) names.
ATTR_TRANSLATION = {
    'hardness': 'strength',
    'verb-speed': 'speed',
}

# For internal sanity checking: the complete list of attributes.
OUR_ATTRS = ['size', 'weight', 'strength', 'rigidness', 'speed']

# obj1 vs obj2, where vs is one of:
LABEL_GREATER = 1
LABEL_EQ = 0
LABEL_LESSER = -1
LABEL_UNK = -42

PERSON_DATA = 'person'
PERSON_TOKEN = 'PERSON'

logger = logging.getLogger(__name__)


# CLASSES
# -----------------------------------------------------------------------------

class DataObjectsTurked(object):

    @staticmethod
    def load_raw(partition, attr_raw, agreement_needed, remove_unk=True, directory=DIR_5):
        """
        Loads up partition, filtering out those with agreement <
        agreement_needed and those with majority of 'unk'.

        Args:
            partition (str): 'train'/'dev'/'test'
            attr_raw (str): 'size', 'weight', 'hardness' (mapped to 'strength'),
                'rigidness', 'verb-speed' (mapped to 'speed')
            agreement_needed (int): 2 or 3
            directory (str): directory to load data from. use DIR_20 to use 20%
                of data, DIR_5 to use 5%

        Returns:
            [[str, str, int]]: [[obj1, obj2, majority label]]
        """
        # translation, if needed
        attr = attr_raw if attr_raw not in ATTR_TRANSLATION else ATTR_TRANSLATION[attr_raw]
        if attr not in OUR_ATTRS:
            logger.error('Unknown attribute: "%s"' % (attr))
            sys.exit(1)

        # load, filter, and transform to list
        fn = directory + partition + '.csv'
        df = pd.read_csv(fn)
        filtered = df[(df[attr + '-agree'] >= agreement_needed)]
        if remove_unk:
            filtered = filtered[(filtered[attr + '-maj'] != LABEL_UNK)]
        data = filtered[['obj1', 'obj2', attr + '-maj']]
        lst = data.values.tolist()

        # switch our lowercased person token to the original
        for l in lst:
            for i in [0, 1]:
                if l[i] == PERSON_DATA:
                    l[i] = PERSON_TOKEN

        return lst

    @staticmethod
    def load(partition, attr_raw, agreement_needed, greater_pot, eq_pot, lesser_pot, split):
        """
        Loads up partition, filtering out those with agreement <
        agreement_needed and those with majority of 'unk'. Then changes gold
        labels to the provided potentials.

        Args:
            partition (str): 'train'/'dev'/'test'
            attr_raw (str): 'size', 'weight', 'hardness' (mapped to 'strength'),
                'rigidness', 'verb-speed' (mapped to 'speed')
            agreement_needed (int): 2 or 3
            greater_pot (np.ndarray: 1 x 3)
            eq_pot (np.ndarray: 1 x 3)
            lessert_pot (np.ndarray: 1 x 3)

        Returns:
            [[str, str, np.ndarray]]: [[obj1, obj2, potential]]
        """
        if split == 5:
            directory = DIR_5
        elif split == 20:
            directory = DIR_20
        else:
            logger.error('Unimplemented split: %r', split)
            sys.exit(1)
        lst = DataObjectsTurked.load_raw(partition, attr_raw, agreement_needed, True, directory)

        # create our own mini mapping for assigning potentials below
        potmap = {
            LABEL_GREATER: greater_pot,
            LABEL_EQ: eq_pot,
            LABEL_LESSER: lesser_pot,
        }

        # replace each list's label with the corresponding passed potential
        for l in lst:
            l[-1] = potmap[l[-1]]

        return lst
