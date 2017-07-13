"""
Factors whose potentials are trained (on the training data) (duh).

author: mbforbes
"""

# IMPORTS
# -----------------------------------------------------------------------------

# stdlib
import code  # code.interact(local=dict(globals(), **locals()))
from collections import Counter
import sys

# 3rd party
import numpy as np
import pandas as pd

# local
import data_turked as td
import data_objects_turked as dot


# CLASSES
# -----------------------------------------------------------------------------

class UnaryFrameEmbedding(object):

    def __init__(self, framesplit):
        """
        Args:
            framesplit (int)
        """
        if framesplit == 5:
            filename = 'data/emb/frames-train5.csv'
        elif framesplit == 20:
            filename = 'data/emb/frames-train20.csv'
        else:
            print 'ERROR: Unknown frame split %r' % (framesplit)
            sys.exit(1)

        self.df = pd.read_csv(filename)

    def get(self, attr, framestr):
        """
        Args:
            attr (str)
            framestr (str)

        Returns:
            np.ndarray of shape (3,) representing
                [p(>), p(<), p(=)]
        """
        row = self.df[(self.df['attr'] == attr) & (self.df['framestr'] == framestr)]
        return row[['prob_greater', 'prob_lesser', 'prob_eq']].get_values().flatten()


class UnaryObjpairEmbedding(object):

    def __init__(self, objpairsplit):
        """
        Args:
            objpairsplit (int)
        """
        if objpairsplit == 5:
            filename = 'data/emb/objpairs-train5.csv'
        elif objpairsplit == 20:
            filename = 'data/emb/objpairs-train20.csv'
        else:
            print 'ERROR: Unknown objpair split %r' % (objpairsplit)
            sys.exit(1)

        self.df = pd.read_csv(filename)

    def get(self, attr, obj1, obj2):
        """
        Args:
            attr (str)
            obj1 (str)
            obj2 (str)

        Returns:
            np.ndarray of shape (3,) representing
                [p(>), p(<), p(=)]
        """
        row = self.df[
            (self.df['attr'] == attr) &
            (self.df['obj1'] == obj1) &
            (self.df['obj2'] == obj2)]
        return row[['prob_greater', 'prob_lesser', 'prob_eq']].get_values().flatten()


class SelPrefEmbedding(object):

    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    def get(self, attr, frame, obj1, obj2):
        """
        Args:
            attr (str)
            frame (str)
            obj1 (str)
            obj2 (str)

        Returns:
            np.ndarray of shape (3,3) representing
                        objp >  objp =  objp <
                frame > [[ p       p       p]
                frame =  [ p       p       p]
                frame <  [ p       p       p]]
        """
        row = self.df[
            (self.df['attr'] == attr) &
            (self.df['frame'] == frame) &
            (self.df['obj1'] == obj1) &
            (self.df['obj2'] == obj2)]
        try:
            return row[['gg', 'ge', 'gl', 'eg', 'ee', 'el', 'lg', 'le', 'll']].get_values().flatten().reshape((3,3))
        except:
            code.interact(local=dict(globals(), **locals()))


class SelPrefMLE(object):

    def __init__(self, pmi):
        """
        Args:
            pmi (ngramdb.PMI)
        """
        self.pmi = pmi

    def get(self, attr, frame_agreement_needed, objpair_agreement_needed,
            pmi_cutoff, objsplit):
        """
        Args:
            attr (str)
            frame_agreement_needed (int)
            objpair_agreement_needed (int)
            pmi_cutoff (float)
            objsplit (int)

        Returns:
            np.ndarray (3,3) selectional preference potential (frame, objpair)
                for attr
        """
        if objsplit == 5:
            objdir = dot.DIR_5
        elif objsplit == 20:
            objdir = dot.DIR_20
        else:
            print 'ERROR: Unimplemented split: %r' % (split)
            sys.exit(1)

        frames_expanded = td.TurkedData.load_raw(
            'train', attr, frame_agreement_needed)
        # pull off just v_s_p str and gold label
        frames = [(fe[4], fe[2]) for fe in frames_expanded]
        objpairs = dot.DataObjectsTurked.load_raw(
            'train', attr, objpair_agreement_needed, True, objdir)

        # counts maps frame gold -> objpair gold. init'ing now rather than
        # checking for missing later.
        counts = {
            td.LABEL_GREATER: Counter(),
            td.LABEL_LESSER: Counter(),
            td.LABEL_EQ: Counter(),
        }
        for f in frames:
            framestr, frame_gold = f
            for o in objpairs:
                obj1, obj2, objpair_gold = o

                # get PMI. only count if >= cutoff
                pmi_score = self.pmi.query(framestr, (obj1, obj2))
                if pmi_score >= pmi_cutoff:
                    counts[frame_gold][objpair_gold] += 1

        flat = np.array([
            float(counts[td.LABEL_GREATER][dot.LABEL_GREATER]),
            float(counts[td.LABEL_GREATER][dot.LABEL_LESSER]),
            float(counts[td.LABEL_GREATER][dot.LABEL_EQ]),
            float(counts[td.LABEL_LESSER][dot.LABEL_GREATER]),
            float(counts[td.LABEL_LESSER][dot.LABEL_LESSER]),
            float(counts[td.LABEL_LESSER][dot.LABEL_EQ]),
            float(counts[td.LABEL_EQ][dot.LABEL_GREATER]),
            float(counts[td.LABEL_EQ][dot.LABEL_LESSER]),
            float(counts[td.LABEL_EQ][dot.LABEL_EQ]),
        ])

        # per-row norm (i.e. marginal)
        res = flat.reshape((3,3))
        for i in range(res.shape[0]):
            res[i, :] /= sum(res[i, :])

        return res
