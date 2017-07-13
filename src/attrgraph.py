"""
Handles building / running / evaluating one attribute's (size, weight, etc.)
factor graph.

author: mbforbes
"""

# IMPORTS
# -----------------------------------------------------------------------------

# builtins
from collections import Counter
import json
import logging
import os

# 3rd party
import factorgraph as fg
import numpy as np

# locals
import data as ng
import data_objects_turked as dot
import data_turked as td
from settings import Settings
import trained_factors as tf
import util


# Possible values for each RV.
RV_LABELS = ['>', '<', '=']

# This is arbitrary but likely won't need to change. It's how we construct RV
# names.
VS = '_vs_'


class AttrGraph(object):
    """
    Graph for a single attribute.
    """

    def __init__(self, glv, ngramdb, pmi, turked_data, name, eval_mode,
            framesplitdir):
        """
        Args:
            glove (glove.Glove)
            ngramdb (ng.Data)
            pmi (ng.PMI)
            turked_data (see data_turked for format)
            name (str) This attribute's name (size, weight, etc.)
            eval_mode (str) in {Settings.EVAL_DEV, Settings.EVAL_TEST}
            framesplitdir (str) Directory containing frame splits
        """
        self.logger = logging.getLogger(name)
        self.logger.debug('Initializing')
        self.graph = fg.Graph(debug=False)
        self.glove = glv
        self.ngramdb = ngramdb
        self.pmi = pmi
        self.name = name

        # Split up the data.
        train_data, dev_data, test_data = td.TurkedData.train_dev_test_split(
            turked_data[name],
            framesplitdir
        )

        # Partition & save. Train is always the same, may eval on dev or test.
        self.seed_data = train_data
        if eval_mode == Settings.EVAL_DEV:
            self.nonseed_data = dev_data
        elif eval_mode == Settings.EVAL_TEST:
            self.nonseed_data = test_data
        else:
            assert False, 'Unknown eval mode: "%s"' % (eval_mode)

        self.logger.debug('Using the following data:')
        self.logger.debug(
            '\tseeds:    %d frames (%d verbs)', len(self.seed_data),
            td.TurkedData.n_verbs(self.seed_data))
        self.logger.debug(
            '\tnonseeds: %d frames (%d verbs)', len(self.nonseed_data),
            td.TurkedData.n_verbs(self.nonseed_data))

        # For clarity, init'ing stuff that gets init'd later.
        # Are you supposed to do this in python?
        self._orig_rvs = []
        self.nonseed_gold_np_rvs = []

    def build(self, settings):
        """
        Constructs the graph.

        Args:
            settings (Settings) How to construct the graph; uses current
                config.
        """

        # Frames
        self.logger.debug('[build] Adding frames...')
        frame_seed_method = settings.get(Settings.FrameSeedMethod)
        if frame_seed_method == Settings.POTENTIAL_METHOD_HARDCODED:
            # Use hand-picked values of high strength (e.g. 0.7 or 0.9 for
            # correct choice) *only* on seed data.
            seed_rvs = self.add_seeds(
                self.seed_data,
                settings.get(Settings.MaxSeeds),
                settings.get(Settings.AddRemainderAsNonseeds))
            nonseed_rvs = self.add_nonseeds(self.nonseed_data)
            verb_rvs = seed_rvs + nonseed_rvs
        elif frame_seed_method == Settings.POTENTIAL_METHOD_TRAINED:
            # Use LR-output guesses as potential weights for *all* data (not
            # only seeds). Now there is no true GT data, but all frames have
            # potential.
            verb_rvs = self.add_frames_emb(settings.get(Settings.FrameSplit))
        elif frame_seed_method == Settings.POTENTIAL_METHOD_BOTH:
            # LR-output plus seed factors
            verb_rvs = self.add_frames_emb(settings.get(Settings.FrameSplit))
            self.add_seed_facs_to_frames(self.seed_data)
        else:
            assert False, 'Unknown frame seed method: "%s"' % (
                frame_seed_method)

        # Gold nounpairs
        gold_np_names = []
        if settings.get(Settings.IncludeGoldNounpairs):
            self.logger.debug('[build] Adding gold-turked nounpairs...')
            seed_gold_np_data = dot.DataObjectsTurked.load(
                'train',
                self.name,
                settings.get(Settings.GoldNounpairAgreementNeeded),
                settings.get(Settings.GoldNounpairGreaterPot),
                settings.get(Settings.GoldNounpairEqPot),
                settings.get(Settings.GoldNounpairLesserPot),
                settings.get(Settings.ObjpairSplit),
            )
            nonseed_gold_np_data = dot.DataObjectsTurked.load(
                settings.get(Settings.Eval),
                self.name,
                settings.get(Settings.GoldNounpairAgreementNeeded),
                settings.get(Settings.GoldNounpairGreaterPot),
                settings.get(Settings.GoldNounpairEqPot),
                settings.get(Settings.GoldNounpairLesserPot),
                settings.get(Settings.ObjpairSplit),
            )
            self.logger.debug(
                '[build] \t adding %d gold-turked nounpairs (%d seed) ' +
                    '(%d nonseed)',
                len(seed_gold_np_data) + len(nonseed_gold_np_data),
                len(seed_gold_np_data),
                len(nonseed_gold_np_data))
            # pick method here and add them. need to save nonseeds before they
            # might get pruned. need for use in eval.
            nounpair_seed_method = settings.get(Settings.NounpairSeedMethod)
            if nounpair_seed_method == Settings.POTENTIAL_METHOD_HARDCODED:
                # add with hardcoded potentials to seeds only
                seed_gold_np_rvs = self.add_seed_nounpairs(seed_gold_np_data)
                self.nonseed_gold_np_rvs = self.add_nonseed_nounpairs(
                    nonseed_gold_np_data)
            elif nounpair_seed_method == Settings.POTENTIAL_METHOD_TRAINED:
                # add with embedding potentials to all
                seed_gold_np_rvs, self.nonseed_gold_np_rvs = self.add_goldnps_emb(
                    seed_gold_np_data,
                    nonseed_gold_np_data,
                    settings.get(Settings.ObjpairSplit)
                )
            elif nounpair_seed_method == Settings.POTENTIAL_METHOD_BOTH:
                # embeddings
                seed_gold_np_rvs, self.nonseed_gold_np_rvs = self.add_goldnps_emb(
                    seed_gold_np_data,
                    nonseed_gold_np_data,
                    settings.get(Settings.ObjpairSplit)
                )
                # tack on seeds at end
                self.add_seed_facs_to_gold_nps(seed_gold_np_data)
            else:
                assert False, 'Unknown nounpair seed method: "%s"' % (
                    nounpair_seed_method)
            gold_np_names = [str(rv) for rv in seed_gold_np_rvs + self.nonseed_gold_np_rvs]

        # In below section we'll (maybe) get unlabeled nouns from various
        # sources.
        raw_nouns = []

        # Get ngramdb nouns
        if settings.get(Settings.IncludeNgramDBNouns):
            self.logger.debug('[build] Getting ngramdb nouns (max %d/frame) ' +
                '(filterabstract=%r) (lemmatize=%r)...',
                settings.get(Settings.MaxNounsPerFrame),
                settings.get(Settings.FilterAbstract),
                settings.get(Settings.Lemmatize))
            nouns_ngramdb = self.get_ng_nouns(
                verb_rvs,
                settings.get(Settings.MaxNounsPerFrame),
                settings.get(Settings.FilterAbstract),
                settings.get(Settings.Lemmatize))
            self.logger.debug(
                '[build] \t got %d ngramdb nouns',
                len(nouns_ngramdb))
            raw_nouns += nouns_ngramdb

        # Get external nouns
        nounfile = settings.get(Settings.RawNounsFilename)
        if len(nounfile) > 0:
            self.logger.debug('[build] Getting external nouns from %s...' % (nounfile))
            nouns_ext = self.get_ext_nouns(nounfile)
            self.logger.debug('[build] \t got %d external nouns', len(nouns_ext))
            raw_nouns += nouns_ext

        # Add all nouns
        self.logger.debug('[build] Adding all non-gold nouns (as pairs) to graph...')
        nouns, noun_rvs = self.add_nouns(raw_nouns)
        self.logger.debug('[build] Added %d unique non-gold nouns (%s RVs) to graph', len(nouns), len(noun_rvs))

        # Get master noun list
        nounset = set(nouns)
        for np_name in gold_np_names:
            pieces = np_name.split(VS)
            for p in pieces:
                nounset.add(p)
        nouns = list(nounset)

        # Factors
        if settings.get(Settings.IncludeVerbSimFactors):
            self.logger.debug('[build] Adding verb similarity factors...')
            verb_sim_facs = self.add_fac_verb_sim(
                verb_rvs,
                settings.get(Settings.GloveVerbSimThresh),
                settings.get(Settings.VerbSimPot))
            self.logger.debug('[build] \t added %d verb sim factors', len(verb_sim_facs))

        if settings.get(Settings.IncludeNounSimFactors):
            self.logger.debug('[build] Adding noun similarity factors...')
            noun_sim_facs = self.add_fac_noun_sim(
                nouns,
                settings.get(Settings.GloveNounSimThresh),
                settings.get(Settings.NounEqPot),
                settings.get(Settings.NounSimPot),
                settings.get(Settings.NounSimRevPot))
            self.logger.debug('[build] \t added %d noun sim factors', len(noun_sim_facs))

        if settings.get(Settings.IncludeInfWithinverbSimframeFactors):
            self.logger.debug('[build] Adding within-verb similar frame inference factors...')
            withinverb_simframe_facs = self.add_fac_inf_simframe(
                verb_rvs,
                settings.get(Settings.WithinverbSimframePot))
            self.logger.debug(
                '[build] \t added %d within-verb similar frame inference factors',
                len(withinverb_simframe_facs))

        if settings.get(Settings.IncludeSelPrefFactors):
            method = settings.get(Settings.SelPrefMethod)
            if method == Settings.SEL_PREF_PMI:
                cutoff = settings.get(Settings.SelPrefPMICutoff)
            elif method == Settings.SEL_PREF_FREQ:
                cutoff = settings.get(Settings.SelPrefFreqCutoff)
            else:
                assert False, 'Unknown sel pref method: "%s"' % (method)

            # mle changes from hardcoded values; trained values get picked
            # within function itself. anyway so not checking.
            # NOTE: freq method broken; only PMI currently usable.
            selpref_pot_method = settings.get(Settings.SelPrefPotMethod)
            if selpref_pot_method == Settings.SEL_PREF_MLE:
                sel_pref_pot = tf.SelPrefMLE(self.pmi).get(
                    self.name,
                    settings.get(Settings.AgreementNeeded),
                    settings.get(Settings.GoldNounpairAgreementNeeded),
                    settings.get(Settings.SelPrefPMICutoff),
                    settings.get(Settings.ObjpairSplit),
                )
            elif selpref_pot_method == Settings.SEL_PREF_HARDCODED or \
                    selpref_pot_method == Settings.SEL_PREF_EMB:
                sel_pref_pot = settings.get(Settings.SelPrefPot)
            else:
                assert False, 'Unknown sel pref potential method: "%s"' % (selpref_pot_method)

            self.logger.debug('[build] Adding selectional preference factors...')
            sel_pref_facs = self.add_fac_sel_pref(
                verb_rvs,
                method,
                cutoff,
                sel_pref_pot,
                settings.get(Settings.SelPrefMinFreqForPMI),
                selpref_pot_method == Settings.SEL_PREF_EMB,
                settings.get(Settings.SelPrefEmbFilename))
            self.logger.debug('[build] \t added %d sel-pref factors', len(sel_pref_facs))

        # Save for later logging
        self._orig_rvs = self.graph.get_rvs().values()

        self.logger.debug('[build] Removing loner RVs...')
        n_rvs_removed = self.graph.remove_loner_rvs()
        # self.logger.debug('Removed %d loner RVs', n_rvs_removed)

        self.graph.debug_stats()
        self.debug_stats(settings)

    def debug_stats(self, settings):
        """
        Print info about this attribute's factor graph.

        NOTE: this should probably be done with either:
         (A) a custom stats object
         (B) pandas

        Args:
            settings (Settings)
        """
        rvs = self._orig_rvs
        seed_frame_rvs, nonseed_frame_rvs, pruned_seed_frame_rvs, pruned_nonseed_frame_rvs, final_seed_rvs, final_nonseed_rvs = [], [], [], [], [], []
        nounpair_rvs, pruned_nounpair_rvs, final_nounpair_rvs = [], [], []
        for rv in rvs:
            if rv.meta['type'] == 'frame':
                if rv.meta['seed']:
                    seed_frame_rvs.append(rv)
                    if rv.meta['pruned']:
                        pruned_seed_frame_rvs.append(rv)
                    else:
                        final_seed_rvs.append(rv)
                else:
                    nonseed_frame_rvs.append(rv)
                    if rv.meta['pruned']:
                        pruned_nonseed_frame_rvs.append(rv)
                    else:
                        final_nonseed_rvs.append(rv)
            elif rv.meta['type'] == 'noun':
                nounpair_rvs.append(rv)
                if rv.meta['pruned']:
                    pruned_nounpair_rvs.append(rv)
                else:
                    final_nounpair_rvs.append(rv)
            else:
                self.logger.warn('Error: RV with unexpected type: %r', rv.meta['type'])

        n_rvs = len(rvs)
        n_orig_seeds = len(seed_frame_rvs)
        n_pruned_seeds = len(pruned_seed_frame_rvs)  # should be 0
        n_final_seeds = len(final_seed_rvs)
        n_orig_nonseeds = len(nonseed_frame_rvs)
        n_pruned_nonseeds = len(pruned_nonseed_frame_rvs)
        n_final_nonseeds = len(final_nonseed_rvs)

        n_orig_nounpairs = len(nounpair_rvs)
        n_pruned_nounpairs = len(pruned_nounpair_rvs)
        n_final_nounpairs = len(final_nounpair_rvs)

        self.logger.debug('Graph stats:')
        self.logger.debug('\t %d RVs', n_rvs)
        self.logger.debug('\t \t %d frames', n_final_seeds + n_final_nonseeds)
        self.logger.debug('\t \t \t %d seeds (%d original, %d pruned)',
            n_final_seeds, n_orig_seeds, n_pruned_seeds)
        self.logger.debug('\t \t \t %d non-seeds (%d original, %d pruned)',
            n_final_nonseeds, n_orig_nonseeds, n_pruned_nonseeds)
        self.logger.debug('\t \t %d noun-pairs (%d original, %d pruned)', n_final_nounpairs, n_orig_nounpairs, n_pruned_nounpairs)

        # Code below adds noun pair stats to settings.
        # settings.add_result('%s noun pair RVs (orig)' % self.name, n_orig_nounpairs)
        # settings.add_result('%s noun pair RVs (after pruning)' % self.name, n_final_nounpairs)
        # settings.add_result('%s pruned frame RVs' % (self.name), n_pruned_nonseeds)

        # get all factor types
        self.logger.debug('RV connectivity stats:')
        tps = set()
        # code.interact(local=dict(globals(), **locals()))
        for fac in self.graph.get_factors():
            tps.add(fac.meta['type'])

        final_frame_rvs = final_seed_rvs + final_nonseed_rvs
        frame_lists = [final_seed_rvs, final_nonseed_rvs, final_frame_rvs, final_nounpair_rvs]
        frame_list_names = ['seed frames', 'nonseed frames', 'all frames', 'nounpairs']
        for i in range(len(frame_lists)):
            lst = frame_lists[i]
            name = frame_list_names[i]

            global_cxn_stats = {}
            for tp in tps:
                global_cxn_stats[tp] = []

            for rv in lst:
                cxn_stats = Counter()
                facs = rv.get_factors()
                for fac in facs:
                    cxn_stats[fac.meta['type']] += 1

                for tp in tps:
                    global_cxn_stats[tp].append(cxn_stats[tp])

            self.logger.debug('\t %s:', name)
            for k, v in global_cxn_stats.iteritems():
                avg = np.average(v) if len(v) > 0 else 0
                med = np.median(v) if len(v) > 0 else 0
                self.logger.debug('\t \t %s: avg %0.2f, median %0.2f:',
                    k, avg, med)

    def run(self, init, normalize, maxiters, progress=True):
        """
        Currently just LBP.

        Args:
            init (bool) Whether to initialize lbp
            normalize (bool) Whether to normalize during LBP
            maxiters (int) Maximum number of times to run LBP
            progress (bool) Should underlying lbp show progress

        Returns:
            bool: Did it converge within maxiters
        """
        self.logger.debug('[inference] Running (L)BP...')
        iters, converged = self.graph.lbp(
            init=init,
            normalize=normalize,
            max_iters=maxiters,
            progress=progress)
        self.logger.debug('[inference] (L)BP ran for %d iterations; converged = %r', iters, converged)

        # Just keeping these here because, ya know, they're there.
        # g.print_messages()
        # g.print_rv_marginals(rvs=noun_rvs, normalize=True)

        return converged

    def eval(self, settings, eval_frames, eval_objpairs, filedump):
        """
        Args:
            settings (Settings)
            eval_frames (bool)
            eval_objpairs (bool)
            filedump (bool)

        Returns:
            (int, int)|None, (int, int)|None: 2-tuple, (frame stats, objpair
                stats), either of which can be None if it wasn't evaluated. If
                it was evaluated, a tuple is (#correct, #total).
        """
        frame_res, np_res = None, None
        if eval_frames:
            frame_res = self._eval_frames(settings, filedump)
        if eval_objpairs:
            np_res = self._eval_gold_nounpairs(settings, filedump)

        return frame_res, np_res

    def _eval_gold_nounpairs(self, settings=None, filedump=True):
        """
        How did we do on predicting gold nounpairs?

        Args:
            settings (Settings, default None) settings if we want to track
                results for outputting in table format later. If None, simply
                not used
            filedump (bool, default True) whether to dump actual correct /
                incorrect markers on all decisions made to file for further
                analysis.

        Returns:
            (int, int) n_correct, n_total
        """
        self.logger.debug('[eval] Evaluating nonseed gold nounpair marginals...')
        correct, wrong = [], []
        for rv in self.nonseed_gold_np_rvs:
            want = rv.meta['goldlabel']
            if rv.meta['pruned']:
                got = -1
            else:
                got = np.argmax(self.graph.rv_marginals(rvs=[rv], normalize=True)[0][1])

            # compare
            if got == -1:
                rv.meta['correct'] = False
                wrong.append('%s\t(PRUNED)\n' % (str(rv)))
            elif got == want:
                rv.meta['correct'] = True
                correct.append('%s\t(%s) (%d edges)\n' % (str(rv), RV_LABELS[got], rv.n_edges()))
            else:
                rv.meta['correct'] = False
                wrong.append('%s\t(wanted %s, got %s) (%d edges)\n' % (
                    str(rv),
                    RV_LABELS[want],
                    RV_LABELS[got],
                    rv.n_edges()))
        n_correct, n_wrong = len(correct), len(wrong)
        n_total = n_correct + n_wrong
        acc_str = '%d/%d (%0.2f%%)' % (n_correct, n_total, float(n_correct * 100) / n_total)
        self.logger.debug(acc_str)

        # save to settings for info printing
        if settings is not None:
            settings.add_result('%s gold np acc' % (self.name), acc_str)

        # dump to file
        if filedump:
            self.logger.debug('[eval] Writing nounpair-level results to output/ ...')
            util.ensure_dir('output/')
            with open('output/results-goldnp-%s-correct.latest' % (self.name), 'w') as f:
                f.writelines(correct)
            with open('output/results-goldnp-%s-wrong.latest' % (self.name), 'w') as f:
                f.writelines(wrong)

        return n_correct, n_total

    def _eval_frames(self, settings=None, filedump=True):
        """
        How did we do on predicting frames?

        Args:
            settings (Settings, default None) settings if we want to track
                results for outputting in table format later. If None, simply
                not used
            filedump (bool, default True) whether to dump actual correct /
                incorrect markers on all decisions made to file for further
                analysis.

        Returns:
            (int, int) # correct, # total
        """
        self.logger.debug('[eval] Evaluating frame marginals...')
        all_rvs = self.graph.get_rvs()
        rvs = all_rvs.values()
        nonseed_rvs = [rv for rv in rvs if rv.meta['type'] == 'frame' and not rv.meta['seed']]
        tuples = self.graph.rv_marginals(rvs=nonseed_rvs, normalize=True)
        prediction = {str(rv): marg for rv, marg in tuples}

        correct, wrong = [], []
        n_correct, n_total = 0, 0
        for node_name, pot in self.nonseed_data:
            want_max_idx = np.argmax(pot)
            if node_name in prediction:
                got_max_idx = np.argmax(prediction[node_name])
            else:
                got_max_idx = -1

            if got_max_idx == want_max_idx:
                correct += ['%s\t(%s)\n' % (node_name, RV_LABELS[got_max_idx])]
                n_correct += 1
            else:
                got_label = '[pruned]' if got_max_idx == -1 else RV_LABELS[got_max_idx]
                wrong += ['%s\t(wanted %s, got %s)\n' % (node_name, RV_LABELS[want_max_idx], got_label)]
            n_total += 1
            # save for viz
            # TODO: fixme: handle pruned nodes here
            # all_rvs[node_name].meta['correct'] = got_max_idx == want_max_idx
        acc_str = '%d/%d (%0.2f%%)' % (n_correct, n_total, float(n_correct * 100) / n_total)
        self.logger.debug(acc_str)

        # save to settings for info printing
        if settings is not None:
            settings.add_result('%s acc' % self.name, acc_str)

        # dump to file
        if filedump:
            self.logger.debug('[eval] Writing frame-level results to output/ ...')
            util.ensure_dir('output/')
            with open('output/results-frame-%s-correct.latest' % (self.name), 'w') as f:
                f.writelines(correct)
            with open('output/results-frame-%s-wrong.latest' % (self.name), 'w') as f:
                f.writelines(wrong)

        return n_correct, n_total

    def add_seeds(self, seeds, maxnum=-1, add_rest=False):
        """
        Args:
            seeds ([(str, np.array(float))]):
                "verb_sub[_prep]", [p(x>y), p(x<y), p(x==y)]

            maxnum (int, default -1): Maximum number of seeds to add. -1 means
                no limit.

            add_rest (boolean, default False): Whether to add the rest as RVs
                but w/o unary seed factors

        Returns:
            [RV] rvs added to graph
        """
        # common
        factor_name = 'seed'

        rvs = []
        for i, s in enumerate(seeds):
            if i == maxnum:
                break
            name, pot = s
            rvs += [self.graph.rv(name, 3, RV_LABELS, {'type': 'frame', 'seed': True, 'attr': self.name})]
            self.graph.factor([name], factor_name, pot, {'type': 'seed', 'seedType': 'hard'})

        # possibly add rest as RVs but w/o unary seed factors
        if maxnum != -1 and maxnum < len(seeds) and add_rest:
            for remainder in seeds[maxnum:]:
                name, _ = remainder
                rvs += [self.graph.rv(name, 3, RV_LABELS, {'type': 'frame', 'seed': True, 'remainder': True, 'attr': self.name})]

        return rvs

    def add_seed_facs_to_frames(self, seeds):
        """
        Adds hardcoded factors ONLY to seeds (frames) (RVs must already exist).

        Args:
            seeds ([(str, np.array(float))]):
        """
        for s in seeds:
            name, pot = s
            self.graph.factor([name], 'seed', pot, {'type': 'seed', 'seedType': 'hard'})

    def add_nonseeds(self, nonseeds):
        """
        Args:
            nonseeds ([(str, np.array(float)]) (same as seeds above) NOTE: Ignores
                everything but verb! This might as well be a [str].
        """
        rvs = []
        for ns in nonseeds:
            name, _ = ns
            rvs += [self.graph.rv(name, 3, RV_LABELS, {'type': 'frame', 'seed': False, 'attr': self.name})]
        return rvs

    def add_frames_emb(self, framesplit):
        """
        Adds frames from both "seed" and "nonseed" data with their unary
        embedding factors.

        Args:
            framesplit (int)
        """
        # load up the embedding factor data
        emb = tf.UnaryFrameEmbedding(framesplit)

        rvs = []
        for datum in self.seed_data:
            rvs.append(self._add_frame_emb(emb, datum, True))
        for datum in self.nonseed_data:
            rvs.append(self._add_frame_emb(emb, datum, False))
        return rvs

    def _add_frame_emb(self, emb, datum, seed):
        """
        Helper for add_frames_emb(...)

        Args:
            emb (tf.UnaryFrameEmbedding)
            datum (str, _) framestr, (unused legacy hardcoded potential)
            seed (boolean) Whether this frame should be considered a 'seed'
                frame. There aren't really any 'seed' frames under this
                scenario, but for legacy reasons code expects 'seed' to
                distinguish training from eval nodes.
        """
        framestr, _ = datum
        potential = emb.get(self.name, framestr)
        rv = self.graph.rv(framestr, 3, RV_LABELS, {'type': 'frame', 'seed': seed, 'attr': self.name})
        self.graph.factor([framestr], 'seed', potential, {'type': 'seed', 'seedType': 'emb'})
        return rv

    def add_seed_nounpairs(self, data):
        """
        Adds seed nounpairs (turked gold labels) to attrgraph.

        Args:
            [[str, str, np.ndarray 1x3]]: [[obj1, obj2, potential]]

        Returns:
            [RV]
        """
        rvs = []
        for datum in data:
            obj1, obj2, pot = datum
            name = obj1 + VS + obj2
            rvs.append(self.graph.rv(name, 3, RV_LABELS, {
                'type': 'noun',
                'seed': True,
                'goldlabel': np.argmax(pot),
                'attr': self.name,
            }))
            self.graph.factor([name], 'seed', pot, {'type': 'seed', 'seedType': 'hard'})
        return rvs

    def add_nonseed_nounpairs(self, data):
        """
        Adds nonseed nounpairs (have turked gold labels but not using) to
        attrgraph.

        Args:
            [[str, str, np.ndarray 1x3]]: [[obj1, obj2, potential (unused)]]

        Returns:
            [RV]
        """
        rvs = []
        for datum in data:
            obj1, obj2, pot = datum
            name = obj1 + VS + obj2
            rvs.append(self.graph.rv(name, 3, RV_LABELS, {
                'type': 'noun',
                'seed': False,
                'goldlabel': np.argmax(pot),
                'attr': self.name,
            }))
        return rvs

    def add_seed_facs_to_gold_nps(self, data):
        """
        Adds *ONLNY SEED (HARDCODED) FACTORS* to gold nps in seed_data.

        Args:
            data
        """
        for datum in data:
            obj1, obj2, pot = datum
            name = obj1 + VS + obj2
            self.graph.factor([name], 'seed', pot, {'type': 'seed', 'seedType': 'hard'})

    def add_goldnps_emb(self, seed_data, nonseed_data, objpairsplit):
        """
        Adds nounpairs from both seed and nonseed data with their unary
        embedding factors.

        Args:
            seed_data ([[str, str, np.ndarray 1x3]]): [[obj1, obj2, hardcoded
                potential]]
            nonseed_data ([[str, str, np.ndarray 1x3]]): [[obj1, obj2, hardcoded
                potential]]
            objpairsplit (int)

        Returns:
            ([RV], [RV]) seed RVs, nonseed RVs
        """
        # load up the embedding factor data
        emb = tf.UnaryObjpairEmbedding(objpairsplit)

        seed_rvs, nonseed_rvs = [], []
        for datum in seed_data:
            seed_rvs.append(self._add_goldnp_emb(emb, datum, True))
        for datum in nonseed_data:
            nonseed_rvs.append(self._add_goldnp_emb(emb, datum, False))
        return seed_rvs, nonseed_rvs

    def _add_goldnp_emb(self, emb, datum, seed):
        """
        Helper for add_goldnps_emb(...)

        Args:
            emb (tf.UnaryObjpairEmbedding)
            datum (str, str, np.ndarray 1x3) obj1, obj2, hardcoded potential
                only used for argmax to find the gold label
            seed (boolean) Whether this frame should be considered a 'seed'
                frame. There aren't really any 'seed' frames under this
                scenario, but for legacy reasons code expects 'seed' to
                distinguish training from eval nodes.
        """
        obj1, obj2, hardcoded_pot = datum
        potential = emb.get(self.name, obj1, obj2)
        objpairstr = obj1 + VS + obj2
        rv = self.graph.rv(objpairstr, 3, RV_LABELS, {
            'type': 'noun',
            'seed': seed,
            'goldlabel': np.argmax(hardcoded_pot),
            'attr': self.name,
        })
        self.graph.factor([objpairstr], 'seed', potential, {'type': 'seed', 'seedType': 'emb'})
        return rv

    def get_ng_nouns(self, verb_rvs, num, filter_abstract, lemmatize):
        """
        Finds nouns from ngramdb using most frequently found nouns with given
        verbs.

        Args:
            verb_rvs ([RV]) These are really frames
            num (int) maximum number of nouns to add per frame
            filter_abstract (bool): Whether to filter abstract nouns out of the
                nouns added.
            lemmatize (bool): Whether to compress nouns into their lemmatized
                form before returning.

        Returns:
            [str] nouns
            [RV] noun RVs added
        """
        nouns = []
        mc = Counter()  # master counter (for noun output dumping)
        verb_names = [str(rv) for rv in verb_rvs]
        for vn in verb_names:
            v, s, p = td.TurkedData.str_to_vsp(vn)
            c = self.ngramdb.get_top_nouns(v, s, p, filter_abstract, lemmatize)
            nouns += [k for k, _ in c.most_common(num)]
            for k, count in c.most_common(num):
                mc[k] += count

        # Write for inspection.
        dumpfile = 'output/noun-freqs.txt'
        self.logger.debug('[build] Dumping noun frequencies to "%s"', dumpfile)
        util.ensure_dir('output/')
        with open(dumpfile, 'w') as f:
            f.write('Found %d nouns:\n' % (len(mc)))
            for n, count in mc.most_common():
                f.write('%s\t%d\n' % (n, count))

        return nouns

    def get_ext_nouns(self, filename):
        """
        Gets nouns from a file (one per line).

        Args:
            filename (str)

        Returns:
            [str] nouns
        """
        with open(filename, 'r') as f:
            nouns = [l.strip() for l in f.readlines()]

        return nouns

    def add_nouns(self, nouns, check_existing=False):
        """
        Adds a set of nouns to this attr graph, replacing all 'person' nouns
        with a generic one.

        Args:
            nouns ([str])
            check_existing (bool, default False)

        Returns:
            [str] nouns
            [RV] noun RVs added
        """
        # Need unique nouns. Also removing from set (below) is O(1) instead of
        # O(n).
        ns = set(nouns)

        # Remove all person-referring nouns (we'll add the global one at the
        # end).
        # person_found = False
        for pn in ng.PERSON_NOUNS:
            if pn in ns:
                # person_found = True
                ns.remove(pn)

        # Turn to a list and add our human noun at the front. (Ordering doesn't
        # matter but this is convenient.)
        ns = list(ns)
        # if person_found:
        ns = [ng.HUMAN_NOUN] + ns

        # add all pairs
        rvs = []
        for i in range(len(ns)):
            for j in range(i, len(ns)):
                # given we're adding nouns in different places, maybe check to not
                # double-add
                if check_existing:
                    if self.graph.has_rv(ns[i] + VS + ns[j]) or self.graph.has_rv(ns[j] + VS + ns[i]):
                        continue
                # legit add it here.
                n = ns[i] + VS + ns[j]
                rvs += [self.graph.rv(n, 3, RV_LABELS, {'type': 'noun', 'seed': False, 'attr': self.name})]

        return ns, rvs

    def add_fac_verb_sim(self, verb_rvs, sim_thresh, verb_sim_pot):
        """
        Args:
            verb_rvs ([RV])
            sim_thresh (float) in [0.0, 1.0], lower bound of how similar verbs
                have to be in order to have a factor added between them.
            verb_sim_pot (np.ndarray of shape 3x3)
        """
        # divide into sets based on ending
        end_sets = {}
        for rv in verb_rvs:
            verb, sub = str(rv).split('_')[:2]
            if sub not in end_sets:
                end_sets[sub] = []
            end_sets[sub] += [rv]

        sim_facs = []
        for sub, rvs in end_sets.iteritems():
            # consider each RV vs others
            for i in range(len(rvs)):
                this_rv = rvs[i]
                this_name = str(this_rv).split('_')[0]
                rest = rvs[i+1:]
                rest_names = [str(r).split('_')[0] for r in rest]
                sims = self.glove.distance(this_name, rest_names)
                for idx, sim in enumerate(sims):
                    other_idx = i + 1 + idx
                    other_rv = rvs[other_idx]
                    other_name = str(other_rv).split('_')[0]

                    # exclude RVs with the same verb---those should be handled
                    # by inference factors (when / if we add them)
                    if this_name == other_name:
                        continue

                    # debug printing
                    # print 'sim of %s vs %s: %f' % (this_name, other_name, sim)

                    if sim > sim_thresh:

                        # debug printing
                        # print 'adding similarity between verb RVs %s and %s' % (
                        #     this_rv, other_rv)

                        sim_facs += [self.graph.factor(
                            [this_rv, other_rv],
                            'verb_sim',
                            verb_sim_pot,
                            {'type': 'verb_sim'})]
        return sim_facs

    def add_fac_noun_sim(self, nouns, sim_thresh, noun_eq_pot, noun_sim_pot,
            noun_sim_rev_pot):
        """
        Args:
            nouns ([str])
            sim_thresh (float) in [0.0, 1.0], lower bound of how similar nouns
                have to be in order to have a factor added between them.
            noun_eq_pot (np.ndarray of 1x3) unary potential for a_vs_b when
                a == b
            noun_sim_pot (np.ndarray of 3x3) binary potential for a_vs_c and
                b_vs_c when a == b
            noun_sim_rev_pot (np.ndarray of 3x3) binary potential for a_vs_c and
                c_vs_b when a == b

        Returns:
            [Factor]
        """
        # Note: naive pairs^2 = n^4 implementation is prohibitively slow (early
        # systems have ~280 nouns = ~39,000 pairs, so pairs^2 takes a loooong
        # time). Finding similar pairs first and then doing matching is order
        # matches*n + matches^2.

        # First, find pairs of similar nouns.
        sims = []
        for i in range(len(nouns)):
            # Get all remaining nouns that are similar to it.
            n = nouns[i]
            ms = nouns[i+1:]
            res = self.glove.distance(n, ms)
            for j, r in enumerate(res):
                if r > sim_thresh:
                    sims += [(n, ms[j])]

                    # Debug printing
                    # print 'Found similarity between nouns %s and %s (%f)' % (n, ms[j], r)

        # Debug printing
        self.logger.debug('[build] \t Found %d pairs of similar nouns', len(sims))

        # Add links between corresponding RVs. First, consider links that can be
        # added only by individual pairs.
        sim_facs = []

        # Add s[0] = s[1] unary factors. This should add sims new factors.
        for i, s in enumerate(sims):
            a, b = s[0], s[1]
            # One of the two combinations should work
            if self.graph.has_rv(a + VS + b):
                name = a + VS + b
            elif self.graph.has_rv(b + VS + a):
                name = b + VS + a
            else:
                # Now not adding all n^2 nouns in so some might not be in the graph.
                # self.logger.error('PROGRAMMER ERROR: object pair (%s, %s) not found in graph' % (a, b))
                # self.logger.error('... jumping into console to debug ...')
                # code.interact(local=dict(globals(), **locals()))
                # assert False
                continue
            sim_facs += [self.graph.factor(
                [name], 'noun_eq', noun_eq_pot, {'type': 'noun_sim'})]

        # Add some binary factors. This should add sims * (nouns-2) new factors.
        for i, s in enumerate(sims):
            # Find RVs where we have matches. With A = s[0] and B = s[1], for every
            # C, try links for the following (and exactly one should work):
            # - A_vs_C <-[+]-> B_vs_C
            # - C_vs_A <-[+]-> C_vs_B
            # - A_vs_C <-[-]-> C_vs_B
            # - C_vs_A <-[-]-> B_vs_C
            a, b = s[0], s[1]
            for c in nouns:
                # Only considering different nouns.
                if c == a or c == b:
                    continue

                # Try combinations
                # These two are with A and B facing C in the same way, so should be
                # linked in the same direction.
                if self.graph.has_rv(a + VS + c) and self.graph.has_rv(b + VS + c):
                    sim_facs += [self.graph.factor(
                        [a + VS + c, b + VS + c],
                        'noun_sim',
                        noun_sim_pot,
                        {'type': 'noun_sim'})]
                elif self.graph.has_rv(c + VS + a) and self.graph.has_rv(c + VS + b):
                    sim_facs += [self.graph.factor(
                        [c + VS + a, c + VS + b],
                        'noun_sim',
                        noun_sim_pot,
                        {'type': 'noun_sim'})]
                # These two are with A and B facing C in the opposite way, so
                # should be linked in the opposite direction.
                elif self.graph.has_rv(a + VS + c) and self.graph.has_rv(c + VS + b):
                    sim_facs += [self.graph.factor(
                        [a + VS + c, c + VS + b],
                        'noun_sim',
                        noun_sim_rev_pot,
                        {'type': 'noun_sim'})]
                elif self.graph.has_rv(c + VS + a) and self.graph.has_rv(b + VS + c):
                    sim_facs += [self.graph.factor(
                        [c + VS + a, b + VS + c],
                        'noun_sim',
                        noun_sim_rev_pot,
                        {'type': 'noun_sim'})]
                else:
                    # Now *not* adding all n^2 nouns in, so some might not be
                    # in the graph.
                    # assert False, 'PROGRAMMER ERROR: NOUN SIMILARITY IS CONFUSED'
                    continue

        return sim_facs

    def add_fac_inf_simframe(self, frame_rvs, pot):
        """
        Adds inference factors to the graph between frames of the same verb that
        should behave similarly (p/op, d/od).

        Args:
            frame_rvs ([RV])
            pot (np.ndarray of shape 3x3)

        Returns:
            [Factor] which factors were added
        """
        matching_pairs = [
            ('_p', '_op'),
            ('_d', '_od'),
        ]

        # group frames by verb
        verbs = {}
        for rv in frame_rvs:
            verb, sub, _ = td.TurkedData.str_to_vsp(str(rv))
            if verb not in verbs:
                verbs[verb] = {}
            verbs[verb][sub] = rv

        # code.interact(local=dict(globals(), **locals()))

        # find cases where subs match and add factors between then
        simframe_facs = []
        for verb, subs in verbs.iteritems():
            for pair in matching_pairs:
                sub1, sub2 = pair
                if sub1 in subs and sub2 in subs:
                    simframe_facs.append(self.graph.factor(
                        [subs[sub1], subs[sub2]],
                        'withinverb_simframe',
                        pot,
                        {'type': 'withinverb_simframe'}))
        return simframe_facs

    def add_fac_inf_prep(self):
        """
        Adds preposition inference factors to graph. Each factor encourages a
        single preposition to behave the same way across different frames.
        """
        pass

    def get_sel_pref_nouns_freq(self, frame_rv, cutoff, _):
        """
        Uses raw frequency; enforces >= cutoff.

        Args:
            verb_rvs ([RV])
            cutoff (int) Frequency below which sel pref factors aren't added
            _ (unused)

        Returns:

            ([str|(str, str)], [bool]) List of either nouns or noun tuples
                (depending on the frame), and list of whether or not each was
                found reversed in the factor graph.
        """
        v, s, p = td.TurkedData.str_to_vsp(str(frame_rv))
        n_els = self.ngramdb.get_freq_nouns(v, s, p, cutoff)
        # NOTE: the following is broken: when the order of the noun pair is
        #       reversed in the graph, need to also reverse the potential. PMI
        #       sidesteps this by querying PMI after the reversal.
        assert False, 'Cannot use sel pref w/ freq until fixing it'
        n_names = self.noun_els_to_rv_names(n_els)
        return n_names, rev_list

    def get_sel_pref_nouns_pmi(self, frame_rv, cutoff, min_freq):
        """
        Uses PMI; enforces >= cutoff.

        Args:
            frame_rv (RV)
            cutoff (float)
            min_freq (int) minimum frequency for nouns even to be checked by PMI
                (hopefully to cut off super-rare occurrences)

        Returns:
            ([str], [str]) forward (standard pot) names of RVs, and backward
                (reversed pot) names of RVs
        """
        frame_str = str(frame_rv)
        v, s, p = td.TurkedData.str_to_vsp(frame_str)
        n_els = self.ngramdb.get_freq_nouns(v, s, p, min_freq)
        n_names = self.noun_els_to_rv_names(n_els)
        n_tuples = [tuple(n.split(VS)) for n in n_names]

        # debugging
        # print 'in get_sel_pref_nouns_pmi(...) ...'
        # code.interact(local=dict(globals(), **locals()))

        # debugging
        # self.logger.debug('[build] frame "%s" using cutoff: %0.2f', frame_str, cutoff)
        # self.logger.debug('[build] frame "%s" had %d noun tuples:', frame_str, len(n_tuples))
        # for n in n_tuples:
        #     self.logger.debug('[build] \t pmi of "%s" and "%s":  %0.2f', frame_str, n, self.pmi.query(frame_str, n))

        # filter by those with pmi >= cutoff, adding depending on which (or
        # both) directions match pmi. note that in either case, we add the
        # standard (n_tuple) node name, as that's the one that's actually in the
        # graph.
        forward_tuples, backward_tuples = [], []
        for n_tuple in n_tuples:
            score = self.pmi.query(frame_str, n_tuple)
            if score >= cutoff:
                forward_tuples.append(n_tuple)
            rev_tuple = (n_tuple[1], n_tuple[0])
            score_rev = self.pmi.query(frame_str, rev_tuple)
            if score_rev >= cutoff:
                backward_tuples.append(n_tuple)
                # curiosity printing
                # print frame_str, n_tuple, score

        # orig
        # forward_tuples = [n for n in n_tuples if self.pmi.query(frame_str, n) >= cutoff]
        # self.logger.debug('[build] \t result: %d passed cutoff', len(forward_tuples))
        forward_strs = [a + VS + b for a,b in forward_tuples]
        backward_strs = [a + VS + b for a,b in backward_tuples]
        return forward_strs, backward_strs

    def noun_els_to_rv_names(self, nels):
        """
        For selection preference factors: Takes a list of noun elements (str or
        (str, str)) and returns the relevant RV names. Filters out ones that
        don't exist in graph.

        Args:

            nels ([str|(str, str)])

        Returns:
            [str] The list of RV names: could have reversed components in nels.

            (previously also returned [bool], the list of whether the
                order is reversed (in which case the potential should be a
                reversed one).
        """
        rv_names = []
        # rev_list = []
        for n in nels:
            # All will be the same, but putting code here because it's
            # easier.
            if type(n) is tuple:
                a, b = n
            else:
                a, b = ng.HUMAN_NOUN, n

            # Make sure to skip HUMAN_NOUN, HUMAN_NOUN combos (as this is
            # trivially =)
            if a == ng.HUMAN_NOUN and b == ng.HUMAN_NOUN:
                # Debug printing.
                # print 'Skipping (%s, %s) as this should not be in graph' % (a, b)
                continue

            # We could have come up with stuff not in our graph. Ignore it
            # if so.
            if self.graph.has_rv(a + VS + b):
                name = a + VS + b
                # rev = False
            elif self.graph.has_rv(b + VS + a):
                name = b + VS + a
                # rev = True
            else:
                continue

            rv_names.append(name)
            # rev_list.append(rev)

        return rv_names#, rev_list

    def add_fac_sel_pref(self, frame_rvs, method, cutoff, sel_pref_pot,
            min_freq, use_emb, emb_filename):
        """
        Args:
            frame_rvs ([RV])

            method (str): Settings.SEL_PREF_PMI or Settings.SEL_PREF_FREQ

            cutoff (int|float) Cutoff for desired sel pref method

            sel_pref_pot (np.ndarray of 3x3) Potential for selectional
                preference factors.

            min_freq (int) For PMI only: minimum frequency for nouns to be
                considered for PMI. Hopefully cuts off super rare nouns.

            use_emb (bool): Whether to use (embedding-)trained potentials. If
                set, ignores sel_pref_pot.

            emb_filename (str); Embedding-trained filename to load from; only
                used if use_emb.

        Returns:
            [Factor]
        """

        # Possible extension: scale the strength of the factors based on the
        # evidence.
        #
        # Current approach: don't scale potential strength.
        if method == Settings.SEL_PREF_PMI:
            sel_func = self.get_sel_pref_nouns_pmi
        elif method == Settings.SEL_PREF_FREQ:
            sel_func = self.get_sel_pref_nouns_freq
        else:
            assert False, 'Unknown sel pref method: "%s"' % (method)

        # setup reversed potential & hardcoded func to access both
        sel_pref_pot_rev = sel_pref_pot.copy()
        sel_pref_pot_rev[:,[0,1]] = sel_pref_pot[:,[1,0]]
        def pot_preset(_1, _2, forward):
            if forward:
                return sel_pref_pot
            return sel_pref_pot_rev

        # pick which one to use based on provided setting (and only load emb
        # file if needed)
        pot_func = pot_preset
        if use_emb:
            # setup trained func
            emb_getter = tf.SelPrefEmbedding(emb_filename)
            def pot_emb(frame_rv, np_name, forward):
                obj1, obj2 = np_name.split(VS)
                if not forward:
                    obj1, obj2 = obj2, obj1
                return emb_getter.get(self.name, str(frame_rv), obj1, obj2)
            pot_func = pot_emb

        sel_pref_facs = []
        for frame_rv in frame_rvs:
            # note that both forward and backward have the correct (i.e., RV
            # with that name is in the graph) name; backward ones just need the
            # reversed potential.
            forward_np_names, backward_np_names = sel_func(frame_rv, cutoff, min_freq)

            # self.logger.debug('[build] frame "%s" got %d nps to add', frame_rv, len(np_names))
            for name in forward_np_names:
                sel_pref_facs += [self.graph.factor(
                        [frame_rv, name],
                        'sel_pref',
                        pot_func(frame_rv, name, True),
                        {'type': 'sel_pref'})]
            for name in backward_np_names:
                sel_pref_facs += [self.graph.factor(
                        [frame_rv, name],
                        'sel_pref',
                        pot_func(frame_rv, name, False),
                        {'type': 'sel_pref', 'subtype': 'rev'})]
        return sel_pref_facs

    def save_marginals(self):
        # Save the decision about each note in its metadata
        self.logger.debug('Saving all marginals to RVs...')
        for rv, marg in self.graph.rv_marginals(normalize=True):
            rv.meta['marginals'] = marg

    def viz(self, dirname):
        """
        Exhaustively writes to files. Visualizes frame RVs and the things
        connected to it.
        """
        self.logger.debug('Dumping RVs for ' + self.name)
        util.ensure_dir(dirname)
        for rv in self.graph.get_rvs().values():
            fn = self.name + '-' + str(rv) + '.json'
            with open(os.path.join(dirname, fn), 'w') as f:
                f.write(self._viz_json(rv) + '\n')

    def _viz_json(self, rv):
        """
        Args:
            rv (RV)

        Returns str (json representation)
        """
        rvs = [rv]
        facs = []
        nodes = [AttrGraph._rv_to_dict(rv, True)]
        links = []
        rv_mappings = {rv: []}
        # loop over factors
        for i, fac in enumerate(rv.get_factors()):
            # factor gets node (square)
            nodes.append(AttrGraph._fac_to_dict(fac, i))
            facs.append(fac)
            # and for each thing it connects, an edge and that rv (if rv not already added)
            for cur_rv in fac.get_rvs():
                links.append(AttrGraph._cxn_to_dict(fac, i, cur_rv))
                # bookkeeping ew
                rv_mappings[rv].append(cur_rv)
                if cur_rv not in rv_mappings:
                    rv_mappings[cur_rv] = []
                rv_mappings[cur_rv].append(rv)
                # add the rv if needed
                if cur_rv not in rvs:
                    nodes.append(AttrGraph._rv_to_dict(cur_rv))
                    rvs.append(cur_rv)

        # loop over rvs to find missing connections in the set. only add those
        # connections in the factor type whitelist (knob to turn over how insane the
        # graph will look)
        fac_type_whitelist = ['noun_sim']
        j = i  # new var for clarity; just keep counting up
        for cur_rv in rvs:
            for fac in cur_rv.get_factors():
                if fac.meta['type'] not in fac_type_whitelist:
                    continue
                for other_rv in fac.get_rvs():
                    if other_rv in rvs and \
                            other_rv not in rv_mappings[cur_rv] and \
                            cur_rv not in rv_mappings[other_rv] and \
                            cur_rv is not other_rv:
                        nodes.append(AttrGraph._fac_to_dict(fac, j))
                        facs.append(fac)
                        links.append(AttrGraph._cxn_to_dict(fac, j, cur_rv))
                        links.append(AttrGraph._cxn_to_dict(fac, j, other_rv))
                        rv_mappings[cur_rv].append(other_rv)
                        j += 1


        rv_c = Counter([rv_cur.meta['type'] for rv_cur in rvs])
        fac_c = Counter([f.meta['type'] for f in facs])
        stats = {
            'focus': str(rv),
            'n_rvs': sum(rv_c.values()),
            'rvs': dict(rv_c),
            'n_facs': sum(fac_c.values()),
            'facs': dict(fac_c),
        }
        if 'correct' in rv.meta:
            stats['correct'] = bool(rv.meta['correct'])

        return json.dumps({
            'nodes': nodes,
            'links': links,
            'stats': stats,
        })

    @staticmethod
    def _rv_to_dict(rv, focus=False):
        """
        Args:
            rv (RV)
            focus (bool, default=False)

        Retruns:
            dict: for future JSON-ing
        """
        res = {
            'type': 'rv',
            'subtype': rv.meta['type'],
            'id': '[%s] %s' % (rv.meta['attr'], rv.name),
            'attr': rv.meta['attr'],
            'rvname': rv.name,
        }

        if 'marginals' not in rv.meta:
            print 'WARNING: rv %s did not have meta "marginals". Meta: %r' % (rv, rv.meta)
        else:
            res['weights'] = list(rv.meta['marginals'])

        if focus:
            res['focus'] = True
        return res

    @staticmethod
    def _cxn_to_dict(fac, i, rv):
        """
        Args:
            fac (Factor)
            i (int) local uid for fac
            rv (RV)

        Returns:
            dict: for future JSON-ing
        """
        return {
            'type': fac.meta['type'],
            'source': '%s%d' % (fac.name, i),
            'target': '[%s] %s' % (rv.meta['attr'], rv.name),
            'weights': list(fac.get_outgoing_for(rv)),
        }

    @staticmethod
    def _fac_to_dict(fac, i):
        """
        Args:
            fac (Factor)
            i (int) local uid for fac
            target (bool, default=False)

        Retruns:
            dict: for future JSON-ing
        """
        res = {
            'type': 'fac',
            'subtype': fac.meta['type'],
            'id': '%s%d' % (fac.name, i),
        }
        if 'subtype' in fac.meta:
            res['specific'] = fac.meta['subtype']
        if 'seedType' in fac.meta:
            res['specific'] = fac.meta['seedType']
        if fac.n_edges() == 1:
            res['weights'] = list(fac.get_potential())
        return res
