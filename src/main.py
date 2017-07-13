"""
Entry point to verbphysics system.

author: mbforbes
"""

# IMPORTS
# -----------------------------------------------------------------------------

# Logging first this was a fun bug.
import logging
import util
util.ensure_dir('log/')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S',
    filename='log/latest.log',
    filemode='w')
base_logger = logging.getLogger(__name__)

# builtins
import argparse
import sys
import time

# 3rd party
import factorgraph as fg

# local
import attrgraph
import data as ng
# If I really have to do this then it's a gross oversight of python's.
from data import SizeQueryN
import glove
import data_turked as td
from settings import Settings


# GLOBALS (SORRY)
# -----------------------------------------------------------------------------

CONSOLE_LOG_LEVEL = logging.DEBUG
VIZ_OUTPUT_DIR = 'viz/'
FRAMES_FILENAME = 'data/verbphysics/action-frames/action-frames.csv'
FRAMES_TRAIN_5_DIR = 'data/verbphysics/action-frames/train-5/'
FRAMES_TRAIN_20_DIR = 'data/verbphysics/action-frames/train-20/'

# Setting configurations follow.

playing = {
    # Set your desired config here. Default values are defined in settings.py
    # in Settings._get_default_map().
}

# Archival configurations.

model_a = {
    Settings.Eval: [Settings.EVAL_DEV, Settings.EVAL_TEST],
    Settings.GloveVerbSimThresh: [0.4],
    Settings.GloveNounSimThresh: [0.4],
    Settings.SelPrefPMICutoff: [5.0],
    Settings.IncludeSelPrefFactors: [True],
    Settings.IncludeXgraph: [False],
    Settings.IncludeVerbSimFactors: [True],
    Settings.IncludeNounSimFactors: [True],
    Settings.IncludeInfWithinverbSimframeFactors: [False],
    Settings.ObjpairSplit: [5],
    Settings.FrameSplit: [5],
}

model_b_frames = {
    Settings.Eval: [Settings.EVAL_DEV, Settings.EVAL_TEST],
    Settings.GloveVerbSimThresh: [0.4],
    Settings.GloveNounSimThresh: [0.4],
    Settings.SelPrefPMICutoff: [4.0],
    Settings.IncludeSelPrefFactors: [True],
    Settings.IncludeXgraph: [False],
    Settings.IncludeVerbSimFactors: [False],
    Settings.IncludeNounSimFactors: [True],
    Settings.IncludeInfWithinverbSimframeFactors: [True],
    Settings.ObjpairSplit: [20],
    Settings.FrameSplit: [5],
}

model_b_objpairs = {
    Settings.Eval: [Settings.EVAL_DEV, Settings.EVAL_TEST],
    Settings.GloveVerbSimThresh: [0.5],
    Settings.GloveNounSimThresh: [0.45],
    Settings.SelPrefPMICutoff: [4.0],
    Settings.IncludeSelPrefFactors: [True],
    Settings.IncludeXgraph: [True],
    Settings.IncludeVerbSimFactors: [True],
    Settings.IncludeNounSimFactors: [True],
    Settings.IncludeInfWithinverbSimframeFactors: [True],
    Settings.ObjpairSplit: [5],
    Settings.FrameSplit: [20],
}


# FUNCTIONS
# -----------------------------------------------------------------------------

def _setup_logging(backup=False):
    util.ensure_dir('log/')

    # Also log to backup file with date.
    if backup:
        fh = logging.FileHandler('log/' + time.strftime('%y-%m-%d_%H-%M-%S') +
            '.log')
        fh.setLevel(logging.DEBUG)
        f_formatter = logging.Formatter(
            fmt='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(f_formatter)
        logging.getLogger('').addHandler(fh)

    # Also log to console.
    console = logging.StreamHandler()
    console.setLevel(CONSOLE_LOG_LEVEL)
    c_formatter = logging.Formatter(
        fmt='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(c_formatter)
    logging.getLogger('').addHandler(console)


def _build_xgraph(graphs, tuples, pot):
    """
    Makes the interconnected (across knowledge dimension) graph.

    Args:
        graphs ([AttrGraph])
        tuples ([(str, str)]) Attr pairs to add cxns between frame RVs
        pot (np.ndarray of shape 3x3)

    Returns
        fg.Graph: the xgraph
    """
    xgraph = fg.Graph(debug=False)
    base_logger.debug('Adding xgraph xfactors...')
    total = 0
    for t in tuples:
        attr1, attr2 = t
        g1 = [g for g in graphs if g.name == attr1]
        g2 = [g for g in graphs if g.name == attr2]

        # Might not have one or both of the graphs because of the current
        # settings.
        if len(g1) != 1 or len(g2) != 1:
            base_logger.debug(
                '\t skipping links between missing graphs %s and %s', attr1,
                attr2)
            continue

        g1 = g1[0]
        g2 = g2[0]

        # Find RVs that match across both graphs. Pruned RVs won't be returned
        # by get_rvs() (as they are actually deleted from the graph's underlying
        # dict), but we do want to make sure we're only linking frames.
        matches = []
        for rv_name, rv1 in g1.graph.get_rvs().iteritems():
            if rv1.meta['type'] != 'frame':
                continue
            if g2.graph.has_rv(rv_name):
                rv2 = g2.graph.get_rvs()[rv_name]
                matches.append([rv1, rv2])

        # add factors to our linking graph
        for match in matches:
            xgraph.factor(match, 'xfactor', pot, {'type': 'xfactor'})

        # reporting
        base_logger.debug(
            '\t added %d links between frame RVs between %s and %s',
            len(matches), attr1, attr2)
        total += len(matches)
    base_logger.debug('Added %d xgraph xfactors in total' % (total))
    return xgraph


def _overall_stats(label, tuples):
    """
    Computes overall accuracy; returns in 'Settings'-friendly format.

    Args: tuples([(int, int)]) Each entry is (# correct, # total) label (str)
        What to call this

    Returns: (str, str) key, val of settings column to add
    """
    n_correct = sum(tp[0] for tp in tuples)
    n_total = sum(tp[1] for tp in tuples)
    return 'OVERALL %s acc' % (label), '%d/%d (%0.2f%%)' % (
        n_correct, n_total, (n_correct*100.0)/n_total)


def main(config, product, viz):
    """
    Runs the verbphysics system using combinations of configurations specified
    by config.

    Args:
        config (dict): The configuration dictionary to use. Keys should be
            Settings.XXX string constants; vals should be lists of values to
            try.

        product (bool): Whether to try all (polynomially many)
            combinations of settings specified in config (True), or whether to
            try varying along each config setting individually (linearly many)
            (False).

        viz (bool): Whether to dump visualization data of the built model.
    """
    # load up stuff needed
    base_logger.debug('Loading ngramdb cached data...')
    d = ng.Data()
    base_logger.debug('Loading PMI...')
    pmi = ng.PMI()
    base_logger.debug('Loading GloVe...')
    glv = glove.Glove()

    # Init settings.
    settings = Settings()
    if product:
        settings.trial_product(config)
    else:
        settings.trial_sequence(config)

    # Keep cycling through experiments.
    base_logger.debug('Beginning experiments...')
    while(settings.next()):
        # Load data and init graphs
        base_logger.debug('Loading turked data...')
        verb_data = td.TurkedData.load(
            FRAMES_FILENAME,
            settings.get(Settings.AgreementNeeded),
            settings.get(Settings.GTBiggerPot),
            settings.get(Settings.GTSmallerPot),
            settings.get(Settings.GTEqPot))
        eval_mode = settings.get(Settings.Eval)
        frame_split = settings.get(Settings.FrameSplit)
        if frame_split == 5:
            framesplitdir = FRAMES_TRAIN_5_DIR
        elif frame_split == 20:
            framesplitdir = FRAMES_TRAIN_20_DIR
        else:
            base_logger.error('Unknown frame split: %r', frame_split)
            sys.exit(1)
        graphs = [attrgraph.AttrGraph(glv, d, pmi, verb_data, a, eval_mode,
            framesplitdir) for a in settings.get(Settings.Attrs)]

        # Build attr graphs
        for g in graphs:
            g.build(settings)

        # Run LBP
        normalize = settings.get(Settings.NormalizeLBP)
        maxiters = settings.get(Settings.LBPMaxIters)
        if not settings.get(Settings.IncludeXgraph):
            # no connections between graphs; run each independently
            for g in graphs:
                g.run(True, normalize, maxiters, True)
        else:
            # build special graph that has connections between graphs
            xgraph = _build_xgraph(graphs, settings.get(Settings.XgraphTuples),
                settings.get(Settings.XgraphPot))

            # init all the graphs
            xgraph.init_messages()
            for g in graphs:
                g.graph.init_messages()

            # Run LBP piecewise across all graphs (including xgraph)
            for i in range(1, maxiters + 1):
                base_logger.debug('Running LBP iter %d on all graphs...', i)
                convg = True

                # run for the attr graphs
                for g in graphs:
                    convg &= g.run(False, normalize, 1, False)

                # run for the xgraph
                xconvg, _ = xgraph.lbp(False, normalize, 1, False)
                convg &= xconvg

                # check convergence
                if convg:
                    base_logger.debug('All graphs converged! Stopping LBP.')
                    break

        # Decide what to eval (5 splits only)
        objpair_split = settings.get(Settings.ObjpairSplit)
        eval_frames = frame_split == 5
        eval_objpairs = objpair_split == 5

        # Eval and pre-viz
        verb_res_list, np_res_list = [], []
        for g in graphs:
            verb_res, np_res = g.eval(settings, eval_frames, eval_objpairs,
                True)
            verb_res_list.append(verb_res)
            np_res_list.append(np_res)
            g.save_marginals()

        # Compute & save overall statistics
        if eval_frames:
            settings.add_result(*_overall_stats('frame', verb_res_list))
        if eval_objpairs:
            settings.add_result(*_overall_stats('np', np_res_list))

        # Viz
        if viz:
            for g in graphs:
                g.viz(VIZ_OUTPUT_DIR)

    settings.log_results()


if __name__ == '__main__':
    # Logic we don't want to worry about throughout
    _setup_logging(backup=True)

    # these are the possible configs to choose from
    config_options = {
        'playing': playing,
        'model_a': model_a,
        'model_b_frames': model_b_frames,
        'model_b_objpairs': model_b_objpairs,
    }
    config_opt_str = ' | '.join(config_options.keys())

    # cmd line
    parser = argparse.ArgumentParser(
        description='verbphysics reference implementation')
    parser.add_argument(
        '--config', metavar='CONFIG', default='model_a',
        help='hyperparameter configuration to use; options: ' +
        config_opt_str + ' (default: model_a')
    parser.add_argument(
        '--poly', type=bool, default=True, help='Whether to try '
        'polynomially-many hyperparameter config combinations (True, default) '
        'or vary config dimension sequentially (False). '
    )
    parser.add_argument(
        '--viz', action='store_true', help='Whether to dump model / data to '
        'JSON for visualization (default False).'
    )
    args = parser.parse_args()

    # checking
    if args.config not in config_options:
        print 'Error: "%s" unknown config. Options are %s' % (args.config,
            config_opt_str)
        sys.exit(1)

    main(config_options[args.config], args.poly, args.viz)
