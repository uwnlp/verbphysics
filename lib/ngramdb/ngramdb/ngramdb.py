import time
import uuid

from myria import MyriaConnection

import util
from constants import *
from ngramtoken import ngrams_from_tupledict


class NgramDb(object):

    def __init__(self, connection_id):
        self._connection = MyriaConnection(
                hostname=REST_URL,
                port=REST_PORT,
                ssl=True)

        connection_id = connection_id.replace(' ', '_')

        if not connection_id.replace('_', '').isalpha():
            raise ValueError("connection_id must be letters only, no "
                             "numbers or punctuation")

        self._connection_id = connection_id

        self.queries = []

    def create_query(
            self,
            words=None,
            postags=None,
            deprels=None,
            headids=None,
            ignore_position=False,
            absolute_position=False,
            limit=None,
            threshold=None,
            description=None,
            output=None):
        """Creates an NgramQuery object, which can be passed to the
        `run_query` method of this NgramDb object.

        Keyword arguments:
            words
                - a list of strings
            postags
                - a list of Penn-treebank style POS tags
            deprels
                - a list of Stanford-style dependency relations
            headids
                - a list of integers corresponding to the list position of
                  this token's head
            ignore_position
                - do not pay attention to the ordering of the tokens in
                  the lists
            absolute_position
                - the positions of tokens must match the positions in the
                  ngram exactly
            limit
                - TODO: NOT COMPLETELY IMPLEMENTED
            threshold
                - only return ngrams with at least this frequency
            description
                - a plain-language description of this query
            output
                - name of the Myria table that will store this query's
                  results; default is this NgramDb's connection_id

        Except for words, all arguments are optional.

        Any position in words, postags, deprels, or headids can be defined as
        `None` to denote that space as a "wildcard". For example,
        postags=["NNS",

        Any string in words, postags, or deprels may use the "|" character to
        signify "or". For example, words=["cat|dog|mous", "eats|runs"] will
        match "cat", "mouse", or "dog" in the first token, and "eats" or "runs"
        in the second token. """
        if not description or not isinstance(description, str):
            description = "[ ngramdb query #{} from {} ]".format(
                len(self.queries), self._connection_id)

        if not output or not isinstance(output, str):
            output = self._connection_id

        return NgramDbQuery(
                words=words,
                postags=postags,
                deprels=deprels,
                headids=headids,
                ignore_position=ignore_position,
                absolute_position=absolute_position,
                limit=limit,
                threshold=threshold,
                description=description,
                output=output)

    def run_query(self, query):
        """Runs an NgramQuery and returns a list of Ngrams.

        See create_query for details on creating a query.
        """
        self.queries.append(query)

        q_plan = self._make_join_context_query_plan(query)

        relation_key = q_plan['fragments'][-1]['operators'][-1]['relationKey']

        try:
            # answer = self._connection.execute_query(q_plan)
            myria_query = self._connection.submit_query(q_plan)
            query_id = myria_query['queryId']

            full_status = self._connection.get_query_status(query_id)
            status = full_status['status']

            while status not in ('UNKNOWN', 'SUCCESS', 'ERROR'):
                time.sleep(0.1)
                full_status = self._connection.get_query_status(query_id)
                status = full_status['status']

            if status in ('UNKNOWN', 'ERROR'):
                raise RuntimeError(
                        "Myria error: {}".format(full_status['message']))

            else:
                raw_results = self._connection.download_dataset(relation_key)
                full_results = ngrams_from_tupledict(raw_results)
                return full_results

        except KeyboardInterrupt:
            raise KeyboardInterrupt

    def create_and_run_query(
            self,
            words=None,
            postags=None,
            deprels=None,
            headids=None,
            ignore_position=False,
            absolute_position=False,
            limit=None,
            threshold=None,
            description=None,
            output=None):
        """Creates and runs an NgramQuery and returns a list of Ngrams.

        See create_query for details on creating a query.
        """

        query = self.create_query(
                words=words,
                postags=postags,
                deprels=deprels,
                headids=headids,
                ignore_position=ignore_position,
                absolute_position=absolute_position,
                limit=limit,
                threshold=threshold,
                description=description,
                output=output)

        return self.run_query(query)

    @classmethod
    def _make_join_context_query_plan(cls, query):
        subquery = cls._build_join_context_subquery(query)
        sql = ' '.join(SQL_CONTEXT_TEMPLATE.format(subquery=subquery).split())
        q_plan = JSON_CONTEXT_TEMPLATE
        q_plan['fragments'][0]['operators'][0]['sql'] = sql
        q_plan['rawQuery'] = query.description
        # Ugh line length.
        last_op = q_plan['fragments'][-1]['operators'][-1]
        last_op['relationKey']['relationName'] = query.output
        return q_plan

    @classmethod
    def _build_join_context_subquery(cls, query):
        sub_rel_str, sub_pred_str = cls._build_join_subquery_components(query)
        sub_template = "SELECT DISTINCT tt0.nid, tt0.freq FROM {} WHERE {}"

        if isinstance(query.threshold, int):
            sub_template = sub_template + \
                " AND tt0.freq >= {}".format(query.threshold)

        sub_template = sub_template + " ORDER BY tt0.freq DESC, tt0.nid ASC"

        if isinstance(query.limit, int):
            sub_template = sub_template + " LIMIT {}".format(query.limit)

        subquery = sub_template.format(sub_rel_str, sub_pred_str, query.limit)
        return subquery

    @classmethod
    def _build_join_subquery_components(cls, query):
        # cheating at refactoring is fun lol
        words = query.words
        postags = query.postags
        deprels = query.deprels
        headids = query.headids
        ignore_position = query.ignore_position
        absolute_position = query.absolute_position
        threshold = query.threshold
        ngram_length = query.ngram_length

        zipped = zip(words, postags, deprels, range(ngram_length), headids)

        relations = []
        predicates = []

        # get all appropriate pairs of tokens
        token_idx_pairs = [
            (i, j) for j in range(ngram_length) for i in range(j)]
            #if ignore_position else [(i, i+1) for i in range(ngram_length-1)]

        # create relations and predicates for each pair
        for i, pair in enumerate(token_idx_pairs):
            tka_idx, tkb_idx = pair
            token_pair = (zipped[tka_idx], zipped[tkb_idx])

            pair_id = "tt{}".format(i)

            if i > 0:
                predicates.append(
                    util.make_predicate(pair_id, "nid", "tt0.nid"))

            tka_raw, tkb_raw = token_pair

            def build_token_kwargs(tka_raw, tkb_raw):
                tka_kwargs = {}
                tkb_kwargs = {}

                tka_kwargs['word'] = tka_raw[0]
                tkb_kwargs['word'] = tkb_raw[0]

                tka_kwargs['postag'] = tka_raw[1]
                tkb_kwargs['postag'] = tkb_raw[1]

                tka_kwargs['deprel'] = tka_raw[2]
                tkb_kwargs['deprel'] = tkb_raw[2]

                if ignore_position:
                    tka_kwargs['offset'] = None
                    tkb_kwargs['offset'] = None
                elif absolute_position:
                    tka_kwargs['offset'] = tka_raw[3]+1
                    tkb_kwargs['offset'] = tkb_raw[3]+1
                else:
                    tka_kwargs['offset'] = None
                    tkb_kwargs['offset'] = tkb_raw[3] - tka_raw[3]

                head = None
                if tka_raw[4] == None and tkb_raw[4] == None:
                    pass
                elif tka_raw[3] == tkb_raw[4]:
                    head = "tka"
                elif tka_raw[3] == tkb_raw[4]:
                    head = "tkb"

                return (tka_kwargs, tkb_kwargs, head)

            tka_kwargs, tkb_kwargs, head = build_token_kwargs(tka_raw, tkb_raw)

            subrelations, subpredicates = cls._build_pair_predicate(
                pair_id, tka_kwargs, tkb_kwargs, head)

            relations.extend(subrelations)

            if ignore_position:
                sr, sp = cls._build_pair_predicate(
                    pair_id, tkb_kwargs, tka_kwargs, head)
                relations.extend(sr)

                sp1 = "({})".format(" AND ".join(subpredicates))
                sp2 = "({})".format(" AND ".join(sp))

                predicates.append("({})".format(" OR ".join((sp1, sp2))))

            else:
                subpredicate = "({})".format(" AND ".join(subpredicates))
                predicates.append(subpredicate)

        # put 'em all together
        sub_rel_str = ", ".join(set(relations))
        sub_pred_str = " AND ".join(set(predicates))

        return (sub_rel_str, sub_pred_str)

    @classmethod
    def _build_pair_predicate(
            cls, this_pair, tka_kwargs, tkb_kwargs, head=None):
        # kwargs: word, postag, deprel, offset

        subrelations = [util.aliased_relation(TT_RELATION, this_pair)]
        subpredicates = []

        i = int(this_pair[2:])

        def make_word_pred(tk, word):
            if word is None:
                return None

            joined_tk_words = ','.join(
                "'{}'".format(w) for w in word.split('|'))
            return util.make_predicate(
                    this_pair,
                    "{}_surface".format(tk),
                    "({})".format(joined_tk_words),
                    ' IN ')

        subpredicates.append(make_word_pred('tka', tka_kwargs['word']))
        subpredicates.append(make_word_pred('tkb', tkb_kwargs['word']))

        def make_postag_pred(tk, postag):
            if postag is None:
                return None

            this_pos = 'pos{}_{}'.format(i, tk)
            subrelations.append(util.aliased_relation(POS_RELATION, this_pos))

            these_postags = postag.split('|')

            pred1 = util.make_predicate(
                    this_pair,
                    "{}_posid".format(tk),
                    "{}.posid".format(this_pos))

            pred2 = util.make_predicate(
                    this_pos,
                    "postag",
                    "({})".format(','.join("'{}'".format(p) for p in
                                  these_postags)),
                    ' IN ')
            return " AND ".join((pred1, pred2))

        subpredicates.append(make_postag_pred('tka', tka_kwargs['postag']))
        subpredicates.append(make_postag_pred('tkb', tkb_kwargs['postag']))

        def make_deprel_pred(tk, deprel):
            if deprel is None:
                return None

            this_deprel = 'deprel{}_{}'.format(i, tk)
            subrelations.append(
                util.aliased_relation(DEP_RELATION, this_deprel))

            these_deprels = deprel.split('|')

            pred1 = util.make_predicate(
                    this_pair,
                    "{}_depid".format(tk),
                    "{}.depid".format(this_deprel))

            pred2 = util.make_predicate(
                    this_deprel,
                    "deprel",
                    "({})".format(','.join("'{}'".format(p) for p in
                                  these_deprels)),
                    ' IN ')

            return " AND ".join((pred1, pred2))

        subpredicates.append(make_deprel_pred('tka', tka_kwargs['deprel']))
        subpredicates.append(make_deprel_pred('tkb', tkb_kwargs['deprel']))

        # if first token's offset is not none, then the position is absolute --
        # (both should be set)
        if tka_kwargs['offset'] is not None:
            subpredicates.append(util.make_predicate(
                this_pair, "tka_position", tka_kwargs['offset']))

            subpredicates.append(util.make_predicate(
                this_pair, "tkb_position", tkb_kwargs['offset']))

        # otherwise, position is relative (but we still care)
        elif tkb_kwargs['offset'] is not None:
            subpredicates.append(util.make_predicate(
                this_pair, "tkb_position",
                "{}.tka_position".format(this_pair),
                '>'))

        if head is not None:
            if head == 'tka':
                subpredicates.append(util.make_predicate(
                    this_pair, "tkb_headposition",
                    "{}.{}".format(this_pair, "tka_position")))
            elif head == 'tkb':
                subpredicates.append(util.make_predicate(
                    this_pair, "tka_headposition",
                    "{}.{}".format(this_pair, "tkb_position")))

        return (subrelations, [s for s in subpredicates if s is not None])


class NgramDbQuery(object):

    def __init__(
            self,
            words=None,
            postags=None,
            deprels=None,
            headids=None,
            ignore_position=False,
            absolute_position=False,
            limit=None,
            threshold=None,
            description="[ ngramdb query ]",
            output="TEMPOUT"):

        # no conflicting args!!
        if ignore_position and absolute_position:
            raise ValueError("ignore_position and absolute_position cannot"
                             "both be True")

        # make sure we have enough reasonable words, or the query could freeze
        # the db
        if words is None:
            raise ValueError("'words' keyword argument must have a list of "
                             "at least " + str(MIN_WORD_COUNT) + " word(s)")

        if sum(1 for w in words if w is not None) < MIN_WORD_COUNT:
            raise ValueError("'words' keyword argument must have a list of "
                             "at least " + str(MIN_WORD_COUNT) + " word(s)")

        if all(len(w) < MIN_WORD_LEN for w in words if w is not None):
            raise ValueError("'words' keyword argument must have at least 1 "
                             "word that is " + str(MIN_WORD_LEN) +
                             " or more letters long")

        # error check the rest of the arguments
        try:
            self.ngram_length, max_name = max([(len(x), y) for x, y in zip(
                        (words, postags, deprels, headids),
                        ("words", "postags", "deprels", "headids"))
                if x is not None],
                key=lambda x: x[0])

        except TypeError as e:
            raise ValueError("Must provide at least keyword arg of 'words', "
                             "'posids', 'depids', 'headids'")

        def check_arg(kw, arg):
            if arg is not None:
                if len(arg) != self.ngram_length:
                    raise ValueError(
                        "{} and {} must have same number of items"
                        " (need {}, found {})".format(
                            max_name, kw, self.ngram_length, len(arg)))

                else:
                    return arg

            else:
                return [None for _ in range(self.ngram_length)]

        # normalize arguments
        self.words = [x.lower() if x is not None else x
                      for x in check_arg('words', words)]
        self.postags = [x.upper() if x is not None else x
                        for x in check_arg('posids', postags)]
        self.deprels = [x for x in check_arg('depids', deprels)]
        self.headids = [int(x) if x is not None else x
                        for x in check_arg('headids', headids)]

        # set all the other stuff
        self.ignore_position = ignore_position
        self.absolute_position = absolute_position
        self.limit = limit
        self.threshold = threshold
        self.description = description
        self.output = output

    def __eq__(self, other):
        '''
        Implementing for caching with these as keys to a dictionary.

        Would be as simple as compairing self.__dict__.items() (as below in
        __str__) but there is some output info stored that we don't want to
        compare.
        '''
        return (self.words == other.words and
                self.postags == other.postags and
                self.deprels == other.deprels and
                self.headids == other.headids and
                self.ignore_position == other.ignore_position and
                self.absolute_position == other.absolute_position and
                self.limit == other.limit and
                self.threshold == other.threshold)

    def __ne__(self, other):
        '''
        Yep, this doesn't happen automatically. Thanks, python.
        '''
        return not self == other

    def __hash__(self):
        '''
        Implementing for caching with these as keys to a dictionary.
        '''
        # Can't use None in hashing because it can return inconsistent numbers
        # (as it just uses None's address in memory, which changes if the OS
        # has memory randomization turned on).
        words = tuple([w if w is not None else '' for w in self.words])
        postags = tuple([p if p is not None else '' for p in self.postags])
        deprels = tuple([d if d is not None else '' for d in self.deprels])
        headids = tuple([h if h is not None else '' for h in self.headids])
        limit = 0 if self.limit is None else self.limit
        threshold = 0 if self.threshold is None else self.threshold
        return hash((
            words,
            postags,
            deprels,
            headids,
            self.ignore_position,
            self.absolute_position,
            limit,
            threshold))

    def __str__(self):
        values = ["{}={}".format(k, repr(v))
                  for k, v in sorted(self.__dict__.items(),
                                     key=lambda x: type(x[1]))]
        return '\n'.join(values)
