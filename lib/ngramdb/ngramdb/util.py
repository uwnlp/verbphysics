def aliased_relation(relation, alias):
    return ' '.join((relation, alias))


def make_predicate(relation_alias, attribute, value, relationship='='):
    return "{}.{}{}{}".format(relation_alias, attribute, relationship, value)


def collapse_ngram_surface(ngrams):
    return sorted((
        (k, sum(x[1] for x in g))
        for k, g in itertools.groupby((
            (n.surface(), n.freq)
            for n in sorted(
                ngrams,
                key=lambda x: x.surface())),
            lambda x: x[0])
        ), key=lambda x: x[1], reverse=True)


def pprint_ngram_list(ngram_list):
    surface_width = max(
        max(len(t.surface) for n in ngram_list for t in n) + 1, 6)

    ngram_format_str = "ID: {}\tFreq: {}\tHeight: {}"
    token_format_str = \
        "{0:>2}\t{1:<" + str(surface_width) + "}\t{2:<5}\t{3}{4}"

    all_lines = []

    for n in ngram_list:
        try:
            all_lines.append(
                ngram_format_str.format(n.nid, n.freq, n.height)
            )
        except AttributeError:
            all_lines.append(
                ngram_format_str.format(n.nid, n.freq, "")
            )

        all_lines += [
            token_format_str.format(
                t.position,
                t.surface,
                t.postag,
                t.deprel,
                '-' + str(t.headposition) if t.headposition > -1 else '')
            for t in n]

        all_lines.append('')

    # return '\n'.join(all_lines)
    print('\n'.join(all_lines))


def collapsed_histogram(kv_list):
    c = Counter()
    for k, v in kv_list:
        c[k] += v
    return c
