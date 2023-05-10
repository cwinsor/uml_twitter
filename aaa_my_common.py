from collections import defaultdict


def my_group_by(iterable, keyfunc, valfunc):
    """Because itertools.groupby is tricky to use

    The stdlib method requires sorting in advance, and returns iterators not
    lists, and those iterators get consumed as you try to use them, throwing
    everything off if you try to look at something more than once.

    Reference: https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby
    """
    ret = defaultdict(list)
    for item in iterable:
        ret[keyfunc(item)].append(valfunc(item))
    return dict(ret)