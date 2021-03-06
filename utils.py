# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random


def read_data(fname):
    data = []
    for line in file(fname):
        label, text = line.strip().lower().split("\t", 1)
        data.append((label, text))
    return data


def text_to_bigrams(text):
    return ["%s%s" % (c1, c2) for c1, c2 in zip(text, text[1:])]


TRAIN = [(l, text_to_bigrams(t)) for l, t in read_data("train")]
DEV = [(l, text_to_bigrams(t)) for l, t in read_data("dev")]

from collections import Counter

fc = Counter()
for l, feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x, c in fc.most_common(600)])

# label strings to IDs
L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
# feature strings (bigrams) to IDs
F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}


########################


def getTEST():
    if getTEST.TEST is None:
        getTEST.TEST = [text_to_bigrams(txt) for label, txt in read_data("test")]
    return getTEST.TEST


getTEST.TEST = None


def get_unigrams():
    if get_unigrams.list is None:

        TRAIN_UNI = [(l, list(t)) for l, t in read_data("train")]
        TEST_UNI = [(l, list(t)) for l, t in read_data("dev")]

        myfc = Counter()
        for l, feats in TRAIN_UNI:
            myfc.update(feats)

        # 600 most common bigrams in the training set.
        vocab_uni = set([x for x, c in myfc.most_common(600)])
        # feature strings (bigrams) to IDs
        F2I_UNI = {f: i for i, f in enumerate(list(sorted(vocab_uni)))}

        get_unigrams.list = [TRAIN_UNI, TEST_UNI, F2I_UNI]
    return get_unigrams.list


get_unigrams.list = None

if __name__ == "__main__":
    print 'TEST'
    for f in getTEST():
        print f
#######################
