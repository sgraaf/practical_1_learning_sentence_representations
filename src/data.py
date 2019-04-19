from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from torchnlp.datasets import snli_dataset
from torchnlp.word_to_vector import GloVe

data_dir = Path.cwd().parent / 'data'


def snli(percentage=None):
    """
    Returns the SNLI dataset, splits included

    :param float percentage: the percentage of the data to use
    :returns: the SNLI dataset in splits
    :rtype: tuple
    """
    train, dev, test = snli_dataset(data_dir, train=True, dev=True, test=True)

    if percentage:
        train = train[:np.int(np.ceil(len(train) * percentage))]
        dev = dev[:np.int(np.ceil(len(dev) * percentage))]
        test = test[:np.int(np.ceil(len(test) * percentage))]

    return train, dev, test


def tokenize(datasets):
    """
    Returns the set of tokens present in the union of the datasets

    :param tuple datasets: the datasets
    :returns: the set of tokens
    :rtype: set
    """
    tokens = set()

    for dataset in datasets:
        for item in dataset:
            for text_type in ['premise', 'hypothesis']:
                tokens.update([token.lower() for token in word_tokenize(item[text_type])])

    return tokens


def glove():
    """
    Returns the glove vectors (pre-filtered on the SNLI dataset and SentEval tasks)

    :returns: the glove vectors
    :rtype: dict{str: torch.tensor}
    """
    glove_vectors = {}

    with open(data_dir / 'small_glove_torchnlp.txt', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        for line in lines:
            key = line.split(' ')[0]
            value = torch.tensor(np.array(line.split(' ')[1:], dtype=float))
            glove_vectors[key] = value

    return glove_vectors
