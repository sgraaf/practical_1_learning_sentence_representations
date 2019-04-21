from pathlib import Path

import numpy as np
import torch
from nltk.tokenize import word_tokenize
from torchtext.data import Field
from torchtext.datasets import SNLI

data_dir = Path.cwd().parent / 'data'


def get_SNLI(text_field, label_field, percentage=None):
    """
    Returns the SNLI dataset in splits

    :param torchtext.data.Field text_field: the field that will be used for premise and hypothesis data
    :param torchtext.data.Field label_field: the field that will be used for label data
    :param float percentage: the percentage of the data to use
    :returns: the SNLI dataset in splits
    :rtype: tuple
    """
    train, dev, test = SNLI.splits(text_field, label_field)

    if percentage:
        train.examples = train.examples[:np.int(np.ceil(len(train) * percentage))]
        dev.examples = dev.examples[:np.int(np.ceil(len(dev) * percentage))]
        test.examples = test.examples[:np.int(np.ceil(len(test) * percentage))]

    return train, dev, test


def get_GloVe():
    """
    Returns the GloVe vectors (pre-filtered on the words present in the SNLI dataset and SentEval tasks)

    :returns: the GloVe vectors
    :rtype: dict{str: torch.tensor}
    """
    GloVe_vectors = {}

    with open(data_dir / 'small_glove_torchnlp.txt', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        for line in lines:
            word, vec = line.split(' ', 1)
            GloVe_vectors[word] = torch.tensor(np.array(vec, dtype=float))

    return GloVe_vectors


def load_data(percentage=None):
    """
    Load all relevant data (GloVe vectors & SNLI dataset) for our experiments

    :param float percentage: the percentage of the data to use
    :returns: the data (train, dev, test, text_field and label_field)
    :rtype: tuple(Dataset, Dataset, Dataset, Field, Field)
    """
    # get the GloVe vectors
    GloVe_vectors = get_GloVe()

    # set the dataset fields
    text_field = Field(sequential=True,
                       use_vocab=True,
                       lower=True,
                       tokenize=word_tokenize,
                       include_lengths=True)

    label_field = Field(sequential=False,
                        use_vocab=True,
                        pad_token=None,
                        unk_token=None,
                        is_target=True)

    # get the SNLI dataset in splits
    train, dev, test = get_SNLI(text_field, label_field)

    # build the text_field vocabulary from all data splits
    text_field.build_vocab(train, dev, test, vectors=GloVe_vectors)

    # build the label_field vocabulary from the train split
    label_field.build_vocab(train)

    return train, dev, test, text_field, label_field
