import torch
import torch.nn as nn

class Baseline(nn.Module):
    """
    Baseline encoder that averages word embeddings of a sentence to obtain a sentence representation
    """

    def __init__(self, embedding_dim=300):
        """
        Initialize the Baseline encoder (average of word embeddings)

        :param int embedding_dim: the dimensionality of the word embeddings
        """
        super(Baseline).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, batch, batch_lens):
        """
        Forward pass of the batch through the Baseline encoder (average of word embeddings)
        """
        return batch.sum(dim=0) / batch_lens.view(-1, 1).to(torch.float)


def pack_padded_sequence_batch(batch, batch_lens):
    """
    Pack the padded batch tensor.
    Inspired by https://stackoverflow.com/q/49203019

    :param torch.tensor batch: the batch tensor
    :param torch.tensor batch_lens: the batch sentence lengths
    :returns: the packed sequence, the sorted batch lengths and the indices to unsort them
    :rtype: tuple(PackedSequence, torch.tensor, torch.tensor)
    """
    # sort the batch lengths in descending order
    batch_lens_sorted, idxs_sorted = batch_lens.sort(dim=0, descending=True)

    # sort the batch according to this descending order of lengths
    batch_sorted = batch[:, idxs_sorted, :]

    # get the indices to unsort the batch later
    _, idxs_unsort = idxs_sorted.sort(dim=0, descending=False)

    # pack the padded batch
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(batch_sorted, batch_lens_sorted)

    return packed_sequence, batch_lens_sorted, idxs_unsort


def pad_packed_sequence_batch(packed_sequence, batch_lens_sorted, idxs_unsort):
    """
    Pad the packed batch tensor.
    Inspired by https://stackoverflow.com/q/49203019

    :param torch.tensor packed_sequence: the packed batch tensor
    :param torch.tensor batch_lens_sorted: the sorted batch sentence lengths
    :returns: the padded sequence and the unsorted batch lengths
    :rtype: tuple(torch.tensor, torch.tensor)
    """
    # pad the packed batch
    batch, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_sequence)

    # unsort the batch and batch_lens
    batch = batch[:, idxs_unsort: :]
    batch_lens = batch_lens_sorted[idxs_unsort, :]

    return batch, batch_lens


def get_last_hidden_states(batch, batch_lens, output_dim):
    """
    Get the last hidden states of the batch.

    :param torch.tensor batch: the batch tensor
    :param torch.tensor batch_lens: the batch sentence lengths
    :param int output_dim: the dimensionality of the output
    :returns: the last hidden states of the batch
    :rtype: torch.tensor
    """
    # get the indices of the last tokens
    last_tokens = [i * max(batch_lens) + batch_lens[i] for i in range(len(batch_lens))]

    # get the hidden states of these last tokens
    batch = batch.view(-1, output_dim)
    batch_hidden = batch[last_tokens, :]

    return batch_hidden


class LSTM(nn.Module):
    """
    LSTM encoder that uses Long Short-Term Memory cells to obtain a sentence representation
    """

    def __init__(self, embedding_dim=300, hidden_dim=2048):
        """
        Initialize the LSTM encoder

        :param int embedding_dim: the dimensionality of the word embeddings
        :param int hidden_dim: the dimensionality of the hidden state
        """
        super(LSTM).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.cell = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim)

    def forward(self, batch, batch_lens):
        """
        Forward pass of the batch through the LSTM encoder
        """
        # pack the padded sequence
        packed_sequence, batch_lens_sorted, idxs_unsort = pack_padded_sequence_batch(batch, batch_lens)

        # pass the packed sequence through the LSTM cell
        y_packed, _ = self.cell(packed_sequence)

        # pad the packed sequence
        y_padded, y_lens = pad_packed_sequence_batch(y_packed, batch_lens_sorted, idxs_unsort)

        # get the hidden states of the last words
        y = get_last_hidden_states(y_padded, y_lens, self.output_dim)

        return y


class BiLSTM(nn.Module):
    """
    BiLSTM encoder that uses bi-directional Long Short-Term Memory cells to obtain a sentence representation
    """

    def __init__(self, embedding_dim=300, hidden_dim=2048):
        """
        Initialize the BiLSTM encoder

        :param int embedding_dim: the dimensionality of the word embeddings
        :param int hidden_dim: the dimensionality of the hidden state
        """
        super(BiLSTM).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 2 * hidden_dim
        self.cell = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, bidirectional=True)

    def forward(self, batch, batch_lens):
        """
        Forward pass of the batch through the LSTM encoder
        """
        # pack the padded sequence
        packed_sequence, batch_lens_sorted, idxs_unsort = pack_padded_sequence_batch(batch, batch_lens)

        # pass the packed sequence through the LSTM cell
        y_packed, _ = self.cell(packed_sequence)

        # pad the packed sequence
        y_padded, y_lens = pad_packed_sequence_batch(y_packed, batch_lens_sorted, idxs_unsort)
        
        # get the hidden states of the last words
        y = get_last_hidden_states(y_padded, y_lens, self.output_dim)

        return y


class MaxBiLSTM(nn.Module):
    """
    MaxBiLSTM encoder that uses bi-directional Long Short-Term Memory cells with max-pooling to obtain a sentence representation
    """

    def __init__(self, embedding_dim=300, hidden_dim=2048):
        """
        Initialize the MaxBiLSTM encoder

        :param int embedding_dim: the dimensionality of the word embeddings
        :param int hidden_dim: the dimensionality of the hidden state
        """
        super(BiLSTM).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 2 * hidden_dim
        self.cell = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, bidirectional=True)

    def forward(self, batch, batch_lens):
        """
        Forward pass of the batch through the LSTM encoder
        """
        # pack the padded sequence
        packed_sequence, batch_lens_sorted, idxs_unsort = pack_padded_sequence_batch(batch, batch_lens)

        # pass the packed sequence through the LSTM cell
        y_packed, _ = self.cell(packed_sequence)

        # pad the packed sequence
        y_padded, y_lens = pad_packed_sequence_batch(y_packed, batch_lens_sorted, idxs_unsort)
        
        # perform "max-pooling"
        y, _ = torch.max(y_padded, dim=0)

        return y
