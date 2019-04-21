import torch
import torch.nn as nn

from utils import (get_last_hidden_states, pack_padded_sequence_batch,
                   pad_packed_sequence_batch)


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
        Forward pass of the batch through the BiLSTM encoder
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
        super(MaxBiLSTM).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 2 * hidden_dim
        self.cell = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, bidirectional=True)

    def forward(self, batch, batch_lens):
        """
        Forward pass of the batch through the MaxBiLSTM encoder
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
