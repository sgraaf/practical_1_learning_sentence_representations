import torch

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


def get_features(u_batch, v_batch):
    """
    Get the features from the encoded premises and hypotheses
    :param torch.tensor u_batch: the encoded premises
    :param torch.tensor v_batch: the encoded hypotheses
    :returns: the features of these encoded sentences
    :rtype: torch.tensor
    """
    return torch.cat((u_batch, v_batch, torch.abs(u_batch - v_batch), u_batch * v_batch), dim=1)