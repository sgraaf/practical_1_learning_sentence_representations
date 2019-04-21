import torch
from pandas import DataFrame as df

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


def print_flags(FLAGS):
  """
  Prints all entries in FLAGS Namespace.

  :param Namespace FLAGS: the FLAGS Namespace
  """
  for key, value in FLAGS.items():
    print(f'{key} : {value}')


def print_model_parameters(model):
    """
    Prints all model parameters and their values.

    :param nn.Module model: the model
    """
    # print(f'Model: {model.__class__.__name__}')
    print('Parameters:')
    named_parameters = model.named_parameters()
    longest_param_name = max([len(named_param[0]) for named_param in named_parameters])
    for name, param in named_parameters:
        print(f' {name:<{longest_param_name}} {param}')


def batch_accuracy(batch_y, pred_y):
    """
    Computes the accuracy of the predicted labels

    :param torch.tensor batch_y: the true labels
    :param torch.tensor pred_y: the one-hot encoded predicted labels
    """
    return (batch_y == pred_y.argmax(dim=1)).float().mean().item()


def create_checkpoint(checkpoint_path, epoch, model, optimizer, results, best_accuracy):
    """
    Creates a checkpoint for the current epoch

    :param pathlib.Path checkpoint_path: the path of the directory to store the checkpoints in
    :param int epoch: the current epoch
    :param nn.Module model: the model
    :param optim.Optimizer optimizer: the optimizer
    :param dict results: the results
    :param float best_accuracy: the best accuracy thus far
    """
    print('Creating checkpoint...', end=' ')
    checkpoint_name = checkpoint_path / (f'{model.__class__.__name__}_{model.encoder.__class__.__name__}_{epoch}_checkpoint.pt')
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results,
            'best_accuracy': best_accuracy
        },
        checkpoint_name
    )
    print('Done!')


def save_model(model_dir, model):
    """
    Saves the model

    :param pathlib.Path model_dir: the path of the directory to save the models in
    :param nn.Module model: the model
    """
    print('Saving the model...', end=' ')
    model_name = model_dir / f'{model.__class__.__name__}_{model.encoder.__class__.__name__}_model.pt'
    encoder_name = model_dir / f'{model.__class__.__name__}_{model.encoder.__class__.__name__}_encoder.pt'
    torch.save(model.state_dict(), model_name)
    torch.save(model.encoder.state_dict(), encoder_name)
    print('Done!')


def save_results(results_dir, results, model):
    """
    Saves the results

    :param pathlib.Path results_dir: the path of the directory to save the results in
    :param dict results: the results
    :param nn.Module model: the model
    """
    print('Saving the results...', end=' ')
    results_df = df.from_dict(results)
    results_name = results_dir / f'{model.__class__.__name__}_{model.encoder.__class__.__name__}_results.csv'
    results_df.to_csv(results_name, sep=';', encoding='utf-8')
    print('Done!')