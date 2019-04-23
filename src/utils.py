from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pandas import DataFrame as df
from pandas import read_csv

plt.style.use('seaborn-white')


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
    batch = batch[:, idxs_unsort, :]
    batch_lens = batch_lens_sorted[idxs_unsort]

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
    last_tokens = [i * max(batch_lens) + batch_lens[i] - 1 for i in range(len(batch_lens))]

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
  FLAGS_dict = vars(FLAGS)
  longest_key_length = max(len(key) for key in FLAGS_dict)
  for key, value in vars(FLAGS).items():
    print(f'{key:<{longest_key_length}}: {value}')


def print_model_parameters(model):
    """
    Prints all model parameters and their values.

    :param nn.Module model: the model
    """
    # print(f'Model: {model.__class__.__name__}')
    print('Parameters:')
    named_parameters = model.named_parameters()
    longest_param_name_length = max([len(named_param[0]) for named_param in named_parameters])
    for name, param in named_parameters:
        print(f' {name:<{longest_param_name_length}}: {param}')


def batch_accuracy(batch_y, pred_y):
    """
    Computes the accuracy of the predicted labels

    :param torch.tensor batch_y: the true labels
    :param torch.tensor pred_y: the one-hot encoded predicted labels
    """
    return (batch_y == pred_y.argmax(dim=1)).float().mean().item()


def create_checkpoint(checkpoint_dir, epoch, model, optimizer, results, best_accuracy):
    """
    Creates a checkpoint for the current epoch

    :param pathlib.Path checkpoint_dir: the path of the directory to store the checkpoints in
    :param int epoch: the current epoch
    :param nn.Module model: the model
    :param optim.Optimizer optimizer: the optimizer
    :param dict results: the results
    :param float best_accuracy: the best accuracy thus far
    """
    print('Creating checkpoint...', end=' ')
    checkpoint_path = checkpoint_dir / (f'{model.__class__.__name__}_{model.encoder.__class__.__name__}_{epoch}_checkpoint.pt')
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results,
            'best_accuracy': best_accuracy
        },
        checkpoint_path
    )
    print('Done!')


def save_model(model_dir, model):
    """
    Saves the model

    :param pathlib.Path model_dir: the path of the directory to save the models in
    :param nn.Module model: the model
    """
    print('Saving the model...', end=' ')
    model_path = model_dir / f'{model.__class__.__name__}_{model.encoder.__class__.__name__}_model.pt'
    encoder_path = model_dir / f'{model.__class__.__name__}_{model.encoder.__class__.__name__}_encoder.pt'
    torch.save(model.state_dict(), model_path)
    torch.save(model.encoder.state_dict(), encoder_path)
    print('Done!')


def save_training_results(results_dir, results, model):
    """
    Saves the training results

    :param pathlib.Path results_dir: the path of the directory to save the results in
    :param dict results: the results
    :param nn.Module model: the model
    """
    print('Saving the results...', end=' ')
    results_df = df.from_dict(results)
    results_path = results_dir / f'{model.__class__.__name__}_{model.encoder.__class__.__name__}_results.csv'
    results_df.to_csv(results_path, sep=';', encoding='utf-8')
    print('Done!')


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Loads a checkpoint

    :param pathlib.Path checkpoint_path: the path of the checkpoint
    :param nn.Module model: the model
    :param optim.Optimizer optimizer: the optimizer
    :returns: tuple of epoch, model, optimizer, results and best_accuracy of the checkpoint
    :rtype: tuple(int, nn.Module, optim.Optimizer, dict, float)
    """
    print('Loading checkpoint...', end=' ')
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    results = checkpoint['results']
    best_accuracy = checkpoint['best_accuracy']
    print('Done!')

    return epoch, results, best_accuracy


def load_model_state(model_path, model):
    """
    Loads a model

    :param pathlib.Path model_path: the path of the model
    :param nn.Module model: the model
    """
    print('Loading the model state...', end=' ')
    if next(model.parameters()).is_cuda:
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, log: storage))
    else:
        model.load_state_dict(torch.load(model_path))
    print('Done!')


def load_encoder_state(encoder_path, encoder):
    """
    Loads an encoder

    :param pathlib.Path encoder_path: the path of the encoder
    :param nn.Module encoder: the encoder
    """
    print('Loading the encoder state...', end=' ')
    encoder.load_state_dict(torch.load(encoder_path))
    print('Done!')


def save_eval_results(results_dir, results, encoder, tasks):
    """
    Saves the eval results

    :param pathlib.Path results_dir: the path of the directory to save the results in
    :param dict results: the results
    :param nn.Module encoder: the encoder
    :param list tasks: the list of tasks used for the eval
    """
    print('Saving the results...', end=' ')
    results_df = df.from_dict(results)
    results_path = results_dir / f'{encoder.__class__.__name__}_{",".join(tasks)}_eval_results.csv'
    results_df.to_csv(results_path, sep=';', encoding='utf-8')
    print('Done!')


def plot_training_results(results_dir):
    results_files = sorted(results_dir.glob('*.csv'))
    if len(results_files) == 0:
        raise ValueError('No results found in the results directory')

    for results_file in results_files:
        # load the results
        df = read_csv(results_file, sep=';')
        x = df['Unnamed: 0'] + 1
        fig, ax1 = plt.subplots()
        ax1.plot(x, df['dev_accuracy'], color='tab:orange', marker='o', label='Dev accuracy')
        ax1.plot(x, df['train_accuracy'], color='tab:red', marker='o', label='Train accuracy')
        ax1.axhline(df['test_accuracy'], color='tab:brown')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params('y')
        ax1.set_ylim([0, 1])

        ax2 = ax1.twinx()
        ax2.plot(x, df['dev_loss'], color='tab:blue', marker='o', label='Dev loss')
        ax2.plot(x, df['train_loss'], color='tab:green', marker='o', label='Train loss')
        ax2.axhline(df['test_loss'], color='tab:olive')
        ax2.set_ylabel('Loss')
        ax2.tick_params('y')
        # ax2.set_ylim([0, 5])

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc=0)
        plt.tight_layout()
        plt.savefig(results_file.parent / (results_file.stem + '_accuracy_loss_curves.png'))
        plt.show()
