# imports
import argparse
from os.path import getctime
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator

from data import load_data
from encoders import LSTM, Baseline, BiLSTM, MaxBiLSTM
from model import InferSent
from utils import (batch_accuracy, create_checkpoint, load_checkpoint,
                   print_flags, print_model_parameters, save_model,
                   save_training_results)

# defaults
FLAGS = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = Path.cwd().parent
DIV_LEARNING_RATE = 5
MIN_LEARNING_RATE = 1e-5

ENCODER_TYPE_DEFAULT = 'Baseline'
CHECKPOINT_PATH_DEFAULT = ROOT_DIR / 'output' / 'checkpoints'
MODELS_PATH_DEFAULT = ROOT_DIR / 'output' / 'models'
RESULTS_PATH_DEFAULT = ROOT_DIR / 'output' / 'results'
PERCENTAGE_DEFAULT = 0.05
LEARNING_RATE_DEFAULT = 0.1
WEIGHT_DECAY_DEFAULT = 0.99
BATCH_SIZE_DEFAULT = 64
MAX_EPOCHS_DEFAULT = 20


def train():
    encoder_type = FLAGS.encoder_type
    checkpoint_path = Path(FLAGS.checkpoint_path)
    models_path = Path(FLAGS.models_path)
    results_path = Path(FLAGS.results_path)
    percentage = FLAGS.percentage
    learning_rate = FLAGS.learning_rate
    weight_decay = FLAGS.weight_decay
    batch_size = FLAGS.batch_size
    max_epochs = FLAGS.max_epochs

    # create non-existing directories
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)
    if not models_path.exists():
        models_path.mkdir(parents=True)
    if not results_path.exists():
        results_path.mkdir(parents=True)

    # load the data
    print('Loading the data...', end=' ')
    train, dev, test, text_field, label_field = load_data()
    embedding = nn.Embedding.from_pretrained(text_field.vocab.vectors)
    embedding.requires_grad = False
    print('Done!')

    print(f'SNLI dataset size (using {percentage * 100} % of the data):')
    print(f' Train: {len(train): {len(str(max(len(train), len(dev), len(test))))}} samples')
    print(f' Dev:   {len(dev): {len(str(max(len(train), len(dev), len(test))))}} samples')
    print(f' Test:  {len(test): {len(str(max(len(train), len(dev), len(test))))}} samples')

    # set the encoder
    print('Initializing the encoder...', end=' ')
    if encoder_type == 'Baseline':
        encoder = Baseline()
    elif encoder_type == 'LSTM':
        encoder = LSTM()
    elif encoder_type == 'BiLSTM':
        encoder = BiLSTM()
    elif encoder_type == 'MaxBiLSTM':
        encoder = MaxBiLSTM()
    print('Done!')
    print(f'Succesfully initialized the {encoder.__class__.__name__} encoder!')

    # set the InferSent model
    print('Initializing the model...', end=' ')
    model = InferSent(
        input_dim=4*encoder.output_dim,
        hidden_dim=512,
        output_dim=3,
        embedding=embedding,
        encoder=encoder
    )
    model.to(DEVICE)
    print('Done!')
    print(f'Succesfully initialized the {model.__class__.__name__} model!')
    print_model_parameters(model)

    # set the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # load the last checkpoint (if it exists)
    checkpoints = list(checkpoint_path.glob(f'{model.__class__.__name__}_{encoder.__class__.__name__}_*.pt'))
    if len(checkpoints) > 0:
        # load the latest checkpoint
        checkpoints.sort(key=getctime)
        latest_checkpoint_path = checkpoints[-1]
        epoch, results, best_accuracy = load_checkpoint(latest_checkpoint_path, model, optimizer)
    else:
         # initialize the epoch, results and best_accuracy
        epoch = 0
        results = {
            'train_accuracy': [],
            'train_loss': [],
            'dev_accuracy': [],
            'dev_loss': [],
            'test_accuracy': None,
            'test_loss': None
        }
        best_accuracy = 0.0

    if epoch == 0:
        print(f'Starting training at epoch {epoch + 1}...')
    else:
        print(f'Resuming training from epoch {epoch + 1}...')
    
    # put the model in train mode
    model.train()

    for i in range(epoch, max_epochs):
        print(f'Epoch {i+1:0{len(str(max_epochs))}}/{max_epochs}:')

        epoch_results = {
            'train_accuracy': [],
            'train_loss': [],
            'dev_accuracy': [],
            'dev_loss': []
        }

        # get the batch iterators for mini-batching
        train_iter, dev_iter, test_iter = BucketIterator.splits(
            datasets=(train, dev, test), 
            batch_sizes=(batch_size, batch_size, batch_size), 
            device=DEVICE, 
            shuffle=True
        )

        # iterate over the train data mini-batches for training
        for batch in train_iter:
            try:
                batch_premises = batch.premise
                batch_hypotheses = batch.hypothesis
                batch_y = batch.label

                # forward pass
                pred_y = model.forward(batch_premises, batch_hypotheses)
                train_loss = criterion(pred_y, batch_y)

                # (re)set the optimizer gradient to 0
                optimizer.zero_grad()
            
                # backward pass
                train_loss.backward()
                optimizer.step()

                # compute and record the results
                epoch_results['train_accuracy'].append(batch_accuracy(batch_y, pred_y))
                epoch_results['train_loss'].append(train_loss.item())
            except RuntimeError:  # happens at the last batch for some odd reason
                # print(f'batch_premises: {batch_premises}')
                # print(f'batch_hypotheses: {batch_hypotheses}')
                # print(f'batch_y: {batch_y}')
                # print(f'batch_y.shape: {batch_y.shape}')
                # print(f'pred_y.shape: {pred_y.shape}')
                # print(f'pred_y: {pred_y}')
                # print(f'train loss: {train_loss}')
                pass

        # compute and record the training results means for this epoch
        results['train_accuracy'].append(np.mean(epoch_results['train_accuracy']))
        results['train_loss'].append(np.mean(epoch_results['train_loss']))
        print(f" TRAIN accuracy: {results['train_accuracy'][-1]:0.10f}, loss: {results['train_loss'][-1]:0.10f}")

        # iterate over the dev data mini-batches for evaluation / tuning
        for batch in dev_iter:
            batch_premises = batch.premise
            batch_hypotheses = batch.hypothesis
            batch_y = batch.label

            # forward pass
            pred_y = model.forward(batch_premises, batch_hypotheses)
            dev_loss = criterion(pred_y, batch_y)

            # compute and record the results
            epoch_results['dev_accuracy'].append(batch_accuracy(batch_y, pred_y))
            epoch_results['dev_loss'].append(dev_loss.item())

        # compute and record the evaluation results means for this epoch
        results['dev_accuracy'].append(np.mean(epoch_results['dev_accuracy']))
        results['dev_loss'].append(np.mean(epoch_results['dev_loss']))
        print(f" DEV   accuracy: {results['dev_accuracy'][-1]:0.10f}, loss: {results['dev_loss'][-1]:0.10f}")

        # create a checkpoint
        create_checkpoint(checkpoint_path, i, model, optimizer, results, best_accuracy)

        # decrease the learning rate if accuracy decreases
        if results['dev_accuracy'][-1] < best_accuracy:
            learning_rate /= DIV_LEARNING_RATE
            print(f'New learning rate: {learning_rate}')
            
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
        else:
            best_accuracy = results['dev_accuracy'][-1]
            save_model(models_path, model)

        # terminate training if the learning rate becomes too low
        if learning_rate < MIN_LEARNING_RATE:
            print('Terminating the training process due to learning rate decay')
            break

    test_results = {
        'test_accuracy': [],
        'test_loss': []
    }

    # iterate over the test data mini-batches for evaluation
    for batch in test_iter:
        batch_premises = batch.premise
        batch_hypotheses = batch.hypothesis
        batch_y = batch.label

        # forward pass
        pred_y = model.forward(batch_premises, batch_hypotheses)
        dev_loss = criterion(pred_y, batch_y)

        # compute and record the results
        test_results['test_accuracy'].append(batch_accuracy(batch_y, pred_y))
        test_results['test_loss'].append(dev_loss.item())

    results['test_accuracy'] = np.mean(test_results['test_accuracy'])
    results['test_loss'] = np.mean(test_results['test_loss'])
    print(f" TEST accuracy: {results['test_accuracy']:0.10f}, loss: {results['test_loss']:0.10f}")
    
    # save the results
    save_training_results(results_path, results, model)


def main():
    # print all flags
    print_flags(FLAGS)

    # start the timer
    start_time = time()

    # train the model
    train()

    # end the timer
    end_time = time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f'Done training in {minutes}:{seconds} minutes.')


if __name__ == '__main__':
    # cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', type=str, default=ENCODER_TYPE_DEFAULT,
                        help='Encoder type (i.e: Baseline, LSTM, BiLSTM or MaxBiLSTM)')
    parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH_DEFAULT,
                        help='Path of directory to store checkpoints')
    parser.add_argument('--models_path', type=str, default=MODELS_PATH_DEFAULT,
                        help='Path of directory to store models')
    parser.add_argument('--results_path', type=str, default=RESULTS_PATH_DEFAULT,
                        help='Path of directory to store results')
    parser.add_argument('--percentage', type=float, default=PERCENTAGE_DEFAULT,
                        help='Percentage of data to be used (for training, testing, etc.)')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY_DEFAULT,
                        help='Weight decay of the learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--max_epochs', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Max number of epochs for training')
    FLAGS, unparsed = parser.parse_known_args()

    main()
