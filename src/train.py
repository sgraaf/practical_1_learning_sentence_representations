# imports
import argparse
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
from utils import (batch_accuracy, create_checkpoint, print_flags,
                   print_model_parameters, save_model, save_results)

# defaults
FLAGS = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = Path.cwd().parent

MODEL_TYPE_DEFAULT = 'Baseline'
CHECKPOINT_PATH_DEFAULT = ROOT_DIR / 'output' / 'checkpoints'
TRAIN_DATA_PATH_DEFAULT = ROOT_DIR / 'output' / 'results'
PERCENTAGE_DEFAULT = 0.05
LEARNING_RATE_DEFAULT = 0.1
WEIGHT_DECAY_DEFAULT = 0.99
BATCH_SIZE_DEFAULT = 64
MAX_EPOCHS_DEFAULT = 20

MODELS_DIR = ROOT_DIR / 'output' / 'models'
DIV_LEARNING_RATE = 5
MIN_LEARNING_RATE = 1e-5


def train():
    # get the flags
    model_type = FLAGS.model_type
    checkpoint_path = Path(FLAGS.checkpoint_path)
    train_data_path = Path(FLAGS.train_data_path)
    percentage = FLAGS.percentage
    learning_rate = FLAGS.learning_rate
    weight_decay = FLAGS.weight_decay
    batch_size = FLAGS.batch_size
    max_epochs = FLAGS.max_epochs

    # create non-existing directories
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    if not train_data_path.exists():
        train_data_path.mkdir(parents=True, exist_ok=True)

    # load the data
    print('Loading the data...', end=' ')
    train, dev, test, text_field, label_field = load_data(percentage)
    embedding = nn.Embedding.from_pretrained(text_field.vocab.vectors)
    embedding.requires_grad = False
    print('Done!')

    print(f'SNLI dataset size (using {percentage * 100} % of the data):')
    print(f' Train: {len(train): {len(str(max(len(train), len(dev), len(test))))}} samples')
    print(f' Dev:   {len(dev): {len(str(max(len(train), len(dev), len(test))))}} samples')
    print(f' Test:  {len(test): {len(str(max(len(train), len(dev), len(test))))}} samples')

    # set the encoder
    print('Initializing the encoder...', end=' ')
    if model_type == 'Baseline':
        encoder = Baseline().to(DEVICE)
    elif model_type == 'LSTM':
        encoder = LSTM().to(DEVICE)
    elif model_type == 'BiLSTM':
        encoder = BiLSTM().to(DEVICE)
    elif model_type == 'MaxBiLSTM':
        encoder = MaxBiLSTM().to(DEVICE)
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
    ).to(DEVICE)
    print('Done!')
    print(f'Succesfully initialized the {model.__class__.__name__} model!')
    print_model_parameters(model)

    # set the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        params=model.params(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # initialize the results containers
    results = {
        'train_accucary': [],
        'train_loss': [],
        'dev_accuracy': [],
        'dev_loss': []
    }
    best_accuracy = 0.0

    for i in range(max_epochs):
        print(f'Epoch {i+1:0{len(str(max_epochs))}}/{max_epochs}:')

        epoch_results = {
            'train_accucary': [],
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
            batch_premises = batch.premise
            batch_hypotheses = batch.hypothesis
            batch_y = batch.label

            # forward pass
            pred_y = model.forward(batch_premises, batch_hypotheses)
            train_loss = criterion(pred_y, y_batch)

            # (re)set the optimizer gradient to 0
            optimizer.zero_grad()

            # backward pass
            train_loss.backward()
            optimizer.step()

            # compute and record the results
            epoch_results['train_accuracy'].append(batch_accuracy(batch_y, pred_y))
            epoch_results['train_loss'].append(train_loss.item())

        # compute and record the training results means for this epoch
        results['train_accuracy'].append(np.mean(epoch_results['train_accuracy']))
        results['train_loss'].append(np.mean(epoch_results['train_loss']))
        print(f" TRAIN accuracy: {results['train_accuracy'][-1]}, loss: {results['train_loss'][-1]}")

        # iterate over the dev data mini-batches for evaluation
        for batch in dev_iter:
            batch_premises = batch.premise
            batch_hypotheses = batch.hypothesis
            batch_y = batch.label

            # forward pass
            pred_y = model.forward(batch_premises, batch_hypotheses)
            dev_loss = criterion(pred_y, y_batch)

            # compute and record the results
            epoch_results['dev_accuracy'].append(batch_accuracy(batch_y, pred_y))
            epoch_results['dev_loss'].append(dev_loss.item())

        # compute and record the evaluation results means for this epoch
        results['dev_accuracy'].append(np.mean(epoch_results['dev_accuracy']))
        results['dev_loss'].append(np.mean(epoch_results['dev_loss']))
        print(f" DEV accuracy: {results['dev_accuracy'][-1]}, loss: {results['dev_loss'][-1]}")

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
            save_model(model)

        # terminate training if the learning rate becomes too low
        if learning_rate < MIN_LEARNING_RATE:
            print('Terminating the training process due to learning rate decay')
            break
    
    # save the results
    save_results(results)


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
    parser.add_argument('--model_type', type=str, default='a',
                        help='Model type (i.e: Baseline, LSTM, BiLSTM or MaxBiLSTM)')
    parser.add_argument('--checkpoint_path', type=str, default=CHECKPOINT_PATH_DEFAULT,
                        help='Path of directory to store training checkpoints')
    parser.add_argument('--train_data_path', type=str, default=TRAIN_DATA_PATH_DEFAULT,
                        help='Path of directory to store training results')
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
