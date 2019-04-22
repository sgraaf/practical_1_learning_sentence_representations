# imports
import argparse
import sys
from pathlib import Path
from time import time

import sklearn
import torch
import numpy as np

from data import load_data, create_dictionary, get_wordvec
from encoders import LSTM, Baseline, BiLSTM, MaxBiLSTM
from utils import load_encoder_state, print_flags, save_eval_results

# defaults
FLAGS = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = Path.cwd().parent

ENCODER_TYPE_DEFAULT = 'Baseline'
ENCODERS_PATH_DEFAULT = ROOT_DIR / 'output' / 'models'
RESULTS_PATH_DEFAULT = ROOT_DIR / 'output' / 'eval'
BATCH_SIZE_DEFAULT = 64
TASKS_DEFAULT = 'MR, CR, MPQA'

# set PATHs
PATH_SENTEVAL = ROOT_DIR / 'src' / 'SentEval'
PATH_TO_DATA = PATH_SENTEVAL / 'data'
PATH_TO_VEC = PATH_TO_W2V = ROOT_DIR / 'src' / '.vector_cache' / 'glove.840B.300d.txt'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    _, params.word2id = create_dictionary(samples)
    # load glove/word2vec format 
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    # dimensionality of glove embeddings
    params.wvec_dim = 300
    return

def batcher(params, batch):
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        # the format of a sentence is a lists of words (tokenized and lowercased)
        for word in sent:
            if word in params.word_vec:
                # [number of words, embedding dimensionality]
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            # [number of words, embedding dimensionality]
            sentvec.append(vec)
        # average of word embeddings for sentence representation
        # [embedding dimansionality]
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)
    # [batch size, embedding dimensionality]
    embeddings = np.vstack(embeddings)
    return embeddings



def eval():
    encoder_type = FLAGS.encoder_type
    encoders_path = Path(FLAGS.encoders_path)
    results_path = Path(FLAGS.results_path)
    tasks = FLAGS.tasks
    tasks = [task.strip() for task in tasks.split(',')]

    # check if dirs exist
    if not encoders_path.exists():
        raise ValueError('The path to the encoders directory does not exist!')
    if not results_path.exists():
        results_path.mkdir(parents=True)

    # load the encoder
    print('Loading the encoder...', end=' ')
    if encoder_type == 'Baseline':
        encoder = Baseline()
    elif encoder_type == 'LSTM':
        encoder = LSTM()
    elif encoder_type == 'BiLSTM':
        encoder = BiLSTM()
    elif encoder_type == 'MaxBiLSTM':
        encoder = MaxBiLSTM()

    encoders = list(encoders_path.glob(f'InferSent_{encoder.__class__.__name__}_encoder.pt'))
    if len(encoders) > 0:
        load_encoder_state(encoders[0], encoder)
    else:
        raise ValueError(f'No encoders of type {encoder_type} exist in the encoders directory!')

    encoder.to(DEVICE)
    print('Done!')
    print(f'Succesfully loaded the {encoder.__class__.__name__} encoder!')

    # define senteval params
    print('Setting up SentEval...', end=' ')
    params_senteval = {
        'task_path': str(PATH_TO_DATA), 
        'usepytorch': True, 
        'kfold': 5
    }
    params_senteval['classifier'] = {
        'nhid': 0, 
        'optim': 'rmsprop', 
        'batch_size': 128,
        'tenacity': 3, 
        'epoch_size': 2
    }
    params_senteval['infersent'] = encoder.to(DEVICE)

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    print('Done!')

    print('Running SentEval...', end=' ')
    results = se.eval(tasks)
    print('Done!')

    print(results)
    
    save_eval_results(results_path, results, encoder, tasks)


def main():
    # print all flags
    print_flags(FLAGS)

    # start the timer
    start_time = time()

    # train the model
    eval()

    # end the timer
    end_time = time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f'Done evaluating in {minutes}:{seconds} minutes.')


if __name__ == '__main__':
    # cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', type=str, default=ENCODER_TYPE_DEFAULT,
                        help='Encoder type (i.e: Baseline, LSTM, BiLSTM or MaxBiLSTM)')
    parser.add_argument('--encoders_path', type=str, default=ENCODERS_PATH_DEFAULT,
                        help='Path of directory where the encoders are stored')
    parser.add_argument('--results_path', type=str, default=RESULTS_PATH_DEFAULT,
                        help='Path of directory to store results')
    parser.add_argument('--tasks', type=str, default=TASKS_DEFAULT,
                        help='The SentEval tasks to evaluate the model on (i.e. MR, QR, SUBJ, etc.)')
    FLAGS, unparsed = parser.parse_known_args()

    main()
