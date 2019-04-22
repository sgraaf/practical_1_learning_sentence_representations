import argparse
from pathlib import Path
from time import time

import torch
from nltk import word_tokenize

from data import load_data
from encoders import LSTM, Baseline, BiLSTM, MaxBiLSTM
from model import InferSent
from utils import load_model_state, print_flags

# defaults
FLAGS = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT_DIR = Path.cwd().parent

ENCODER_TYPE_DEFAULT = 'Baseline'
ENCODERS_PATH_DEFAULT = ROOT_DIR / 'output' / 'models'


def infer():
    input_file = Path(FLAGS.input_file[0])
    output_file = input_file.parents / (input_file.name + '.out')
    encoder_type = FLAGS.encoder_type
    encoders_path = Path(FLAGS.encoders_path)

    # load the text_field
    print('Loading the data...', end=' ')
    _, _, _, text_field, _ = load_data()
    embedding = nn.Embedding.from_pretrained(text_field.vocab.vectors)
    embedding.requires_grad = False
    print('Done!')

    # load the encoder
    print('Loading the model...', end=' ')
    if encoder_type == 'Baseline':
        encoder = Baseline()
    elif encoder_type == 'LSTM':
        encoder = LSTM()
    elif encoder_type == 'BiLSTM':
        encoder = BiLSTM()
    elif encoder_type == 'MaxBiLSTM':
        encoder = MaxBiLSTM()

    model = InferSent(
        input_dim=4*encoder.output_dim,
        hidden_dim=512,
        output_dim=3,
        embedding=embedding,
        encoder=encoder
    )
    model.to(DEVICE)

    models = list(encoders_path.glob(
        f'InferSent_{encoder.__class__.__name__}_model.pt'))
    if len(models) > 0:
        load_model_state(models[0], model)
    else:
        raise ValueError(
            f'No models with the {encoder_type} encoder exist in the models directory!')
    print('Done!')
    print(f'Succesfully loaded the {model.__class__.__name__} model!')

    # process the input file
    print('Processing the input file...', end=' ')
    premises_hypotheses = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            premise, hypothesis = [line_part.strip()
                                   for line_part in line.split(';')]
            premises_hypotheses.append(
                {'premise': premise, 'hypothesis': hypothesis})
    print('Done!')

    with open(output_file, 'w') as f:
        f.write('premise;hypothesis;inference\n')
        
        for premise_hypothesis in premises_hypotheses:
            # tokenize the sentences
            premise = word_tokenize(premise_hypothesis['premise'])
            hypothesis = word_tokenize(premise_hypothesis['hypothesis'])

            # get their indices
            premise = torch.tensor([text_field.vocab.stoi[token] for token in premise])
            hypothesis = torch.tensor([text_field.vocab.stoi[token] for token in hypothesis])

            # predict entailment
            y_pred = model.forward(
                (premise.expand(1, -1).transpose(0, 1), len(premise)),
                (hypothesis.expand(1, -1).transpose(0, 1), len(hypothesis))
            )

            # determine the type of inference
            if y_pred.argmax().item() == 0:
                infer = 'entailment'
            elif y_pred.argmax().item() == 1:
                infer = 'contradiction'
            elif y_pred.argmax().item() == 2:
                infer = 'neutral'
            else:
                infer = 'error!'

            f.write(f'{premise_hypothesis['premise']};{premise_hypothesis['hypothesis']};{infer}\n')


def main():
    # print all flags
    print_flags(FLAGS)

    # start the timer
    start_time = time()

    # perform the inference
    infer()

    # end the timer
    end_time = time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f'Done inferring in {minutes}:{seconds} minutes.')


if __name__ == '__main__':
    # cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, metavar='path',
                        nargs=1, help='Path of the input file')
    parser.add_argument('--encoder_type', type=str, default=ENCODER_TYPE_DEFAULT,
                        help='Encoder type (i.e: Baseline, LSTM, BiLSTM or MaxBiLSTM)')
    parser.add_argument('--encoders_path', type=str, default=ENCODERS_PATH_DEFAULT,
                        help='Path of directory where the encoders are stored')

    FLAGS, unparsed = parser.parse_known_args()

    main()
