# practical_1_learning_sentence_representations
This repository holds the assignment description, relevant papers, code and output related to this NLP assignment, made in the context of the [Statistical Methods for Natural Language Semantics course](https://cl-illc.github.io/semantics/), taught in the Spring 2019 semester at the University of Amsterdam.

## Installation
```
git clone https://github.com/sgraaf/practical_1_learning_sentence_representations/
cd practical_1_learning_sentence_representations/src
conda create -f environment.yml
conda activate smnls
python
>>> import nltk
>>> nltk.download('punkt')
```

## Execution
### Training
```
python train.py [OPTIONS]
```
#### OPTIONS
```
--encoder_type ENCODER_TYPE   Encoder type (i.e: Baseline, LSTM, BiLSTM or MaxBiLSTM)
--checkpoint_path PATH        Path of directory to store (and load) checkpoints
--models_path PATH            Path of directory to store models
--results_path PATH           Path of directory to store results
--percentage FLOAT            Percentage of data to be used (for training, evaluating, testing, etc.)
--learning_rate FLOAT         Learning rate
--weight_decay FLOAT          Weight decay of the learning rate
--batch_size INT              Batch size
--max_epochs INT              Max number of epochs for training
```

### Evaluation
```
python eval.py [OPTIONS]
```
#### OPTIONS
```
--encoder_type ENCODER_TYPE   Encoder type (i.e: Baseline, LSTM, BiLSTM or MaxBiLSTM)
--encoders_path PATH          Path of directory where the encoders are stored
--results_path PATH           Path of directory to store results
--tasks STR[, STR...]         The SentEval tasks to evaluate the model on (i.e. MR, QR, SUBJ, etc.)')
```

### Inference:
```
python src/infer.py INPUT_FILE [OPTIONS]
```
#### OPTIONS
```
--encoder_type ENCODER_TYPE   Encoder type (i.e: Baseline, LSTM, BiLSTM or MaxBiLSTM)
--models_path PATH            Path of directory where the moodels are stored
```
