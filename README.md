# practical_1_learning_sentence_representations
This repository holds the assignment description, relevant papers, code and output related to this NLP assignment, made in the context of the [Statistical Methods for Natural Language Semantics course](https://cl-illc.github.io/semantics/), taught in the Spring 2019 semester at the University of Amsterdam.

## Installation
```
git clone https://github.com/sgraaf/practical_1_learning_sentence_representations/
cd practical_1_learning_sentence_representations
conda create -f environment.yml
conda activate smnls
python
>>> import nltk
>>> nltk.download('punkt')
```

## Execution
### Training
```
python src/train.py [OPTIONS]
```
#### OPTIONS
```
--model_type MODEL_TYPE   Model type (i.e: Baseline, LSTM, BiLSTM or MaxBiLSTM)
--checkpoint_path PATH    Path of directory to store checkpoints
--models_path PATH        Path of directory to store models
--results_path PATH       Path of directory to store results
--percentage FLOAT        Percentage of data to be used (for training, evaluating, testing, etc.)
--learning_rate FLOAT     Learning rate
--weight_decay FLOAT      Weight decay of the learning rate
--batch_size INT          Batch size
--max_epochs INT          Max number of epochs for training
```

### Evaluation
```
python src/eval.py <checkpoint_path> <eval_data_path>
```

### Inference:
```
python src/infer.py <checkpoint_path> <input_file> <output_file>
```
