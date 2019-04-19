# practical_1_learning_sentence_representations
This repository holds the assignment description, relevant papers, code and output related to this NLP assignment, made in the context of the Statistical Methods for Natural Language Semantics course, taught in the Spring 2019 semester at the University of Amsterdam.

## Installation
```
git clone https://github.com/sgraaf/practical_1_learning_sentence_representations/
cd practical_1_learning_sentence_representations
conda create -f environment.yml
activate smnls
```

## Execution
For training:
```
python train.py <model_type> <model_name> <checkpoint_path> <train_data_path>
```

For evaluation:
```
python eval.py <checkpoint_path> <eval_data_path>
```

For inference:
```
python infer.py <checkpoint_path> <input_file> <output_file>
```
