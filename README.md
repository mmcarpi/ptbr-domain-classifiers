# Overview

Code used to fine-tune encoders for Brazilian Portuguese for discourse domain classification.

# Files and directories
## Requirements
- [requirements.txt](./requirements.txt), pip freeze dump of the requirements necessary to run all the scripts and programs in this repository.

## Directories
- [Config](./Config), folder with the configuration files to train the models
- [Models](./Models), folder with the training results
- [pbs_scripts](./pbs_scripts), folder with PBS scripts for hyperparameter search and and model training

## Dataset creation
- [caroldb.py](./caroldb.py), script for downloading [Carol (D + B)](https://github.com/marianasturzeneker/SubcorporaCarolina)
- [sentences.py](./sentences.py), script for splitting Carol (D + B) texts into sentences
- [train-test-split.py](./train-test-split.py), script for further splitting the sentences into train, test and hyperparameter-search datasets

## Model training and evaluation
### Baseline
- [baseline.py](./baseline.py), script for training and evaluating naive Bayes and SVM classifiers using [scikit-learn](https://scikit-learn.org/stable/)
### Transformers
- [config.py](./config.py), load and write model configuration files
- [dataloader.py](./dataloader.py), custom dataset and dataloader used for model training
- [hyperparameter-search.py](./hyperparameter-search.py), script for model hyperparameter search with [Optuna](https://optuna.org/)
- [dist.py](./dist.py), script for distributed model training
- [eval.py](./eval.py), script for evaluationg model on test dataset

## Utilities
- [plots_and_tables.ipynb](./plots_and_tables.ipynb), notebook for generating graphs and tables for the paper
- [clean.sh](./clean.sh), script for removing zip and csv files generetade by [plots_and_tables.ipynb](plots_and_tables.ipynb)
- [upload.py](./upload.py), script for uploading specific model checkpoint to [Hugging Face](./https://huggingface.co/)
- [upload_folder.py](./upload_folder.py), script for uploading folder to the Hugging Face


# Dataset
The dataset used for training this models is available [here](https://huggingface.co/datasets/carolina-c4ai/carol-domain-sents).

# Models and checkpoints
The models can be found [here](https://huggingface.co/collections/carolina-c4ai/ptbr-discourse-domain-classifiers-67c0d76167e68677cf30285b).
