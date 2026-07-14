# Dialectometry Distances
This repo contains the code for the preprocessing, experiment and evaluation of models for the dialectometry distances project.

## Data Preprocessing
- `dataset_split.py` takes as input the `ud-treebanks-v2.17` and resamples it. The languages that are selected are those that contain
more than 320 sentences of 8 or more words. The split performed holds 100 sentences as the train set, 20 sentences as the dev set and 200 sentences
as the test set. From originally 339 treebanks, 231 treebanks meet this requirement. The newly sampled datasets are saved at `ud-treebanks-v2,17-samples`.

- `romanize_files.py` takes as input `ud-treebanks-v2.17-samples` and romanizes the text. The files are stored in
`ud-treebanks-v2.17-samples-romanized`.

## Model Training
- `Maria-Apostolidou-3.ipynb` is used to train Stanza models using multilingual word embeddings.
- `Maria-Apostolidou-no-embeds.ipynb` is used to train Stanza models using FastText embeddings.
Both these experiments create a `parsed_data` containing the output of a model on a particular language and `test_data` which contains the gold test data.

Example:
```
parsed_data
├── af_afribooms-on-af_afribooms-ud-test.conllu
├── af_afribooms-on-de_gsd-ud-test.conllu
├── af_afribooms-on-el_gud-ud-test.conllu
├── af_afribooms-on-it_isdt-ud-test.conllu
├── de_gsd-on-af_afribooms-ud-test.conllu
├── de_gsd-on-de_gsd-ud-test.conllu
├── de_gsd-on-el_gud-ud-test.conllu
├── de_gsd-on-it_isdt-ud-test.conllu
├── el_gud-on-af_afribooms-ud-test.conllu
├── el_gud-on-de_gsd-ud-test.conllu
├── el_gud-on-el_gud-ud-test.conllu
├── el_gud-on-it_isdt-ud-test.conllu
├── it_isdt-on-af_afribooms-ud-test.conllu
├── it_isdt-on-de_gsd-ud-test.conllu
├── it_isdt-on-el_gud-ud-test.conllu
└── it_isdt-on-it_isdt-ud-test.conllu
```
```
test_data
├── af_afribooms-ud-test.conllu
├── de_gsd-ud-test.conllu
├── el_gud-ud-test.conllu
└── it_isdt-ud-test.conllu
```
## Evaluation
- `evaluation.ipynb` performs the evaluation using ud-tools eval and stores the results on the folder `metrics`.
- `matrix.ipynb` creates a matrix that holds the metrics we have evaluated.
