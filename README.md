# ADL HW1

## How TA Create Testing Environment
```bash
bash install_packages.sh
```

## Set Environment
```bash
# for python environment
bash ./install_packages.sh
# for download model
bash ./download.sh
```

## Quick Start for Testing
```bash
bash ./extractive.sh ./data/test.jsonl ./data/predict.jsonl
bash ./seq2seq.sh    ./data/test.jsonl ./data/predict.jsonl
bash ./attention.sh  ./data/test.jsonl ./data/predict.jsonl
```

## How to train model
Here is the example structure of data
* datasets/
* * seq_tag/
* * * train.pickle
* * * valid.pickle
* * * test.pickle
* * * embedding.pickle
* * * config.json
* * seq2seq/
* * * train.pickle
* * * valid.pickle
* * * test.pickle
* * * embedding.pickle
* * * config.json
* * * id2token.json

```bash
# for extrative
python3.6 src/extrative_train.py
# for seq2seq
python3.6 src/train.py --isatt=0
# for seq2seq + attention
python3.6 src/train.py --isatt=1
```

## How to plot the figures
```bash
# ${1}
python3.6 src/
```