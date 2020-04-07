# ADL HW1

## How to set Environment
```bash
# for python environment
bash ./install_packages.sh
# for download model, config and embedding 
bash ./download.sh
```

## Quick Start for Testing
```bash
bash ./extractive.sh ./data/valid.jsonl ./data/predict_tag.jsonl
bash ./seq2seq.sh    ./data/valid.jsonl ./data/predict_seq.jsonl
bash ./attention.sh  ./data/valid.jsonl ./data/predict_att.jsonl
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
python3.6 src/train.py --isatt=0 --no=0
# for seq2seq + attention
python3.6 src/train.py --isatt=1 --no=1
```

## How to plot the figures
```bash
python3.6 ./src/plot.py --raw_data="./data/valid.jsonl" --result_data="./predict_tag.jsonl"
# it will output histogram.png on ./histogram.png
```