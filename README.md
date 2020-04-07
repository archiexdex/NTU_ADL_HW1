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
bash ./extractive.sh ./data/valid.jsonl ./predict_tag.jsonl
bash ./seq2seq.sh    ./data/valid.jsonl ./predict_seq.jsonl
bash ./attention.sh  ./data/valid.jsonl ./predict_att.jsonl
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

```bash
# for preparing the seq_tag data
python3.6 src/preprocess_seq_tag.py datasets/seq_tag/ --train=1
# for extrative
python3.6 src/extrative_train.py

# for preparing the seq2seq data
python3.6 src/preprocess_seq2seq.py datasets/seq2seq/ --train=1
# for seq2seq
python3.6 src/train.py --isatt=0 --no=0
# for seq2seq + attention
python3.6 src/train.py --isatt=1 --no=1
```

![image](https://github.com/archiexdex/NTU_ADL_HW1/blob/master/imgs/histogram.png)
## How to plot the figures
```bash
python3.6 ./src/plot.py --raw_data="./data/valid.jsonl" --result_data="./predict_tag.jsonl"
# it will output histogram.png on ./ like ./histogram.png
```
