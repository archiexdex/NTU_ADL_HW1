import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import pickle
import argparse
from tqdm import tqdm

st_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str, default="./data/valid.jsonl")
parser.add_argument("--result_data", type=str, default="./predict_tag.jsonl")

args = parser.parse_args()

with open(args.result_data) as f:
    predicts = [json.loads(valid) for valid in f]

with open(args.raw_data) as f:
    raws = [json.loads(valid) for valid in f]


data = []
for raw, pred in zip(raws, predicts):
    total_len = len(raw["sent_bounds"])
    sents = np.array(pred["predict_sentence_index"])
    sents = sents / total_len
    sents.tolist()
    data.extend(sents)

data = np.array(data)

bins = [i*0.01 for i in range(0,100)]
plt.hist(data, bins=bins, linewidth=1,
                          edgecolor='#EFB28C',
                          color='#EED19C')
# plt.hist(data)
plt.savefig("histogram.png")
