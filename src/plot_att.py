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
parser.add_argument("--result_data", type=str, default="./predict_att.jsonl")
parser.add_argument("--result_data", type=str, default="./predict_att.jsonl")


args = parser.parse_args()

with open(args.result_data) as f:
    predicts = [json.loads(valid) for valid in f]

with open(args.raw_data) as f:
    raws = [json.loads(valid) for valid in f]



def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("attention.png")