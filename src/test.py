import os, sys, shutil
import pickle
import numpy as np
import json
from datetime import datetime
import argparse
from argparse import Namespace
from tqdm import tqdm
from utils import Tokenizer, Embedding

import torch
from torch.utils.data import DataLoader
from dataset import SeqTaggingDataset
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("test_data_path", type=str, help="the path of test data file")
parser.add_argument("store_path", type=str, help="the path of saving file")
parser.add_argument("--isatt", type=int, default=1)
parser.add_argument("--no", type=int, default=0)
parser.add_argument("--load_model_path", type=str, default="./datasets/seq2seq/ckpt/model_0")

args = parser.parse_args()

isatt = True if args.isatt > 0 else False 

st_time = datetime.now()

hparams = Namespace(**{
    'no': args.no,
    "model": "seq2seq",
    'mode': "test",

    'embedding_path': "datasets/seq2seq/embedding.pkl",
    'embed_size': 300,
    'max_summary': 80,

    'valid_rawdata_path': "data/valid.jsonl",
    'test_rawdata_path': "data/test.jsonl",
    'id2token_path': "datasets/seq2seq/id2token.json",

    'train_dataset_path': "datasets/seq2seq/train.pkl",
    'valid_dataset_path': "datasets/seq2seq/valid.pkl",
    'test_dataset_path': "datasets/seq2seq/test.pkl",
    'ignore_idx': -100,

    'total_epoch': 100,
    'batch_size': 32,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'learning_rate': 1e-3,
    'early_stop_epoch': 5,
    
    'pos_weight': 1,
    'rnn_hidden_size': 256,
    'teacher_forcing_ratio': 0,
    'n_layers': 1,
    'dropout': 0.5,
    'attention': isatt,
    # 'attention': False,
    'isbidir': True,

    'ckpt_dir': "datasets/seq2seq/ckpt",
    'load_model_path': args.load_model_path, #"datasets/seq2seq/ckpt/model_0",
    'predict_path': "datasets/seq2seq/predict.jsonl"
})

# with open(hparams.test_dataset_path, 'rb') as fp:
with open(args.test_data_path, 'rb') as fp:
    dataset = pickle.load(fp)


tokenizer = Tokenizer(lower=True)
with open(hparams.embedding_path, 'rb') as f:
    embedding = pickle.load(f)
    
tokenizer.set_vocab(embedding.vocab)

dataLoader = DataLoader(dataset, hparams.batch_size, shuffle=False,  collate_fn=dataset.collate_fn)

model = Model(hparams)
model.load(hparams.load_model_path)

st_time = datetime.now()
predicts = model.eval(dataLoader)

results = []
for predict in predicts:
    res = {
        "id": predict["id"],
        "predict": tokenizer.decode(predict["predict"])
    }
    results.append(res)

with open(args.store_path, 'w') as fp:
    for result in results:
        s = json.dumps(result)
        fp.write(f"{s}\n")

print(f"Cost time: {datetime.now()-st_time}")
