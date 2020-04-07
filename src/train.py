import os, sys, shutil
import pickle
import numpy as np
import json
from datetime import datetime
import argparse
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataset import SeqTaggingDataset
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--isatt", type=int, default=1)
parser.add_argument("--no", type=int, default=0)
parser.add_argument("--load_model_path", type=str, default="./datasets/seq2seq/ckpt/model_0")

args = parser.parse_args()

isatt = True if args.isatt > 0 else False 

st_time = datetime.now()

hparams = Namespace(**{
    'no': args.no,
    "model": "seq2seq",
    'mode': "train",

    'embedding_path': "datasets/seq2seq/embedding.pkl",
    'embed_size': 300,
    'max_summary': 80,

    'valid_rawdata_path': "data/valid.jsonl",
    'test_rawdata_path': "data/test.jsonl",

    'train_dataset_path': "datasets/seq2seq/train.pkl",
    'valid_dataset_path': "datasets/seq2seq/valid.pkl",
    'test_dataset_path': "datasets/seq2seq/test.pkl",
    'ignore_idx': -100,

    'total_epoch': 100,
    'batch_size': 64,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'learning_rate': 1e-3,
    'early_stop_epoch': 5,
    
    'pos_weight': 1,
    'rnn_hidden_size': 256,
    'teacher_forcing_ratio': 1,
    'n_layers': 1,
    'dropout': 0.5,
    'attention': isatt,
    'isbidir': True,

    'ckpt_dir': "datasets/seq2seq/ckpt",
    'load_model_path': None, #"datasets/seq2seq/ckpt/model_0"
    'predict_path': "datasets/seq2seq/predict.jsonl"
})

if not os.path.exists(hparams.ckpt_dir):
    os.mkdir(hparams.ckpt_dir)

with open(hparams.train_dataset_path, 'rb') as fp:
    train_dataset = pickle.load(fp)
with open(hparams.valid_dataset_path, 'rb') as fp:
    valid_dataset = pickle.load(fp)

train_dataLoader = DataLoader(train_dataset, hparams.batch_size, shuffle=True,  collate_fn=train_dataset.collate_fn)
valid_dataLoader = DataLoader(valid_dataset, hparams.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)

# Main
model = Model(hparams)
best_valid = 1e9
early_cnt = 0

for epoch in range(hparams.total_epoch):
    
    train_loss = model.train(train_dataLoader)
    valid_loss = model.valid(valid_dataLoader)

    if best_valid > valid_loss:
        best_valid = valid_loss
        early_cnt = 0
        model.save(hparams.no)
    else:
        early_cnt += 1
        if early_cnt > hparams.early_stop_epoch:
            break
    
    print(f"epoch: {epoch}")
    print(f"train loss: {train_loss}")
    print(f"valid loss: {valid_loss}")
print(f"Cost time: {datetime.now()-st_time}")