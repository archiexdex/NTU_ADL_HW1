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

st_time = datetime.now()

hparams = Namespace(**{
    'no': 0,
    "model": "seq2seq",
    'mode': "test",

    'embedding_path': "datasets/seq2seq/embedding.pkl",
    'embed_size': 300,

    'valid_rawdata_path': "data/valid.jsonl",
    'test_rawdata_path': "data/test.jsonl",
    'id2token_path': "datasets/seq2seq/id2token.json",

    'train_dataset_path': "datasets/seq2seq/train.pkl",
    'valid_dataset_path': "datasets/seq2seq/valid.pkl",
    'test_dataset_path': "datasets/seq2seq/test.pkl",
    'ignore_idx': -100,

    'total_epoch': 100,
    'batch_size': 2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'learning_rate': 1e-3,
    'early_stop_epoch': 5,
    
    'pos_weight': 1,
    'rnn_hidden_size': 128,
    'teacher_forcing_ratio': 0,
    'n_layers': 1,
    'dropout': 0.5,
    # 'attention': True,
    'attention': False,
    'isbidir': True,

    'ckpt_dir': "datasets/seq2seq/ckpt",
    'load_model_path': "datasets/seq2seq/ckpt/model_0",
    'predict_path': "datasets/seq2seq/predict.jsonl"
})

with open(hparams.id2token_path, "r") as fp:
    id2token = json.load(fp)

# with open(hparams.test_dataset_path, 'rb') as fp:
with open(hparams.valid_dataset_path, 'rb') as fp:
    dataset = pickle.load(fp)


dataLoader = DataLoader(dataset, hparams.batch_size, shuffle=False,  collate_fn=dataset.collate_fn)

# for data in tqdm(dataLoader):
#     for key in data:
#         print(key)
#         print(data[key])
#         input("##")

model = Model(hparams)
model.load(hparams.load_model_path)

st_time = datetime.now()
predicts = model.eval(dataLoader)
# results = []
# for ptr in ptr_results:
#     results.append({
#         "id": ptr["id"],
#         "predict_sentence_index": ref[ptr["id"]][ptr["ptr"]]
#     })

with open(hparams.predict_path, 'w') as fp:
    for result in results:
        s = json.dumps(result)
        fp.write(f"{s}\n")

print(f"Cost time: {datetime.now()-st_time}")