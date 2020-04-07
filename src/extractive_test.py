import os, sys, shutil
import pickle
import numpy as np
import json
from datetime import datetime
from argparse import Namespace
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataset import SeqTaggingDataset
from model import Model

st_time = datetime.now()

hparams = Namespace(**{
    'no': 0,
    'model': "seq_tag",
    'mode': "test",

    'datapath': "dataset/seq_tag",
    'embedding_path': "datasets/seq_tag/embedding.pkl",
    # 'embedding_path': "embedding.pkl",
    'embed_size': 300,

    'valid_rawdata_path': "data/valid.jsonl",
    'test_rawdata_path': "data/test.jsonl",

    'train_dataset_path': "datasets/seq_tag/train.pkl",
    'valid_dataset_path': "datasets/seq_tag/valid.pkl",
    'test_dataset_path': "datasets/seq_tag/test.pkl",
    # 'train_dataset_path': "train.pkl",
    # 'valid_dataset_path': "valid.pkl",
    # 'test_dataset_path': "test.pkl",
    'ignore_idx': -100,

    'total_epoch': 100,
    'batch_size': 64,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'learning_rate': 1e-2,
    'early_stop_epoch': 5,
    
    'pos_weight': 1,
    'rnn_hidden_size': 512,
    'teacher_forcing_ratio': 1,
    'n_layers': 2,
    'dropout': 0.5,
    'attention': False,
    'isbidir': True,

    # 'ckpt_dir': "ckpt",
    'ckpt_dir': "datasets/seq_tag/ckpt",
    'load_model_path': "datasets/seq_tag/ckpt/model_0",
    'predict_path': "datasets/seq_tag/predict.jsonl"
})

# with open(hparams.test_dataset_path, 'rb') as fp:
with open(hparams.valid_dataset_path, 'rb') as fp:
    dataset = pickle.load(fp)

# with open(hparams.test_rawdata_path) as f:
with open(hparams.valid_rawdata_path) as f:
    ref = [json.loads(line) for line in f]
    ref = {a['id']: a["sent_bounds"] for a in ref}


dataLoader = DataLoader(dataset, hparams.batch_size, shuffle=False,  collate_fn=dataset.collate_fn)

model = Model(hparams)
model.load(hparams.load_model_path)

st_time = datetime.now()
ptr_results = model.predict(dataLoader)
results = []
for ptr in ptr_results:
    results.append({
        "id": ptr["id"],
        "predict_sentence_index": ref[ptr["id"]][ptr["ptr"]]
    })

with open(hparams.predict_path, 'w') as fp:
    for result in results:
        s = json.dumps(result)
        fp.write(f"{s}\n")

print(f"Cost time: {datetime.now()-st_time}")