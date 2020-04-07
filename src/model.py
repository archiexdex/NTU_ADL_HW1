import numpy as np
import random
import pickle
from tqdm import tqdm
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Model:
    def __init__(self, hparams):
        self.hparams = hparams
        self.device = hparams.device
        if hparams.model == "seq_tag":
            self.net = SeqTagger(hparams)
        elif hparams.model == "seq2seq":
            self.net = Seq2Seq(hparams)
        
        self.net.to(self.device)
        if hparams.mode == "train":
            # self.net.apply(self.init_weights)
            if hparams.model == "seq_tag":
                self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(hparams.pos_weight)).to(self.device)
            elif hparams.model == "seq2seq":
                self.criterion = nn.CrossEntropyLoss(ignore_index = 0).to(self.device)
            self.optim = optim.Adam(self.net.parameters(), lr=hparams.learning_rate)
    
    def train(self, dataloader):
        self.net.train()
        mean_loss = 0
        for data in tqdm(dataloader):
            if self.hparams.model == "seq_tag":
                text, label = self._unpack_batch(data)
                predict = self.net(text).squeeze(-1)
                self.optim.zero_grad()
                loss = self._get_loss(predict, label)
                loss.backward()
                self.optim.step()
            else:
                text, summary = self._unpack_batch(data)
                self.optim.zero_grad()
                predict, _ = self.net(text, summary, self.hparams.teacher_forcing_ratio)
                predict = predict[:, 1:].reshape(-1, predict.size(2))
                summary = summary[:, 1:].reshape(-1)
                loss = self._get_loss(predict, summary)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optim.step()
            
            mean_loss += loss.item() / len(data)
        
        mean_loss /= len(dataloader)
        return mean_loss

    def valid(self, dataloader):
        self.net.eval()
        mean_loss = 0
        for data in tqdm(dataloader):
            if self.hparams.model == "seq_tag":
                text, label = self._unpack_batch(data)
                predict = self.net(text).squeeze(-1)
                loss = self._get_loss(predict, label)
            else:
                text, summary = self._unpack_batch(data)
                predict, _ = self.net(text, summary, 0)
                predict = predict[:, 1:].reshape(-1, predict.size(2))
                summary = summary[:, 1:].reshape(-1)
                loss = self._get_loss(predict, summary)
            
            mean_loss += loss.item() / len(data)
        
        mean_loss /= len(dataloader)
        return mean_loss

    def eval(self, dataloader):
        self.net.eval()
        ans = []
        for data in tqdm(dataloader):
            no = data["id"]
            text = data["text"].to(self.device)
            _, predicts = self.net(text)

            predicts = predicts.cpu()
            for idx, predict in enumerate(predicts):
                ptr = -1
                for jdx, w in enumerate(predict):
                    if w == 2:
                        ptr = jdx
                        break
                ans.append({
                    "id": no[idx],
                    "predict": predict[:ptr+1]
                })
        return ans            

    def predict(self, dataloader):
        self.net.eval()
        bound_ans = []
        for data in tqdm(dataloader):
            no = data["id"]
            bounds = data["sent_range"]
            text = data["text"].to(self.device)
            predict = self.net(text).squeeze(-1)
            
            for idx, bound_list in enumerate(bounds):
                bounds_mean_list = []
                for bound in bound_list:
                    predict_mean = torch.mean(predict[idx][bound[0]:bound[1]])
                    bounds_mean_list.append(predict_mean)
                ans = np.argmax(bounds_mean_list)
                bound_ans.append({
                    "id": no[idx],
                    "ptr": ans
                })
                # bound_ans.append({
                #     "id": no[idx],
                #     "predict_sentence_index": bound_list[ans]
                # })
        return bound_ans

    def _get_loss(self, predict, label):
        if self.hparams.model == "seq_tag":
            return self.criterion(predict[label > -100], label[label > -100])
        else:
            return self.criterion(predict, label)

    def _unpack_batch(self, batch) -> Tuple[torch.tensor, torch.tensor]:
        if self.device:
            if self.hparams.model == "seq_tag":
                return batch['text'].cuda(), batch['label'].float().cuda()
            else:
                return batch['text'].cuda(), batch['summary'].cuda()
        else:
            if self.hparams.model == "seq_tag":
                return batch['text'], batch['label'].float()
            else:
                return batch['text'], batch['summary']

    def save(self, epoch):
        torch.save(self.net.state_dict(), f'{self.hparams.ckpt_dir}/model_{epoch}.ckpt')
    
    def load(self, path):
        self.net.load_state_dict(torch.load(f'{path}.ckpt'))

    def init_weights(self, m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

class Encoder(nn.Module):
    def __init__(self, embedding_path, embed_size, hidden_size, n_layers, dropout, isbidir) -> None:
        super(Encoder, self).__init__()
        
        with open(embedding_path, 'rb') as f:
          embedding = pickle.load(f)
        embedding_weight = embedding.vectors

        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

        if n_layers == 1:
            self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=isbidir)
        else:
            self.rnn = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, batch_first=True, bidirectional=isbidir)
        

    def forward(self, idxs) -> Tuple[torch.tensor, torch.tensor]:
        embed = self.embedding(idxs)
        # embed = self.dropout(embed)
        output, state = self.rnn(embed)
        return output, state

class SeqTagger(nn.Module):
    def __init__(self, hparams) -> None:
        super(SeqTagger, self).__init__()

        self.model = hparams.model
        self.encoder = Encoder(hparams.embedding_path, hparams.embed_size, hparams.rnn_hidden_size, hparams.n_layers, hparams.dropout, hparams.isbidir)
        if hparams.isbidir:
            self.fc = nn.Linear(hparams.rnn_hidden_size * 2, 1)
        else:
            self.fc = nn.Linear(hparams.rnn_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, idxs) -> torch.tensor:
        logits, states = self.encoder(idxs)
        logits = self.fc(logits)
        logits = self.sigmoid(logits)
        return logits    

class Decoder(nn.Module):
    def __init__(self, embedding_path, embed_size, hidden_size, n_layers, dropout, isatt):
        super().__init__()

        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.voc_size = embedding_weight.shape[0]
        self.isatt = isatt

        if isatt:
            self.attention = Attention(hidden_size)
        self.input_size = embed_size + hidden_size * 2 if isatt else embed_size

        if n_layers == 1:
            self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(self.input_size, self.hidden_size, n_layers, dropout=dropout, batch_first=True)
        if isatt:
            self.fc1 = nn.Linear(self.input_size + self.hidden_size, self.voc_size)
        else:
            self.fc1 = nn.Linear(self.hidden_size, self.voc_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idxs, hidden, encoder_outputs):
        # input  = [batch size, vocab size]
        # hidden = [batch size, n layers * directions, hid dim]
        # encoder_outputs = [batch size, src len, hid dim]
        idxs = idxs.unsqueeze(-1)

        embed = self.dropout(self.embedding(idxs))
        
        if self.isatt:
            # attn = [batch size, src len]
            attn = self.attention(encoder_outputs, hidden)
            attn = attn.unsqueeze(1)
            weight = torch.bmm(attn, encoder_outputs)
            rnn_input = torch.cat((embed, weight), dim=2)
            output, hidden = self.rnn(rnn_input, hidden)
            output = torch.cat((output, weight, embed), dim=-1)
        else:
            output, hidden = self.rnn(embed, hidden)
            # output = [batch size, 1, hid dim]
            # hidden = [num_layers, batch size, hid dim]

        # 將 RNN 的輸出轉為每個詞出現的機率
        prediction = self.fc1(output.squeeze(1))
        # output = self.fc2(output)
        # prediction = self.fc3(output)
        # prediction = [batch size, vocab size]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.encoder = Encoder(hparams.embedding_path, hparams.embed_size, hparams.rnn_hidden_size, hparams.n_layers, hparams.dropout, hparams.isbidir)
        self.decoder = Decoder(hparams.embedding_path, hparams.embed_size, hparams.rnn_hidden_size, hparams.n_layers, hparams.dropout, hparams.attention)
        self.device = hparams.device
        self.isatt = hparams.attention
        self.enfc = nn.Linear(hparams.rnn_hidden_size * 2, hparams.rnn_hidden_size)
        self.mode = hparams.mode
        self.max_summary = hparams.max_summary
            
    def forward(self, x, target=None, teacher_forcing_ratio=0):
        # x      = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = x.shape[0]
        max_len    = self.max_summary
        max_target = 0 if target is None else target.shape[1]
        voc_size   = self.decoder.voc_size

        outputs = torch.zeros(batch_size, max_len, voc_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(x)

        # hidden =  [num_layers * directions, batch size, hid dim] -> [num_layers, directions, batch size, hid dim] -> [num_layers, batch size, hid dim * 2]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        hidden = torch.tanh(self.enfc(hidden))
        
        if self.mode == "train":
            x = target[:, 0] 
        else:
            # x = [<s>]
            x = torch.ones(batch_size).long().to(self.device)
        
        preds = []
        max_iter = max_target if max_target > 0 else max_len

        for t in range(1, max_iter):
            output, hidden = self.decoder(x, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() <= teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            x = top1 if teacher_force == 0 else target[:, t]
            preds.append(top1.unsqueeze(1))

        # append </s>
        EOS = torch.full_like(x, 2).long().unsqueeze(1).to(self.device)
        preds.append(EOS)
        preds = torch.cat(preds, 1)

        if max_target > 0:
            outputs = outputs[:, :max_target]
        
        return outputs, preds

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        
        self.attn = nn.Linear((hidden_size * 2) + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, encoder_outputs, hidden):
        
        #hidden = [num_layers, batch size, hid dim]
        #encoder_outputs = [batch size, src len, hid dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        if hidden.shape[0] < 2:
            hidden = hidden[0]
        else:
            hidden = hidden[0] + hidden[1]
        #hidden = [num_layers, batch size, hid dim] -> [batch size, hid dim]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        #repeat hidden state src_len times
        #hidden = [batch size, src len, hid dim]
        #encoder_outputs = [batch size, src len, hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=-1))) 
        #energy = [batch size, src len, hid dim]

        attention = self.v(energy).squeeze(-1)
        #attention= [batch size, src len]

        return F.softmax(attention, dim=-1)
