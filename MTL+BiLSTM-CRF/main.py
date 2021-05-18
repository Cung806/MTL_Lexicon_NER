import torch
import torch.utils.data as Data
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch import nn
import torch
import torch.optim as optim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from collections import Counter
import numpy as np
import random
import os
from conlleval import evaluate
import pandas as pd

embeded_path = "cn_char_fastnlp_100d.txt"
train_path = "data/weibo/weiboNER_2nd_conll.train"
test_path = "data/weibo/weiboNER_2nd_conll.test"
dev_path = "data/weibo/weiboNER_2nd_conll.dev"

MAX_SEQ_LEN = 128
BATCH_SIZE = 12
embedding_size = 100
hidden_size = 200
lambda_value = 0.1

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(1024)

def build_corpus(data_dir):
    word_lists = []
    tag_lists = []
    bin_tag_lists = []  ## 是否存在实体的标签
    with open(data_dir, 'r', encoding='utf-8') as f:
        words = []
        tags = []
        for line in f:
            if line != '\n':
                word, tag = line.strip("\n").split(" ")
                word = word
                words.append(word)
                if tag=="":
                    tag = "O"
                tags.append(tag)
            else:
                word_lists.append(words)
                tag_lists.append(tags)
                if len(Counter(tags))==1:
                    bin_tag_lists.append(0)
                else:
                    bin_tag_lists.append(1)
                words = []
                tags = []
    return word_lists, tag_lists, bin_tag_lists

train_words_lists, train_tag_lists, train_bin_tag_lists = build_corpus(train_path)
test_words_lists, test_tag_lists, test_bin_tag_lists = build_corpus(test_path)
dev_words_lists, dev_tag_lists, dev_bin_tag_lists = build_corpus(dev_path)

def get_word_tag(train_words_lists, test_words_lists,dev_words_lists, train_tag_lists):
    words_lists = []
    words_lists.extend(train_words_lists)
    words_lists.extend(test_words_lists)
    words_lists.extend(dev_words_lists)
    words_map = {}
    for list in words_lists:
        for e in list:
            if e not in words_map:
                words_map[e] = len(words_map)+2
    words_map['<pad>'] = 0
    words_map['<unk>'] = 1
    
    id2word = {}
    for x in words_map:
        id2word[words_map[x]] = x
    
    tags_map = {}
    for li in train_tag_lists:
        for e in li:
            if e not in tags_map:
                tags_map[e] = len(tags_map)
    id2tag = {}
    for x in tags_map:
        id2tag[tags_map[x]] = x
    return words_map,id2word, tags_map, id2tag

## 得到单词和标签到id的映射
word2id, id2word, tag2id, id2tag = get_word_tag(train_words_lists, test_words_lists, dev_words_lists, train_tag_lists)

## 将单词映射到id
def tokenize2id(words_list, tag_list, word2id, tag2id):
    words2id_list = []
    tags2id_list = []
    for i in range(len(words_list)):
        words2id_list.append(list(map(lambda x: word2id[x], words_list[i] )))
        tags2id_list.append(list(map(lambda x: tag2id[x], tag_list[i])))
#     words2id_list.sort(key=lambda x: len(x))
#     tags2id_list.sort(key=lambda x: len(x))
    return  words2id_list, tags2id_list

train_words_id, train_tags_id = tokenize2id(train_words_lists, train_tag_lists, word2id, tag2id)
test_words_id, test_tags_id = tokenize2id(test_words_lists, test_tag_lists, word2id, tag2id)
dev_words_id, dev_tags_id = tokenize2id(dev_words_lists, dev_tag_lists, word2id, tag2id)

train_df = pd.DataFrame()
test_df = pd.DataFrame()
dev_df = pd.DataFrame()
train_df['words_id'], train_df['tags_id'], train_df['bin_tag']= train_words_id, train_tags_id, train_bin_tag_lists
test_df['words_id'], test_df['tags_id'], test_df['bin_tag']= test_words_id, test_tags_id, test_bin_tag_lists
dev_df['words_id'], dev_df['tags_id'], dev_df['bin_tag']= dev_words_id, dev_tags_id,dev_bin_tag_lists
train_df['seq_len'] = train_df['words_id'].apply(lambda x:len(x))
test_df['seq_len'] = test_df['words_id'].apply(lambda x:len(x))
dev_df['seq_len'] = dev_df['words_id'].apply(lambda x:len(x))
### 根据长度排序，可以实现先训练短的，再训练长的，如果不排序，就是随机
# train_df.sort_values(by='seq_len', ascending=True, inplace=True)
# test_df.sort_values(by='seq_len', ascending=True, inplace=True)
# dev_df.sort_values(by='seq_len', ascending=True, inplace=True)

class MyData(Data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx]['words_id'], self.data.loc[idx]['tags_id'], self.data.loc[idx]['bin_tag']

def collate_fn(data):
    x_data = []
    y_data = []
    y_bin_data = []
    for t in data:
        x_data.append(torch.tensor(t[0]))
        y_data.append(torch.tensor(t[1]))
        y_bin_data.append(t[2])
    x_data = rnn_utils.pad_sequence(x_data, batch_first=True, padding_value=0)
    y_data = rnn_utils.pad_sequence(y_data, batch_first=True, padding_value=0)
    masks = torch.zeros(len(data),len(x_data[0])).type(torch.ByteTensor)
    for i,x in enumerate(x_data):
        if 0 in x:
            zero_index = list(x).index(0)
            masks[i,:zero_index] = torch.ones(zero_index)
        else:
            masks[i] = torch.ones(len(x_data[0]))
    y_bin_data = torch.tensor(y_bin_data)
    return x_data, y_data, masks, y_bin_data

train_dataset = MyData(train_df)
test_dataset = MyData(test_df)
dev_dataset = MyData(dev_df)

train_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8,pin_memory=True)
test_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8,pin_memory=True)
dev_loader = Data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=8,pin_memory=True)

## 读取词向量
def pretrained_embedding(embed_path):
    wvmodel = KeyedVectors.load_word2vec_format(embeded_path)
    embed_size = len(wvmodel.get_vector(wvmodel.index2word[3]))
    vocab_size = len(word2id)
    weight = torch.empty(vocab_size, embed_size)
    nn.init.normal_(weight)   ## 01 正态分布初始化
    for wid in id2word:
        try:
            weight[wid] = torch.from_numpy(wvmodel.get_vector(id2word[wid]))
        except:
            continue
    return weight
weight = pretrained_embedding(embeded_path)

from torchcrf import CRF
## reference : https://github.com/kmkurn/pytorch-crf/issues/40
class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_tags, emb_size, hidden_size, weight=None,drop_out=0.5):
        super(BiLSTMCRF, self).__init__()
        if weight!=None:
            self.embedding = nn.Embedding.from_pretrained(weight)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(2*hidden_size, num_tags)
        self.bin_fc = nn.Linear(2*hidden_size, 2)   ### 二分类全连接
        self.crf = CRF(num_tags,batch_first =True)

        
    def neglikelihood(self, sentence, batch_y, masks):
        feats,_ = self.get_lstm_features(sentence)
        crf_loss = self.crf(feats, batch_y, mask = masks)
        return -crf_loss
        
    def get_lstm_features(self, sentence):
        emb = self.embedding(sentence)  # [B, L, emb_size]
        lstm_out, _ = self.bilstm(emb)
        scores = self.fc(lstm_out)
        scores = self.dropout(scores)
        return scores, lstm_out
        
    def forward(self, sentence, masks):
        scores, lstm_out = self.get_lstm_features(sentence)
        
#         bin_hidden = torch.cat((h_n[-1,:,:],h_n[-2,:,:]),dim= 1)   ## 二分类loss，
#         bin_scores = torch.min(lstm_out, dim =1).values
        bin_scores = torch.mean(lstm_out, dim =1)
        bin_scores = self.bin_fc(bin_scores)
        bin_scores = self.dropout(bin_scores)
        crf_out = self.crf.decode(scores, mask =masks)   ## CRF loss
        
        return crf_out, bin_scores

def ner_evaluate(model, data_loader):
    pred_all = []
    target_all = []
    model.eval()
    with torch.no_grad():
        for step, (batch_x, batch_y, mask, batch_bin_y) in enumerate(data_loader):
            batch_x, batch_y, mask,batch_bin_y = batch_x.to(device),batch_y.to(device), mask.to(device),batch_bin_y.to(device)
            pred, _ = model(batch_x, mask)
            pred_batch = []
            target_batch = []
            for i,p in enumerate(pred):
                pred_batch.extend(p)
                target_batch.extend(list(batch_y[i][:len(p)]))
            pred = list(map(lambda x:id2tag[x],pred_batch))
            target = list(map(lambda x:id2tag[int(x)], target_batch))
            pred_all.extend(pred)
            target_all.extend(target)
    PRF = evaluate(target_all, pred_all, verbose=True)
    return PRF

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print( " Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


"""
BI-LSTM+CRF  train msra ner_loss + bin_loss
"""
f1_list = []
vocab_size = len(word2id)
out_size = len(tag2id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
crfmodel = BiLSTMCRF(vocab_size,out_size, embedding_size, hidden_size, weight=weight)
crfmodel.to(device)
optimizer = optim.AdamW(crfmodel.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()
for epoch in range(100):
    train_loss = 0
    crfmodel.train()
    optimizer = lr_decay(optimizer, epoch, 0.05, 0.002)
    for step, (batch_x, batch_y, batch_masks, batch_bin_y) in enumerate(train_loader):
        batch_x, batch_y, batch_masks, batch_bin_y = batch_x.to(device),batch_y.to(device),batch_masks.to(device),batch_bin_y.to(device)
        optimizer.zero_grad()
        predictions, pred_bin = crfmodel(batch_x, batch_masks)
        loss_ner = crfmodel.neglikelihood(batch_x, batch_y, batch_masks)
        loss_bin = criterion(pred_bin,batch_bin_y)
        loss = loss_ner + lambda_value*loss_bin
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=crfmodel.parameters(), max_norm=5, norm_type=2)
        train_loss+=loss.item()
        optimizer.step()
    print("epoch:",epoch+1,"train_loss:",train_loss)
    precision, recall, f1 = ner_evaluate(crfmodel, test_loader)
    if epoch>12:
        f1_list.append(f1)
print("-----"*20)
print("mean_f1:",sum(f1_list)/len(f1_list))