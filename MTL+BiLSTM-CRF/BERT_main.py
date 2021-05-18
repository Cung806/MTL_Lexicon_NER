from transformers import AutoTokenizer,BertModel,BertPreTrainedModel,BertConfig,BertForTokenClassification
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as reset
from transformers import AdamW, get_linear_schedule_with_warmup
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from collections import Counter
from conlleval import evaluate
import torch.optim as optim
from torchcrf import CRF
from torch.optim.lr_scheduler import LambdaLR

BATCH_SIZE = 12
MAX_LEN = 128
bert_path = "data/bert_model/bert-base-chinese/"

train_path = "data/weibo/weiboNER_2nd_conll.train"
test_path = "data/weibo/weiboNER_2nd_conll.test"
dev_path = "data/weibo/weiboNER_2nd_conll.dev"

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
                word = word[0]
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

tags_set = set()
for sen in train_tag_lists:
    for x in sen:
        tags_set.add(x)
tag2idx = {
        "O": 0,
#         "B-<start>": 1,
#         "B-<end>": 2
    }
idx2tag = {
    0:"O",
#     1:"B-<start>",
#     2:"B-<end>"
}
for tag in tags_set:
    if tag not in tag2idx:
        tag2idx[tag] = len(tag2idx)
        idx2tag[len(idx2tag)] = tag

class CustomDataset(Data.Dataset):
    def __init__(self, data, maxlen, with_labels=True, model_name='bert-base-chinese'):
        self.data = data  # pandas dataframe

        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data['X'])

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = self.data["X"][index]
        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair=self.tokenizer.encode_plus(
            sent,
            max_length=self.maxlen,
            add_special_tokens=True,   # 'Add [SEP] & [CLS]'
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,  # Reurns array of 0's & 1's to distinguish padded tokens from real tokens.
            return_token_type_ids=True,
            return_tensors='pt'         # Returns pytorch tensors
        )
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
        if self.with_labels:  # True if the dataset has labels
            label = self.data['Y'][index]
            if len(label)>self.maxlen-2:
#                 label = ["B-<start>"] + label[:self.maxlen-2] + ["B-<end>"]
                label = ["O"] + label[:self.maxlen-2] + ["O"]
            else:
                label = ["O"] + label + ["O"] + ['O']*(self.maxlen - len(sent)-2)
#                 label = ["B-<start>"] + label + ["B-<end>"] + ['O']*(self.maxlen - len(sent)-2)
            label = torch.tensor(list(map(lambda x: tag2idx[x],label)))
            bin_label = torch.tensor(self.data['Y_bin'][index])
    
            return token_ids, attn_masks, token_type_ids, label, bin_label
        else:
            return token_ids, attn_masks, token_type_ids

train_dict = {"X":train_words_lists, "Y":train_tag_lists,"Y_bin": train_bin_tag_lists}
test_dict = {"X":test_words_lists, "Y":test_tag_lists,"Y_bin": test_bin_tag_lists}
dev_dict = {"X":dev_words_lists, "Y":dev_tag_lists,"Y_bin": dev_bin_tag_lists}
train_set = CustomDataset(train_dict, maxlen=MAX_LEN, model_name=bert_path, with_labels=True)
test_set = CustomDataset(test_dict, maxlen=MAX_LEN, model_name=bert_path, with_labels=True)
dev_set = CustomDataset(dev_dict, maxlen=MAX_LEN, model_name=bert_path, with_labels=True)
train_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
dev_loader = Data.DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False)

    
class BertMTNER_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.bilstm = nn.LSTM(config.hidden_size, 200,
                              batch_first=True,
                              bidirectional=True)
        self.fc = nn.Linear(400, config.num_labels)
        self.bin_fc = nn.Linear(400, 2)
        self.dropout = nn.Dropout(0.5)
        self.crf = CRF(self.num_labels, batch_first =True)
        self.init_weights()
    
    def neglikelihood(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, labels =None):
        feats,_ = self.get_lstm_features(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        crf_loss = self.crf(feats, labels, mask = attention_mask.type(torch.uint8))
        return -crf_loss
    
    def get_lstm_features(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        bert_outs = self.bert(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
        )
        sequence_output = bert_outs[0]   ## [B, S, H]
        sequence_output = self.dropout(sequence_output)
        lstm_out, _ = self.bilstm(sequence_output)
        lstm_out = self.dropout(lstm_out) 
        logits = self.fc(lstm_out)
        return logits, lstm_out
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        logits, lstm_out = self.get_lstm_features(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bin_logits = torch.mean(lstm_out, dim =1)     ## [B, S]
        bin_logits = self.dropout(bin_logits)
        bin_logits = self.bin_fc(bin_logits)    ## [B,2]
        crf_out = self.crf.decode(logits, mask = attention_mask.type(torch.uint8))   ## CRF loss
        return crf_out, bin_logits

def ner_evaluate(model, data_loader):
    pred_all =[]
    target_all = []
    model.eval()
    with torch.no_grad():
        for i,batch in enumerate(data_loader):
            input_ids, attn_mask,token_type,target,bin_target = batch[0], batch[1], batch[2], batch[3],batch[4]
            input_ids, attn_mask,token_type,target,bin_target = input_ids.to(device),attn_mask.to(device),token_type.to(device),target.to(device),bin_target.to(device)
            pred, _ = model(input_ids, attn_mask,token_type)
            pred_batch = []
            target_batch = []
            for i,p in enumerate(pred):
                pred_batch.extend(p)
                target_batch.extend(list(target[i][:len(p)]))
            pred = list(map(lambda x:idx2tag[x],pred_batch))
            target = list(map(lambda x:idx2tag[int(x)], target_batch))
            pred_all.extend(pred)
            target_all.extend(target)
    PRF = evaluate(target_all, pred_all, verbose=True)
    return PRF

def ner_loss(outputs, labels, attention_mask):
    loss_fct = nn.CrossEntropyLoss()
    active_loss = attention_mask.view(-1) == 1
    active_logits = outs.view(-1, len(tag2idx))
    active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
    loss = loss_fct(active_logits, active_labels)
    return loss

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    print( "Special Layer Learning rate is setted as:", lr)
    for i,param_group in enumerate(optimizer.param_groups):
        if i==1:
            param_group['lr'] = lr    ### 只更新lstm 和 crf的学习率
    return optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_config = BertConfig.from_pretrained(bert_path+'config.json',num_labels=len(idx2tag))
MTNER = BertMTNER_CRF.from_pretrained(bert_path, config = bert_config)
MTNER.to(device)

learning_rate = 3e-5
special_lr = 0.02
special_layers = nn.ModuleList([MTNER.bilstm, MTNER.fc, MTNER.bin_fc,MTNER.crf])
special_layers_params = list(map(id, special_layers.parameters()))
base_params = filter(lambda p: id(p) not in special_layers_params, MTNER.parameters())
optimizer_grouped_parameters = [{'params': base_params}, 
                                {'params': special_layers.parameters(), 'lr': special_lr}]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))
criterion_bin = nn.CrossEntropyLoss()
for epoch in range(15):
    train_ner_loss= 0
    train_bin_loss = 0
    MTNER.train()
    optimizer = lr_decay(optimizer, epoch, 0.05, special_lr)
    scheduler.step()
    for i,batch in enumerate(train_loader):
            input_ids, attn_mask,token_type,target,bin_target = batch[0], batch[1], batch[2], batch[3],batch[4]
            input_ids, attn_mask,token_type,target,bin_target = input_ids.to(device),attn_mask.to(device),token_type.to(device),target.to(device),bin_target.to(device)
            outs, bin_outs = MTNER(input_ids, attn_mask.type(torch.uint8),token_type)
            loss_ner = MTNER.neglikelihood(input_ids, attn_mask.type(torch.uint8),token_type, labels = target)
            loss_bin = criterion_bin(bin_outs, bin_target)
#             loss = loss_ner# + 0.1*loss_bin
            loss_ner.backward()
            torch.nn.utils.clip_grad_norm_(parameters=MTNER.parameters(), max_norm=5, norm_type=2)
            train_ner_loss += loss_ner.item()
            train_bin_loss += loss_bin.item()
            optimizer.step()
    print("epoch:",epoch+1,"train_ner_loss:",train_ner_loss,"train_bin_loss:",train_bin_loss,"train_loss:",train_ner_loss+ train_bin_loss)
    precision, recall, f1 = ner_evaluate(MTNER, test_loader)
