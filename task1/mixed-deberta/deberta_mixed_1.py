
import csv
import re
import random
import sys
import os
import json
import pickle
import tensorflow as tf
import numpy as np

def return_data(file):
    postText_arr = []
    tags_arr=[]
    with open(file, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        postText_arr.append(result['targetTitle'])
#         print(result['targetTitle'])
#         print(result['postText'][0])
        tags_arr.extend(result['tags'])
    # print(postText_arr)
    return postText_arr, tags_arr


def tokenize(line):
    ''' Input is a line.
    Tokenizes and returns line as array where each element is a token.
    There is no stop word removal in this function.
    '''
    # tokenLine = [] #to store every token
    # print(line)
    line=line.lower()
    splitLine = re.split('\W+', line)
    tokenizedLine =[]
    for i in range(len(splitLine)):
        if(splitLine[i]!=''):
            tokenizedLine.append(splitLine[i])
    return tokenizedLine

train_file = '../task1/clickbait-detection-msci641-s23/train.jsonl'
val_file = '../task1/clickbait-detection-msci641-s23/val.jsonl'
train_text, train_label = return_data(train_file)
val_text, val_label = return_data(val_file)

label2id = {
"passage": 0,
"phrase": 1,
"multi": 2
}
id2label = {
    0: "passage",
    1:"phrase",
    2:"multi"
}

y_train = [label2id[key] for key in train_label]
y_val = [label2id[key] for key in val_label]

y_train = []   #to compare multi vs others
for val in train_label:
    if val == 'multi':
        y_train.append(1)
    else:
        y_train.append(0)
y_train = np.array(y_train)

y_val = []        
for val in val_label:
    if val == 'multi':
        y_val.append(1)
    else:
        y_val.append(0)
y_val = np.array(y_val) 


from transformers import AutoTokenizer

tokenizer1 = AutoTokenizer.from_pretrained("microsoft/deberta-base")
train_encodings = tokenizer1(train_text, truncation=True, padding=True)
val_encodings = tokenizer1(val_text, truncation=True, padding=True)

import torch

class ClickDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ClickDataset(train_encodings, y_train)
val_dataset = ClickDataset(val_encodings, y_val)
print(train_dataset[1])

from transformers import DebertaV2ForSequenceClassification, Trainer, TrainingArguments
from transformers import DebertaForSequenceClassification

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=4,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.4,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    report_to="none"
)

model1 = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base")


from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model1.to(device)
model1.train()

from transformers import AdamW
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# optim = AdamW(model.parameters(), lr=5e-5)
optim = torch.optim.AdamW(model1.parameters(), lr = 5.9e-6)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

trainer = Trainer(
    model=model1,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


trainer.train()
trainer.evaluate()

model1.save_pretrained("deberta_model1_1", from_pt=True) 


def return_non_multi(file):
    postText_arr = []
    tags_arr=[]
    with open(file, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
#         print(result['tags'])
        if(result['tags']==['multi']):
#             print("hello")
            continue
        postText_arr.append(result['targetTitle'])
        tags_arr.extend(result['tags'])
#     print(tags_arr)
    return postText_arr, tags_arr

train_text_nonMulti, train_label_nonMulti = return_non_multi(train_file)
val_text_nonMulti, val_label_nonMulti = return_non_multi(val_file)

y_train_nonMulti =[]
for val in train_label_nonMulti:
    if val == 'phrase':
        y_train_nonMulti.append(1)
    else:
        y_train_nonMulti.append(0)
y_train_nonMulti = np.array(y_train_nonMulti) 

y_val_nonMulti =[]
for val in val_label_nonMulti:
    if val == 'phrase':
        y_val_nonMulti.append(1)
    else:
        y_val_nonMulti.append(0)
y_val_nonMulti = np.array(y_val_nonMulti)

from transformers import AutoTokenizer

tokenizer2 = AutoTokenizer.from_pretrained("microsoft/deberta-base")
train_encodings = tokenizer2(train_text_nonMulti, truncation=True, padding=True)
val_encodings = tokenizer2(val_text_nonMulti, truncation=True, padding=True)

train_dataset = ClickDataset(train_encodings, y_train_nonMulti)
val_dataset = ClickDataset(val_encodings, y_val_nonMulti)
print(train_dataset[1])

from transformers import DebertaV2ForSequenceClassification, Trainer, TrainingArguments
from transformers import DebertaForSequenceClassification

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=4,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.5,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    report_to="none"
)

model2 = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base")

from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model2.to(device)
model2.train()

from transformers import AdamW
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# optim = AdamW(model.parameters(), lr=5e-5)
optim = torch.optim.AdamW(model2.parameters(), lr = 8.2e-6)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

trainer = Trainer(
    model=model2,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


trainer.train()

trainer.evaluate()

model2.save_pretrained("/kaggle/working/deberta_model2_1", from_pt=True) 

from transformers import TextClassificationPipeline

pipe1 = TextClassificationPipeline(model=model1, tokenizer=tokenizer1, device=0)
pipe2 = TextClassificationPipeline(model=model2, tokenizer=tokenizer2, device=0)



out = pipe2("yo this is goos")
print(out[0]['label'])

input_file = '../task1/clickbait-detection-msci641-s23/val.jsonl'
output_file = 'mixed_deberta1_val.csv'

with open(input_file, 'r') as inp, open(output_file, 'w') as out:
    out.write("id,spoilerType\n")
    test_arr = []
    for i in inp:
        i = json.loads(i)
#         print(str(i["targetTitle"]))
        out1 = pipe1(str(i["targetTitle"]))
        out2 = pipe2(str(i["targetTitle"]))
#         print(out1)
#         print(out2[0])
        if(out1[0]['score']>out2[0]['score'] and out1[0]['label']=="LABEL_1"):
            result = 'multi'
        else:
            if(out2[0]['label']=="LABEL_1"):
                result = 'phrase'
            else:
                result = "passage"
                
        out.write(str(i['id'])+ ',' + result+ ',' +  str(i['tags'][0]) +'\n')

input_file = '../task1/clickbait-detection-msci641-s23/test.jsonl'
output_file = 'working/mixed_deberta1.csv'

with open(input_file, 'r') as inp, open(output_file, 'w') as out:
    out.write("id,spoilerType\n")
    test_arr = []
    for i in inp:
        i = json.loads(i)
#         print(str(i["targetTitle"]))
        out1 = pipe1(str(i["targetTitle"]))
        out2 = pipe2(str(i["targetTitle"]))
#         print(out1)
#         print(out2[0])
        if(out1[0]['score']>out2[0]['score'] and out1[0]['label']=="LABEL_1"):
            result = 'multi'
        else:
            if(out2[0]['label']=="LABEL_1"):
                result = 'phrase'
            else:
                result = "passage"
        out.write(str(i['id'])+ ',' + result +'\n')
