
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

def return_concat(file):
    article = []
    tag = []
    with open(file, "r") as json_file:
        data = json.load(json_file)
#     print(len(data['data']))
    for i in range(len(data['data'])):
        article.append(data["data"][i]["article"])
        tag.append(data["data"][i]["tags"][0])
    
    return article,tag

# train_file = '/kaggle/input/concated/train_4summary_all-concat.json'
# val_file = '/kaggle/input/concated/val_4summary_all-concat.json'
# train_text, train_label = return_concat(train_file)
# val_text, val_label = return_concat(val_file)

train_file = '/kaggle/input/msci-clickbait/train.jsonl'
val_file = '/kaggle/input/msci-clickbait/val.jsonl'
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

from transformers import DebertaV2ForSequenceClassification, Trainer, TrainingArguments
from transformers import DebertaForSequenceClassification

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=5,             
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,   
    warmup_steps=500,                
    weight_decay=0.4,               
    logging_dir='./logs',            
    logging_steps=10,
    report_to="none",
    save_total_limit='2',
    save_strategy='no',
    load_best_model_at_end=False
)

model1 = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels = 3)

from torch.utils.data import DataLoader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model1.to(device)
model1.train()

from transformers import AdamW
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# optim = AdamW(model.parameters(), lr=5e-5)
optim = torch.optim.AdamW(model1.parameters(), lr = 3e-5)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import load_metric
def custom_metrics(eval_pred):
    metric1 = load_metric("precision")
    metric2 = load_metric("recall")
    metric3 = load_metric("f1")
    metric4 = load_metric("accuracy")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric1.compute(predictions=predictions, references=labels, average="macro")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="macro")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="macro")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

trainer = Trainer(
    model=model1,                         
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,
    compute_metrics=custom_metrics
)


trainer.train()
trainer.evaluate()

model1.save_pretrained("/kaggle/working/deberta_concat_multiclass_1", from_pt=True) 

from transformers import TextClassificationPipeline

pipe1 = TextClassificationPipeline(model=model1, tokenizer=tokenizer1, device=0)


input_file = '/kaggle/input/msci-clickbait/val.jsonl'
output_file = '/kaggle/working/debertav1_multi2_val.csv'
with open(input_file, 'r') as inp, open(output_file, 'w') as out:
    out.write("id,spoilerType\n")
    test_arr = []
    for i in inp:
        i = json.loads(i)
#         print(str(i["targetTitle"]))
        out1 = pipe1(str(i["targetTitle"]))
#         print(out1)
#         print(out2[0])
        if(out1[0]['label']=="LABEL_0"):
            result = 'passage'
        elif(out1[0]['label']=="LABEL_1"):
            result = 'phrase'
        else:
            result = "multi"
                
        out.write(str(i['id'])+ ',' + result +',' +str(i['tags'][0])+'\n')

input_file = '/kaggle/input/msci-clickbait/test.jsonl'
output_file = '/kaggle/working/debertav1_multi2.csv'
with open(input_file, 'r') as inp, open(output_file, 'w') as out:
    out.write("id,spoilerType\n")
    test_arr = []
    for i in inp:
        i = json.loads(i)
#         print(str(i["targetTitle"]))
        out1 = pipe1(str(i["targetTitle"]))
#         print(out1)
#         print(out2[0])
        if(out1[0]['label']=="LABEL_0"):
            result = 'passage'
        elif(out1[0]['label']=="LABEL_1"):
            result = 'phrase'
        else:
            result = "multi"
        # print(combinedRow, y_pred_real)
        out.write(str(i['id'])+ ',' + result + '\n')

