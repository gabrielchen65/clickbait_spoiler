import torch
from torch.utils.data import Dataset, DataLoader
import json
import spacy
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
import time
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split

data_file = '/Users/krishthek/Documents/uWaterloo/msci641/project/clickbait-detection-msci641-s23/train.jsonl'
class CustomDataset(Dataset):
    def __init__(self, data_file):
        # with open(data_file, 'r') as f:
        #     for line in f:
        #         json_obj = json.loads(line)  # Load each JSON object from a separate line
        #         data.append(json_obj) 
        postText_arr = []
        tags_arr=[]
        with open(data_file, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            postText_arr.extend(result['postText'])
            tags_arr.extend(result['tags'])
            
        # print(postText_arr)
        # Extract relevant fields from the JSON data
        self.postText = postText_arr
        self.tags = tags_arr

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        # Preprocess and return the sample and label
        sample = self.postText[index]
        label = self.tags[index]
        return label, sample

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim,hidden_size, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.init_weights()
        #lstm 
        self.lstm = nn.LSTM(input_size=embed_dim,hidden_size=self.hidden_dim,num_layers=10, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, num_class)
        self.sig = nn.Sigmoid()


    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        
        return self.fc(embedded)


class SentimentRNN(nn.Module):
    def __init__(self,output_dim, no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
        super(SentimentRNN,self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #lstm 
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,num_layers=no_layers, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden



    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden

def main():
    train_file = '/Users/krishthek/Documents/uWaterloo/msci641/project/clickbait-detection-msci641-s23/train.jsonl'
    val_test_file = '/Users/krishthek/Documents/uWaterloo/msci641/project/clickbait-detection-msci641-s23/val.jsonl'
    test_file = '/Users/krishthek/Documents/uWaterloo/msci641/project/clickbait-detection-msci641-s23/test.jsonl'
    # train_iter = iter(CustomDataset(train_file))

    train_iter = CustomDataset(train_file)
    val_test_iter = CustomDataset(val_test_file)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))

    label_mapping = {
    "passage": 0,
    "phrase": 1,
    "multi": 2,
    # Add more mappings as needed
    }
    label_pipeline = lambda label: label_mapping[label]

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)
    
    dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize,num_class=3,hidden_size=10).to(device)

    # no_layers = 2
    # vocab_size = len(vocab) + 1 #extra 1 for padding
    # embedding_dim = 64
    # output_dim = 1
    # hidden_dim = 256

    # model = SentimentRNN(output_dim, no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
    # #moving to gpu
    # model.to(device)

    import time


    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches "
                    "| accuracy {:8.3f}".format(
                        epoch, idx, len(dataloader), total_acc / total_count
                    )
                )
                total_acc, total_count = 0, 0
                start_time = time.time()


    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc / total_count
    
    # Hyperparameters
    EPOCHS = 20 # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64  # batch size for training
    
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(val_test_iter)

    num_train = int(len(train_dataset) * 0.80)
    split_train_, split_valid_ = random_split(
        train_dataset, [num_train, len(train_dataset) - num_train]
    )

    train_dataloader = DataLoader(
        split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    valid_dataloader = DataLoader(
        split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)

    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader)
    print("test accuracy {:8.3f}".format(accu_test))

if __name__ == '__main__':
    main()
    # get_json()