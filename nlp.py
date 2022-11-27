import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


train = pd.read_csv('./train.csv')
val = pd.read_csv('./val.csv')
test = pd.read_csv('./test.csv')

train = pd.concat([train,val])
test_label =  [0 for i in range(len(test))]
test['label'] = test_label

bertmodel, vocab = get_pytorch_kobert_model()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(train['class'])
train['class'] = encoder.transform(train['class'])
test['class'] = encoder.transform(test['class'])
train.head()

mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
mapping

lst=[list(train.iloc[i]) for i in range(len(train))]
print(lst[:10])

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


max_len = 512 # 해당 길이를 초과하는 단어에 대해선 bert가 학습하지 않음
batch_size = 4
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 100
learning_rate = 5e-5

from sklearn.model_selection import train_test_split
train, valid = train_test_split(lst, test_size=0.2, random_state=42)
print("train shape is:", len(train))
print("valid shape is:", len(valid))

data_train = BERTDataset(train, 1, 2, tok, max_len, True, False)
data_valid = BERTDataset(valid, 1, 2, tok, max_len, True, False)
data_train[0]

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
valid_dataloader = torch.utils.data.DataLoader(data_valid, batch_size=batch_size, num_workers=5)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=12,  # softmax 사용 <- binary일 경우는 2
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


model = BERTClassifier(bertmodel, dr_rate=0.5).cuda()
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


for e in range(num_epochs):
    print(f' ====================== epoch %d ======================' % (e+1) )
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().cuda()
        segment_ids = segment_ids.long().cuda()
        valid_length= valid_length
        label = label.long().cuda()
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # gradient clipping
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print(f'Iteration %3.d | Train Loss  %.4f | Classifier Accuracy %2.2f' % (batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    model.eval() # 평가 모드로 변경

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(valid_dataloader)):
        token_ids = token_ids.long().cuda()
        segment_ids = segment_ids.long().cuda()
        valid_length= valid_length
        label = label.long().cuda()
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} valid acc {}".format(e+1, test_acc / (batch_id+1)))


torch.save(model, './model.pt')

model = torch.load('./model.pt')
t = test.drop('class', axis=1)

lst=[list(test.iloc[i]) for i in range(len(test))]
data_test = BERTDataset(lst, 1, 2, tok, 512, True, False)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=4, num_workers=5)

predictions =[]
with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().cuda()
        segment_ids = segment_ids.long().cuda()
        valid_length = valid_length
        print(token_ids, segment_ids)
        label = label.long().cuda()
        out = model(token_ids, valid_length, segment_ids)
        _, max_indices = torch.max(out, 1)

        predictions.extend(max_indices.squeeze(0).detach().cpu().numpy())


print(predictions)

test['label'] = predictions

test[['class', 'label']]

num =0
for i in range(len(test)):
    if test.iloc[i]['class'] == test.iloc[i]['label']:
        num += 1

num/len(test)