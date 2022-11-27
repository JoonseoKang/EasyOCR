from easyocr.easyocr import *

import os
import csv
from tqdm import tqdm
import nlptutti as metrics
import pandas as pd
import cv2
import gluonnlp as nlp

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

from PIL import ImageFont, ImageDraw, Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prob", dest="prob", type=float, action="store")
parser.add_argument("-f", "--file_path", dest="file_path", action="store")
parser.add_argument("-w", "--wrong", dest="wrong", action="store_true", default=True)
parser.add_argument("-o", "--output_path", dest="output_path", action="store", default='./')
args = parser.parse_args()

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


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

def putText(cv_img, text, x, y, color=(0, 0, 0), font_size=22):
    # Colab이 아닌 Local에서 수행 시에는 gulim.ttc 를 사용하면 됩니다.
    # font = ImageFont.truetype("fonts/gulim.ttc", font_size)
    font = ImageFont.truetype('/home/jaykang/Desktop/deep-text-recognition-benchmark/EasyOCR/NanumFont/NanumGothicBold.ttf', font_size)
    img = Image.fromarray(cv_img)

    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=color)

    cv_img = np.array(img)

    return cv_img

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

reader = Reader(['ko'], gpu=True,
                model_storage_directory='./custom_model',
                user_network_directory='./custom_net',
                recog_network='custom')

file_path = args.file_path
# file_path = './sample/kb.jpg'

custom_ans = reader.readtext(file_path, detail=0)
custom = reader.readtext(file_path)

# reader_ori = Reader(lang_list=['ko', 'en'], gpu=True)
# custom_ans = reader_ori.readtext(file_path, detail=0)
# custom = reader_ori.readtext(file_path)

img = cv2.imread(file_path)
for (bbox, text, prob) in custom:
    if prob < args.prob and len(text) < 10:
        # print("[INFO] {:.4f}: {}".format(prob, text))

        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        # 추출한 영역에 사각형을 그리고 인식한 글자를 표기합니다.
        cv2.rectangle(img, tl, br, (0, 0, 255), 2)
        img = putText(img, text, tl[0], tl[1] - 15, (0, 0, 255), 15)
    # elif args.wrong:
    #     (tl, tr, br, bl) = bbox
    #     tl = (int(tl[0]), int(tl[1]))
    #     tr = (int(tr[0]), int(tr[1]))
    #     br = (int(br[0]), int(br[1]))
    #     bl = (int(bl[0]), int(bl[1]))
    #
    #     # 추출한 영역에 사각형을 그리고 인식한 글자를 표기합니다.
    #     cv2.rectangle(img, tl, br, (0, 255, 0), 2)
    #     img = putText(img, text, tl[0], tl[1] - 15, (0, 255, 0), 15)


text = ' '.join(custom_ans)

model = torch.load('./model.pt')

akqjqtk123!
lst = [[file_path, text, 0]]
bertmodel, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

data_test = BERTDataset(lst, 1, 2, tok, 512, True, False)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, num_workers=5)


predictions =[]
with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        # print(batch_id, (token_ids, valid_length, segment_ids, label))
        token_ids = token_ids.long().cuda()
        segment_ids = segment_ids.long().cuda()
        valid_length = valid_length

        label = label.long().cuda()
        out = model(token_ids, valid_length, segment_ids)
        _, max_indices = torch.max(out, 1)

        predictions.extend(max_indices.detach().cpu().numpy())


mapping ={0: '농림.축산지원', 1: '도시개발', 2: '산업진흥', 3: '상.하수도관리', 4: '인.허가', 5: '일반행정', 6: '주민복지', 7: '주민생활지원', 8: '주민자치', 9: '지역문화', 10: '지역환경.산림', 11: '회계.예산'}
print(mapping[predictions[0]])
print('=' * 50)
print(text)
print('=' * 50)
print('Save image file:', file_path)
cv2.imwrite(args.output_path + (args.file_path).split('/')[1]+ '_' + str(args.prob) + '_output.jpg', img)