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

from nlp import BERTDataset, BERTClassifier
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prob", dest="prob", type=float, action="store")
parser.add_argument("-f", "--file_path", dest="file_path", action="store")
parser.add_argument("-w", "--wrong", dest="wrong", action="store_true", default=True)
parser.add_argument("-o", "--output_path", dest="output_path", action="store", default='./')
parser.add_argument("-m", "--model_path", dest="model_path", action="store", default='./model.pt')
args = parser.parse_args()


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

custom_ans = reader.readtext(file_path, detail=0)
custom = reader.readtext(file_path)

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
    elif args.wrong:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))

        # 추출한 영역에 사각형을 그리고 인식한 글자를 표기합니다.
        cv2.rectangle(img, tl, br, (0, 255, 0), 2)
        img = putText(img, text, tl[0], tl[1] - 15, (0, 255, 0), 15)


text = ' '.join(custom_ans)

# load model
model = torch.load(args.model_path)

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


mapping = {0: '농림.축산지원', 1: '도시개발', 2: '산업진흥', 3: '상.하수도관리', 4: '인.허가', 5: '일반행정', 6: '주민복지', 7: '주민생활지원', 8: '주민자치', 9: '지역문화', 10: '지역환경.산림', 11: '회계.예산'}
print(mapping[predictions[0]])
print('=' * 50)
print(text)
print('=' * 50)
print('Save image file:', file_path)
cv2.imwrite(args.output_path + (args.file_path).split('/')[1]+ '_' + str(args.prob) + '_output.jpg', img)