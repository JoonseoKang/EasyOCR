import sys
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
import re
import time
from collections import Counter
import shutil
import json

import transformers
from transformers import BertTokenizer,AdamWeightDecay,TFRobertaModel,TFBertModel

# import tensorflow as tf
# import keras
# from keras.callbacks import EarlyStopping,ModelCheckpoint

import sklearn
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import StratifiedKFold

# import pororo
# from pororo import Pororo
tokenizer = BertTokenizer.from_pretrained('klue/roberta-large')
# path = '/home/jaykang/Desktop/kb/ocr/Validation/output/test/'
#
# def read_all_file(path):
#     output = os.listdir(path)
#     file_list = []
#
#     for i in output:
#         if os.path.isdir(path+"/"+i):
#             file_list.extend(read_all_file(path+"/"+i))
#         elif os.path.isfile(path+"/"+i):
#             file_list.append(path+"/"+i)
#
#     return file_list
#
# def copy_all_file(file_list, new_path):
#     for src_path in file_list:
#         file = src_path.split("/")[-1]
#         shutil.copyfile(src_path, new_path+"/"+file)
#
# file_list = read_all_file(path)
# copy_all_file(file_list, path)
# print(3)
# my = os.listdir('/home/jaykang/Desktop/kb/ocr/Validation/my/')
# new_path = '/home/jaykang/Desktop/kb/ocr/Validation/my_json/'
#
# for i in tqdm(my):
#     file_list = read_all_file(os.path.join(path, i))
#     copy_all_file(file_list, os.path.join(new_path, i))

path ='/home/jaykang/Desktop/kb/ocr/Validation/output/train/'
train_list = os.listdir(path)
train = []
tr = train.append
for i in tqdm(train_list):
    with open(os.path.join(path, i)) as json_file:
        json_data = json.load(json_file)

    # from pprint import pprint
    # pprint(json_data)
    # json_data['annotations'][0]
    s = ''
    for j in json_data['annotations']:
        # print(''.join(i['annotation.text']))
        s += j['annotation.text'] + ' '

    s = s.rstrip()
    cls = json_data['images'][0]['image.category']
    tr([i, s, cls])

train.insert(0, ['file', 'text', 'class'])
train[0]

df = pd.DataFrame(train)
df.to_csv('./train.csv', index=False, header=None)


path ='/home/jaykang/Desktop/kb/ocr/Validation/output/test/'
train_list = os.listdir(path)
train = []
tr = train.append
for i in tqdm(train_list):
    with open(os.path.join(path, i)) as json_file:
        json_data = json.load(json_file)

    # from pprint import pprint
    # pprint(json_data)
    # json_data['annotations'][0]
    s = ''
    for j in json_data['annotations']:
        # print(''.join(i['annotation.text']))
        s += j['annotation.text'] + ' '

    s = s.rstrip()
    cls = json_data['images'][0]['image.category']
    tr([i, s, cls])

train.insert(0, ['file', 'text', 'class'])
train[0]

df = pd.DataFrame(train)
df.to_csv('./test.csv', index=False, header=None)