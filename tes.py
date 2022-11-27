import datetime
from easyocr.easyocr import *
import os
import csv
from tqdm import tqdm
import nlptutti as metrics
import pandas as pd
from numba import jit
import random
# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# def get_files(path):
#     file_list = []
#
#     files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
#     files.sort()
#     abspath = os.path.abspath(path)
#     for file in files:
#         file_path = os.path.join(abspath, file)
#         file_list.append(file_path)
#
#     return file_list, len(file_list)

num = 0
custom_num = 0
reader_ori = 0
reader = Reader(['ko'], gpu=True,
                model_storage_directory='./custom_model',
                user_network_directory='./custom_net',
                recog_network='custom')

reader_ori = Reader(lang_list=['ko', 'en'], gpu=True)

f_name =[]
with open('./data/crop/test.txt', 'r') as f:
    rdr = csv.reader(f)
    for line in tqdm(rdr):
        # print(line)
        file_name, text = line[0].split('\t')
        # print(file_name, text)
        f_name.append([file_name, text])
        # file, original, custom, gt
    f.close()


# len(f_name)
# random.seed(2021)
sample = random.sample(f_name, 10000)

result = []
r_app = result.append

for line in tqdm(sample):
    file_name, text = line[0], line[1]
    print(file_name, text)

    r_app([file_name, reader_ori.readtext(os.path.join('./data/crop/image/', file_name), detail=0),
           reader.readtext(os.path.join('./data/crop/image/', file_name), detail=0), text])
    # print(result)

df = pd.DataFrame(result)
df.to_csv('./result.csv', header=None, index=None)


# result = pd.read_csv('./EasyOCR/result.csv', header=None)
# result.head()



with open('./result.csv', 'r') as f:
    rdr = csv.reader(f)
    ori = 0
    cus = 0
    n = 0
    for line in rdr:
        n += 1
        file, original, custom, gt = line
        pred_ori = original[2:-2]
        pred_cus = custom[2:-2]
        refs = gt
        try:
            result_ori = metrics.get_cer(refs, pred_ori)['cer']
            result_cus = metrics.get_cer(refs, pred_cus)['cer']
        except:
            print(gt)
            result_ori = 0
            result_cus = 0
        ori += result_ori
        cus += result_cus
    f.close()

print('ori', ori/n)
print('custom', cus/n)



with open('./result.csv', 'r') as f:
    rdr = csv.reader(f)
    ori = 0
    cus = 0
    n =0
    for line in rdr:
        n += 1
        file, original, custom, gt = line
        if original[2:-2] == gt:
            ori += 1
        if custom[2:-2] == gt:
            cus += 1

        # if original[2:-2] != gt or custom[2:-2] != gt:
        #     print(file, original[2:-2], custom[2:-2], gt)
    f.close()

print('ori', ori/n)
print('custom', cus/n)
# file_ath = './sample/sample.jpg'
#
# reader = Reader(['ko'], gpu=True,
#                 model_storage_directory='./custom_model',
#                 user_network_directory='./custom_net',
#                 recog_network='custom')
#
# result = reader.readtext(file_path)
# tmp = reader.readtext(file_path, detail=0)
#
# reader_ori = Reader(lang_list=['ko', 'en'], gpu=True)
# results = reader.readtext(file_path)
# len(result)
# len(results)
# t = reader.readtext(file_path, detail=0)
# for i in range(len(result)):
#     # if tmp[i] != t[i]:
#     print(tmp[i], t[i])