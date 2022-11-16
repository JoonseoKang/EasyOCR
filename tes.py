from easyocr.easyocr import *
import os
import csv

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
result = []
with open('./data/crop/test.txt', 'r') as f:
    rdr = csv.reader(f)
    for line in rdr:
        num += 1
        file_name, text = line[0].split('\t')
        ori_ans = reader_ori.readtext(file_path, detail=0)
        custom_ans = reader.readtext(file_path, detail=0)
        # file, original, custom, gt
        result.append([file_name, ori_ans, custom_ans, text])
    f.close()

file_path = './sample/sample.jpg'

reader = Reader(['ko'], gpu=True,
                model_storage_directory='./custom_model',
                user_network_directory='./custom_net',
                recog_network='custom')

result = reader.readtext(file_path)
tmp = reader.readtext(file_path, detail=0)

reader_ori = Reader(lang_list=['ko', 'en'], gpu=True)
results = reader.readtext(file_path)
len(result)
len(results)
t = reader.readtext(file_path, detail=0)
for i in range(len(result)):
    # if tmp[i] != t[i]:
    print(tmp[i], t[i])