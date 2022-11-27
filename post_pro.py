from easyocr.easyocr import *
from hanspell import spell_checker
from pprint import pprint
import cv2
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

reader = Reader(['ko'], gpu=True,
                model_storage_directory='./custom_model',
                user_network_directory='./custom_net',
                recog_network='custom')
reader_ori = Reader(lang_list=['ko', 'en'], gpu=True)

sample_file = './sample/5350261-2006-0001-0018.jpg'
custom_ans = reader.readtext(sample_file, detail=0)
custom = reader.readtext(sample_file)
custom

ori = reader_ori.readtext(sample_file)

#
# tmp = spell_checker.check(custom_ans)
# pprint(tmp)

import nlptutti as metrics
refs = "(장유)"
preds = "(장유"
r = metrics.get_cer(refs, preds)

img = cv2.imread(sample_file)
tmp_img = img

for i in custom:
    bb, text, confidence = i
    # if confidence < 0.7:
    #     tmp_img = cv2.rectangle(tmp_img, list(map(int, bb[0])), list(map(int, (bb[2]))), (0, 0, 255), 3)
    print(bb, text, confidence)



tmp = ' '.join(custom_ans)
tmp

from LMkor.examples.bertshared_summarization import Summarize
summarize = Summarize('kykim/bertshared-kor-base')
summarize(tmp)