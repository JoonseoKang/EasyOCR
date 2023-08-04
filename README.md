# Text Classification with EasyOCR

OCR결과를 이용한 행정 문서 카테고리 분류입니다. OCR library는 [EasyOCR](https://github.com/JaidedAI/EasyOCR) 을 활용했습니다.  
[korean_g2](https://www.jaided.ai/easyocr/modelhub/) pretrain 모델을
[recognition model](https://github.com/clovaai/deep-text-recognition-benchmark) 을 이용하여 고유 명사, 특수 문자에 대한 디테일을 개선했습니다.  
OCR text 결과를 BertClassifier의 input으로 활용해 문서 분류를 진행합니다. 


## Text Classification

Bertclassifier를 이용하여 분류 모델을 학습합니다:

``` bash
$ python custom_net/nlp.py
```
Note 1: `train.csv, val.csv, test.csv`을 아래와 같은 형태로 만들어 `BertClassifier.py`에 파싱해줍니다.  
Note 2: Classifier 모델을 저장하여 활용합니다.



text	 | class
------	 | -----
세무시확에지 도6로 서울역사편찬원 seou.go.kr. 수신신 국립중앙극장장(공연예술박물관장) (경유) 제목 2022년 하반기 서울 역사강좌 강의 협조 요청(수정) 1. 귀 기관의 무궁한 발전을 기원합니다.. 2. 서울역사편찬원에서는 매년 정기적으로 시민 대 상의 교양강좌인 서울역사강좌률 운영 하여 서울시민들에게 서울에 대한 이해와 가치물 제공하고 있습니다. 3. 이와 관련하 여 귀 기관 소속 학예연구사에게 아래와 같은 내용으로 2022년 하반기서 울역사강좌 강의률 의퇴하고자 하오니 출강에 적극 협조 부탁드립니다.. 가. 의론주제: 2022년도 하반기 서울역사강좌) <문화공간, 서울역사이야기> 10강 나. 의퇴내용= 시민강 좌 * *** * **" 다. 사: 주선영- 라. 강의일시: 2022년 11월 11일(금) 2회(총 4시간) 강의 13:00~15:00(a반), 15:00~17:00(b반) 마. 강의장소: 서울역사편찬원 강의: 기타 궁금하신 사항은 서울역사편찬원(02-413-9622 연구원 김동하)으로 연락주시기 바랍니 다. 붙임임 1. 2022년 하반기 서울역사강좌 일정표 1부. 끝| 주민생활지원
## Usage

``` python
$ python custome_net/ocr_with_classifier.py
```
Image, text 2가지 Output이 나오게 됩니다. 

```
Options:
  -w <wrong>, 틀릴 확률이 높은 부분만 표시할지 전체를 다 표시할지 선택합니다.(default: True)
  -o <ouput_path>, --output_path output 저장 경로
  -f <input_image_path>, --file_path OCR Input image path 
  -p <probabilty>, --prob confidence 임계값
  -m <pretrained model path>, --model_path NLP pretrained model 경로