import argparse
import logging
import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import nlptutti as metrics
from BertClassifier import TextClassificationModel, TextClassificationDataset
from easyocr.easyocr import Reader
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup


def main(args):
    logging.basicConfig(level=logging.INFO)

    # Initialize EasyOCR reader
    reader = Reader(
        ["ko"],
        gpu=True,
        model_storage_directory="./custom_model",
        user_network_directory="./custom_net",
        recog_network="custom",
    )

    text, regions = read_text_from_image(reader, args.file_path, args.prob, args.wrong)
    category = classify_text(text, args.model_path)
    save_output_image(args.file_path, args.output_path, args.prob, regions, category)


def read_text_from_image(reader, file_path, prob_threshold, show_wrong_regions):
    img = cv2.imread(file_path)
    results = reader.readtext(file_path, detail=1)

    text = ""
    regions = []
    for bbox, text, prob in results:
        if prob < prob_threshold and len(text) < 10:
            if show_wrong_regions:
                draw_bounding_box(img, bbox, (0, 255, 0))
                draw_text(img, text, bbox[0], bbox[1] - 15, (0, 255, 0), 15)
        else:
            draw_bounding_box(img, bbox, (0, 0, 255))
            draw_text(img, text, bbox[0], bbox[1] - 15, (0, 0, 255), 15)
            regions.append(text)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    text = " ".join(regions)

    return text, regions


def classify_text(text, model_path):
    model = torch.load(model_path)

    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data = [[text, 0]]
    dataset = BERTDataset(data, 1, 2, tok, 512, True, False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=5)

    predictions = []
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(dataloader)):
            token_ids = token_ids.long().cuda()
            segment_ids = segment_ids.long().cuda()
            valid_length = valid_length
            label = label.long().cuda()
            out = model(token_ids, valid_length, segment_ids)
            _, max_indices = torch.max(out, 1)
            predictions.extend(max_indices.detach().cpu().numpy())

    # Map predicted category index to category name
    mapping = {
        0: "농림.축산지원",
        1: "도시개발",
        2: "산업진흥",
        3: "상.하수도관리",
        4: "인.허가",
        5: "일반행정",
        6: "주민복지",
        7: "주민생활지원",
        8: "주민자치",
        9: "지역문화",
        10: "지역환경.산림",
        11: "회계.예산",
    }
    category = mapping[predictions[0]]

    return category


def save_output_image(file_path, output_path, prob_threshold, regions, category):
    img = cv2.imread(file_path)

    for bbox, text, prob in regions:
        if prob < prob_threshold and len(text) < 10:
            draw_bounding_box(img, bbox, (0, 255, 0))
            draw_text(img, text, bbox[0], bbox[1] - 15, (0, 255, 0), 15)
        else:
            draw_bounding_box(img, bbox, (0, 0, 255))
            draw_text(img, text, bbox[0], bbox[1] - 15, (0, 0, 255), 15)
    draw_text(img, category, 10, 30, (0, 0, 255), 20)

    output_file_name = os.path.join(output_path, os.path.basename(file_path))
    output_file_name = output_file_name.replace(".jpg", f"_{prob_threshold}_output.jpg")
    cv2.imwrite(output_file_name, img)


def draw_bounding_box(img, bbox, color):
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    cv2.rectangle(img, tl, br, color, 2)


def draw_text(img, text, x, y, color, font_size):
    font = ImageFont.truetype(
        "/home/jaykang/Desktop/deep-text-recognition-benchmark/EasyOCR/NanumFont/NanumGothicBold.ttf", font_size
    )
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prob", dest="prob", type=float, action="store")
    parser.add_argument("-f", "--file_path", dest="file_path", action="store")
    parser.add_argument("-w", "--wrong", dest="wrong", action="store_true", default=True)
    parser.add_argument("-o", "--output_path", dest="output_path", action="store", default="./")
    parser.add_argument("-m", "--model_path", dest="model_path", action="store", default="./model.pt")
    args = parser.parse_args()

    main(args)
