import sys
import torch
import json
import pandas as pd
from collections import defaultdict
from transformers import BartTokenizer, BartForConditionalGeneration
from data_process import *
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'
# load_data
def controllable_predict(data, labels, bart_model_path, result_save_path, max_len=64, num_beams=5, return_seq_num=3):
    result = {}
    print('load tokenizer and model')
    tokenizer = BartTokenizer.from_pretrained('PATH-Bart')
    model = BartForConditionalGeneration.from_pretrained(bart_model_path).to('cuda')
    for i in range(len(data)):
        input_text = data[i]
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
        if len(input_ids[0]) > 512:
            input_ids = input_ids[:, :512]

        # 1. Beam Search
        outputs = model.generate(input_ids, max_length=max_len, num_beams=num_beams, early_stopping=True, num_return_sequences=return_seq_num)
        outputs_str = []
        for j, output in enumerate(outputs):
            output_str = tokenizer.decode(output, skip_special_tokens=True)
            outputs_str.append(output_str)
            print("{}: {}".format(j, output_str))

        result[labels[i]] = outputs_str
    with open(result_save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4)) 

def sequence_predict(data, labels, bart_model_path, result_save_path, max_len=128, num_beams=5):
    result = {}
    tokenizer = BartTokenizer.from_pretrained('PATH-Bart')
    model = BartForConditionalGeneration.from_pretrained(bart_model_path).to('cuda')
    for i in range(len(data)):
        input_text = data[i]
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
        if len(input_ids[0]) > 512:
            input_ids = input_ids[:, :512]

        # 1. Beam Search
        outputs = model.generate(input_ids, max_length=max_len, num_beams=num_beams, early_stopping=True, num_return_sequences=1)
        outputs_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'query: {labels[i]} || label: {outputs_str}')

        result[labels[i]] = outputs_str.split(',')
    with open(result_save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4)) 



if __name__ == '__main__':
    query = ['apple']
    docs = [['apple is a kind of fruit']]
    predict_data = predict_your_data(query, docs)
    controllable_predict(predict_data[0], predict_data[1], 'Path-model', './res.json', 32, 5, 3)
    # sequence_predict(predict_data[0], predict_data[1], 'Path-model', './res.json', 128, 5)
