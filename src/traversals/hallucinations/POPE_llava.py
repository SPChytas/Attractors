import requests
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import matplotlib.pyplot as plt 
import argparse
from tqdm.auto import tqdm
import json
import random
import pandas as pd
import re



parser = argparse.ArgumentParser()
parser.add_argument("--strength", type=float, required=True)
parser.add_argument("--target_layer", type=int, default=17)

parser.add_argument("--max_tokens", type=int, default=10)
parser.add_argument("--pope_split", type=str, default='random')

parser.add_argument("--dataset_path", type=str, default='/data/takis/datasets/MSCOCO/2014/val2014', help='MSCOCO dataset path')

parser.add_argument("--output_folder", type=str, default='/data/takis/Hallucinations/POPE')
args = parser.parse_args()






def recorder(pred):
    NEG_WORDS = ["No", "not", "no", "NO"]
    
    pred = re.sub(r'\s+', ' ', pred.strip())
    pred = pred.replace('.', '')
    pred = pred.replace(',', '')
    words = pred.split(' ')
    
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return 'no'
    else:
        return 'yes'


def print_acc(pred_list, label_list):
    pos = 'yes'
    neg = 'no'
    yes_ratio = pred_list.count(pos) / len(pred_list)
    
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))



################################################### <Set up> ###################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_id, 
                                                      torch_dtype=torch.float16, 
                                                      low_cpu_mem_usage=True, 
                                                      cache_dir='/data/takis/models').to(device)

processor = AutoProcessor.from_pretrained(model_id, cache_dir='/data/takis/models')

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


# Dataset
df = pd.read_json('/data/takis/POPE/coco/coco_pope_%s.json' %(args.pope_split), lines=True)
################################################### </Set up> ###################################################


################################################### <Hook> ###################################################
concept = None

def remind_image(target_layer):
    global concept
    def hook(model, input, output):
        global concept
        if (output[0].shape[1] != 1):
            concept = torch.mean(output[0], dim=1)
        else:
            output = (output[0] + args.strength*concept, *output[1:])
        return output
    return hook

hook = model.language_model.model.layers[args.target_layer].self_attn.register_forward_hook(remind_image(args.target_layer))
################################################### <Hook> ###################################################



predictions = []
true = []




for i, row in tqdm(df.iterrows(), total=len(df)):

    # Prompt
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": row['text']},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Image
    img_path = os.path.join(args.dataset_path, row['image'])
    raw_image = Image.open(img_path).convert("RGB")
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
    



    with torch.inference_mode():
        with torch.no_grad():
            concept = None
            output = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=True)
            output = recorder(processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
            
    predictions.append(output)
    true.append(row['label'])


print (args)
print_acc(predictions, true)