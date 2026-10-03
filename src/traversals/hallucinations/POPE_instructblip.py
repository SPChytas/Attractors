import requests
from PIL import Image
import torch
import numpy as np
from lavis.models import load_model_and_preprocess
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

parser.add_argument("--max_tokens", type=int, default=3)
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
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
processor = vis_processors['eval']

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
            concept = torch.mean(output[0], dim=1, keepdim=True)
        else:
            output = (output[0] + args.strength*concept, *output[1:])
        return output
    return hook

hook = model.llm_model.model.layers[args.target_layer].self_attn.register_forward_hook(remind_image(args.target_layer))
################################################### <Hook> ###################################################



predictions = []
true = []


for i, row in tqdm(df.iterrows(), total=len(df)):

    # Prompt
    prompt = row['text']

    # Image
    img_path = os.path.join(args.dataset_path, row['image'])
    raw_image = Image.open(img_path).convert("RGB")
    inputs = processor(raw_image).unsqueeze(0)



    with torch.inference_mode():
        with torch.no_grad():
            concept = None
            output = model.generate({"image": inputs.to(device), "prompt": prompt}, 
                                    use_nucleus_sampling=True, 
                                    max_length=args.max_tokens,
                                     num_beams=1,
                                     temperature=1,
                                     top_p=1, 
                                     repetition_penalty=1,)[0]
            

    predictions.append(recorder(output))
    true.append(row['label'])


print (args)
print_acc(predictions, true)