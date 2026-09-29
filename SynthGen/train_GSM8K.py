from datasets import load_dataset
from tqdm.auto import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM, GPT2Tokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from utils import get_original_samples, get_in_context_prompt, parse_new_samples, process_gsm8k_prediction
import argparse 
from rich import print as pprint
from dataset import GSM8K_dataset
from sklearn.metrics import f1_score, confusion_matrix





parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=['none', 'original', 'sample', 'seed', 'sample_seed'], required=True)
parser.add_argument("--keep", type=int, default=100, help='how many samples from the original dataset to keep')

parser.add_argument("--dataset_path", type=str, help='dataset path in case of args.dataset in [\'sample\', \'original\']')

parser.add_argument("--model", type=str, choices=['qwen', 'gemma', 'llama', 'gptneo'], required=True, help='model to finetune')

parser.add_argument("--output_folder", type=str, default='/data/takis/SynthGen/models')
args = parser.parse_args()

print (args)




################################################### <Set up> ###################################################
torch.manual_seed(0)
np.random.seed(0)

# Get datasets/dataloaders
train_dataset = GSM8K_dataset(args.keep, args.dataset_path if (args.dataset in ['sample', 'seed', 'sample_seed']) else None, 'train')
valid_dataset = GSM8K_dataset(None, None, 'test')

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)


# Get model 
device = "cuda" if torch.cuda.is_available() else "cpu"


if (args.model == 'qwen'):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", cache_dir='/data/takis/models', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", cache_dir='/data/takis/models', torch_dtype=torch.bfloat16).to(device)

    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", cache_dir='/data/takis/models', padding_side='left')
    # tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", cache_dir='/data/takis/models', torch_dtype=torch.bfloat16).to(device)
elif (args.model == 'llama'):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", cache_dir='/data/takis/models', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", cache_dir='/data/takis/models', torch_dtype=torch.bfloat16).to(device)
elif (args.model == 'gptneo'):
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",  cache_dir='/data/takis/models', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B",  cache_dir='/data/takis/models', torch_dtype=torch.bfloat16).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
optim.zero_grad()

print (type(model))
################################################### </Set up> ###################################################



################################################### <Train> ###################################################
best_acc = 0

for epoch in range(5):

    if (args.dataset == 'none'):
        break

    ## Train
    model.train()

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='epoch %d' %(epoch+1))
    for batch_idx, (inputs, outputs) in pbar:



        total_text = [inputs[i] + ' ' + outputs[i] for i in range(len(inputs))]
        total_text_encodings = tokenizer.batch_encode_plus(total_text, return_tensors="pt", padding=True).to(device)
        labels = total_text_encodings.input_ids.clone() 
        
        for i in range(len(inputs)):
            cur_ids = tokenizer.batch_encode_plus([outputs[i]], return_tensors="pt", padding=False).input_ids[0]
            labels[i, :-cur_ids.shape[0]-1] = -100
        
        
        loss = model.forward(**total_text_encodings, labels=labels).loss
        loss.backward()

        if (batch_idx+1 == len(train_dataloader) or (batch_idx+1)%4 == 0):
            optim.step()
            optim.zero_grad()

        pbar.set_description('epoch %d, train loss: %.3f' %(epoch+1, loss.item()))



    ## Eval
    model.eval()

    pred_responses = []
    true_responses = []

    pbar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
    for batch_idx, (inputs, outputs) in pbar:

        with torch.no_grad():
            inputs = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True).to(device)
            true_responses.extend(map(process_gsm8k_prediction, outputs))

            out = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            output_text = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=False)

        pred_responses.extend(map(process_gsm8k_prediction, output_text))

        acc = (np.array(pred_responses) == np.array(true_responses)).sum()/len(pred_responses)
        pbar.set_description('accuracy: %.4f' %(acc))

    # pprint(confusion_matrix(true_responses, pred_responses))

    best_acc = max(best_acc, acc)


## Final eval
model.eval()

pred_responses = []
true_responses = []

pbar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
for batch_idx, (inputs, outputs) in pbar:


    with torch.no_grad():
        inputs = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True).to(device)
        true_responses.extend(map(process_gsm8k_prediction, outputs))

        out = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        output_text = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=False)

    # for i in range(len(output_text)):
    #     pprint (output_text[i])
    #     pprint (list(map(process_gsm8k_prediction, outputs))[i], list(map(process_gsm8k_prediction, output_text))[i], list(map(process_gsm8k_prediction, outputs))[i]==list(map(process_gsm8k_prediction, output_text))[i])



    pred_responses.extend(map(process_gsm8k_prediction, output_text))

    acc = (np.array(pred_responses) == np.array(true_responses)).sum()/len(pred_responses)
    pbar.set_description('accuracy: %.4f' %(acc))


# pprint(confusion_matrix(true_responses, pred_responses))

best_acc = max(best_acc, acc)


print ('\n\n-------------------------------------\n')
if (args.dataset in ['seed', 'sample']):
    print (args.dataset_path)
else:
    print (args.dataset)
print ('Best accuracy: %.4f\n\n' %(best_acc))



