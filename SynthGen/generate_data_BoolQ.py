from datasets import load_dataset
from tqdm.auto import tqdm
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from utils import get_original_samples, get_in_context_prompt, parse_new_samples
import argparse 
from rich import print as pprint






################################################### <Hooks> ###################################################
activations = {}

def get_activation(name):
    global activations
    def hook(model, input, output):
        global activations
        if (name in activations):
            activations[name].append(output[0])
        else:
            activations[name] = [output[0]]
    return hook


def add_noise(target_layer):
    def hook(model, input, output):
        output = (output[0] + noise.to(device), *output[1:])
        return output
    return hook


def change_behavior(target_layer):
    def hook(model, input, output):
        if (abs(strength) > 1e-5):
            output = (output[0] + (strength*concepts[prompt_id][target_layer]/torch.linalg.norm(concepts[prompt_id][target_layer]) + noise).to(device), *output[1:])
        return output
    return hook
################################################### </Hooks> ###################################################




parser = argparse.ArgumentParser()
parser.add_argument("--generation", type=str, choices=['sample', 'seed', 'sample_seed'], required=True)
parser.add_argument("--keep", type=int, default=100, help='how many samples from the original dataset to keep')
parser.add_argument("--reps", type=int, default=3, help='each generation rep generates 3 new prompts. Determine how many generation reps to run for each sample')

parser.add_argument("--target_layer", type=int, default=17, help='layer in which to add the concept vector')
parser.add_argument("--denom", type=float, default=100, help='noise deviation control')

parser.add_argument("--topK", type=int, default=50, help='generation top_k')
parser.add_argument("--topP", type=float, default=0.99, help='generation top_p')
parser.add_argument("--temp", type=float, default=1.0, help='generation temperature')

parser.add_argument("--output_folder", type=str, default='/data/takis/SynthGen/datasets')
args = parser.parse_args()

print (args)

################################################### <Set up> ###################################################
# Get original dataset
original_prompts = get_original_samples('BoolQ', 'train')[:args.keep]
in_context_prompt = get_in_context_prompt('BoolQ')

# Get model to generate new samples
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir='/data/takis/models')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir='/data/takis/models').to(device)
model = model.eval()
################################################### </Set up> ###################################################


################################################### <Concept vectors> ###################################################
# Get concept vectors
if (args.generation == 'concept'):
    
    ## set forward hooks
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(get_activation(i)))


    ## get intermediate activations
    concepts = []
    activations = {}

    pbar = tqdm(enumerate(original_prompts), total=len(original_prompts), desc='calculating concept vectors...')
    for i, p in pbar:
        
        activations = {}

        with torch.no_grad():
            inputs = tokenizer(p, return_tensors="pt").input_ids
            outputs = model.forward(inputs.to(device))
            
        for k in activations.keys():
            activations[k] = torch.cat(activations[k], dim=1).detach().cpu()
            activations[k] =  torch.mean(activations[k], dim=-2)

        concepts.append(copy.deepcopy(activations))
    
    activations = {}


    for hook in hooks:
        hook.remove()
    hooks = []
################################################### </Concept vectors> ###################################################



# Set steering hook
strength = args.strength if (args.generation == 'concept') else 0
noise = 0


if (args.generation == 'seed' or args.generation == 'sample_seed'):
    print ('seed: adding forward hook...')
    hook = model.model.layers[args.target_layer].self_attn.register_forward_hook(add_noise(args.target_layer))
# hook = model.model.layers[args.target_layer].self_attn.register_forward_hook(change_behavior(args.target_layer))



################################################### <Synthetic data generation> ###################################################
# Generate new samples
generated_texts = []
do_sample = True if args.generation in ['in-context', 'sample', 'sample_seed'] else False

for prompt_id, prompt in tqdm(enumerate(original_prompts), total=len(original_prompts), desc='getting new samples...'):

    if (args.generation != 'concept'):
        aug_prompt = prompt + '\n' + in_context_prompt
    else:
        aug_prompt = in_context_prompt

    # pprint (aug_prompt)

    for _ in range(args.reps):
        
        ## add noise in the intermediate layer
        # noise = torch.randn(concepts[prompt_id][args.target_layer].shape)/20
        noise = torch.randn((4096,))/args.denom

        ## get the generated text
        with torch.no_grad():
            inputs = tokenizer(aug_prompt, return_tensors="pt").input_ids
            outputs = model.generate(inputs.to(device), max_new_tokens=400, do_sample=do_sample, 
                                     top_k=args.topK, top_p=args.topP, temperature=args.temp,
                                     pad_token_id=tokenizer.eos_token_id, use_cache=True).detach().cpu()
        
        output_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
        generated_texts.append(output_text.strip())

        # pprint (prompt_id, output_text)


# Process text
synthetic_dataset = parse_new_samples(generated_texts, 'BoolQ')


os.makedirs(os.path.join(args.output_folder, 'BoolQ_%d' %(args.keep), args.generation), exist_ok=True)

if (args.generation in ['in-context', 'sample']):
    synthetic_dataset.save_to_disk(os.path.join(args.output_folder, 
                                                'BoolQ_%d' %(args.keep), 
                                                args.generation,
                                                'topK_%d_topP_%.3f_temp_%.3f_reps_%d.hf' %(args.topK, 
                                                                                           args.topP, 
                                                                                           args.temp, 
                                                                                           args.reps)))
elif (args.generation in ['seed']):
    synthetic_dataset.save_to_disk(os.path.join(args.output_folder, 
                                                'BoolQ_%d' %(args.keep), 
                                                args.generation,
                                                'layer_%d_denom_%.2f_reps_%d.hf' %(args.target_layer, 
                                                                                   args.denom, 
                                                                                   args.reps)))
elif (args.generation in ['sample_seed']):
    synthetic_dataset.save_to_disk(os.path.join(args.output_folder, 
                                                'BoolQ_%d' %(args.keep), 
                                                args.generation,
                                                'topK_%d_topP_%.3f_temp_%.3f_layer_%d_denom_%.2f_reps_%d.hf' %(args.topK, 
                                                                                                               args.topP, 
                                                                                                               args.temp,args.target_layer, 
                                                                                                               args.denom, 
                                                                                                               args.reps)))
################################################### </Synthetic data generation> ###################################################

























































