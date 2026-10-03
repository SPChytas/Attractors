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



parser = argparse.ArgumentParser()
parser.add_argument("--strength", type=float, required=True)
parser.add_argument("--target_layer", type=int, default=17)

parser.add_argument("--keep", type=int, default=100, help='how many images to test on')
parser.add_argument("--question", type=str, default='Describe the image.')
parser.add_argument("--max_tokens", type=int, default=512)

parser.add_argument("--dataset_path", type=str, default='/data/takis/datasets/MSCOCO/2017/val2017', help='MSCOCO dataset path')

parser.add_argument("--output_folder", type=str, default='/data/takis/Hallucinations/CHAIR')
args = parser.parse_args()

print (args)





################################################### <Set up> ###################################################
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
processor = vis_processors['eval']

# Image list
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

image_list = os.listdir(args.dataset_path)
image_list = [img for img in image_list if img.endswith('jpg')]
image_sublist = np.random.choice(image_list, size=args.keep, replace=False)

# Prompt
prompt = args.question

# Output directory
experiment_dir = os.path.join(args.output_folder, 'instructblip')
os.makedirs(experiment_dir, exist_ok=True)
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

for img in tqdm(image_sublist, total=len(image_sublist)):

    img_id = int(img.split('.')[0])
    img_path = os.path.join(args.dataset_path, img)
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
            
            predictions.append({'image_id': img_id, 'caption': output})

            
with open(os.path.join(experiment_dir, 'layer_%d_strength_%.3f_tokens_%d_keep_%d.jsonl' %(args.target_layer, args.strength, args.max_tokens, args.keep)), 'w') as f:
    for l in predictions:
        f.write(json.dumps(l) + '\n')

