import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from diffusers import FluxFillPipeline
import requests
import PIL
import json
from PIL import Image
import numpy as np

seeds = [42]
model_id = '/mnt/disk/ModelHub/stable-diffusion/flux.1-fill-dev'
lora_path = '/mnt/disk/ModelHub/stable-diffusion/normal-lora'
pipe = FluxFillPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(lora_path)
pipe = pipe.to("cuda")

'''Subset'''
json_names = ['three-chain', 'COCO-obj-attr-global', 'COCO-three-obj', 'COCO-two-obj-one-attr', 'two-chain']

for json_name in json_names:
    with open(f'/home/clwang/wcls/hubs/editing/MagicBrush/Subset/{json_name}/final_update_v2.json', 'r') as f:
        data = json.load(f)
    
    prompts = []  #  319.
    input_images, gt_images, img_ids = [], [], []
    for key, value in data.items():
        prompt = ",".join(value['new_ins'].split('\r\n'))+"." 
        prompts.append(prompt)
        img_dir = f"/home/clwang/wcls/hubs/editing/COCO/no_multi3/{key}-{data[key]['img_path']}"
        input_images.append(img_dir)
        gt_images.append(f"/home/clwang/wcls/hubs/editing/COCO/no_multi3/{key}-{data[key]['img_path']}")  # str. Not list, cause gt.
        img_ids.append(key)

    height, width = PIL.Image.open(gt_images[0]).convert('RGB').size
    print(f"image size is: {height}, {width}") 

    # generate images.
    for seed in seeds:
        generator = torch.Generator('cuda').manual_seed(seed)
        for i, (prompt, url) in enumerate(zip(prompts, input_images)):
            ## 判断当前image是否被处理过了.
            if os.path.exists(f'/home/clwang/wcls/hubs/editing/MagicBrush/baselines/edited-image/icedit/{json_name}/testResults_{seed}'):
                if img_ids[i] in os.listdir(f"/home/clwang/wcls/hubs/editing/MagicBrush/baselines/edited-image/icedit/{json_name}/testResults_{seed}"):
                    print(f"skip {img_ids[i]} because it has been processed.")
                    continue
            image = PIL.Image.open(url).convert("RGB")
            image = image.resize((512, 512))

            instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {prompt}'

            width, height = image.size
            combined_image = Image.new("RGB", (width * 2, height))
            combined_image.paste(image, (0, 0))
            combined_image.paste(image, (width, 0))
            mask_array = np.zeros((height, width * 2), dtype=np.uint8)
            mask_array[:, width:] = 255 
            mask = Image.fromarray(mask_array)

            result_image = pipe(
                prompt=instruction,
                image=combined_image,
                mask_image=mask,
                height=height,
                width=width * 2,
                guidance_scale=50,
                num_inference_steps=28,
                generator=generator,
            ).images[0]

            result_image = result_image.crop((width,0,width*2,height))
            
            img_dir = f"/home/clwang/wcls/hubs/editing/MagicBrush/baselines/edited-image/icedit/{json_name}/testResults_{seed}/{img_ids[i]}"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)

            result_image.save(os.path.join(img_dir, f"{img_ids[i]}_1.png"))