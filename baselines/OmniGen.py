## test multi-instruction from PIE-Bench++ 140 random.
import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from OmniGen import OmniGenPipeline
from torchvision.transforms.functional import InterpolationMode

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # select a gpu to run OmniGen
# os.environ['HF_HUB_CACHE'] = 'path_to_save_downloaded_model'
print("OK")
seeds = [42]
pipe = OmniGenPipeline.from_pretrained("/model_dirs/models--Shitao--OmniGen-v1") # or huggingface.


json_names = ['three-chain', 'COCO-obj-attr-global', 'COCO-three-obj', 'COCO-two-obj-one-attr', 'two-chain']  # 'two-object-one-attr'
for json_name in json_names:

    print(f"Start {json_name}...")
    try:
        with open(f'./data/instructions/{json_name}/final_update_v2.json', 'r') as f:
            data = json.load(f)
    except:
        continue
    ### prepare the prompt and image.
    prompts = []  #  319.
    input_images, img_ids = [], []
    for key, value in data.items():
        prompt = "<img><|image_1|><img>\n" + ",".join(value['new_ins'].split('\r\n'))+"."  # 这里的ins是list，所以需要拼接为字符串先。
        prompts.append(prompt)
        # img_dir = os.path.join('./eval/OIR_Bench', value['image_path'])
        img_dir = f"./data/more-object-no-multi3/{key}-{data[key]['img_path']}"
        input_images.append([img_dir])
        img_ids.append(key)

    '''处理image'''
    for img in input_images:
        image = img[0]

    height, width = Image.open(input_images[0][0]).convert('RGB').size
    print(f"image size is: {height}, {width}")  # 1024, 1024.


    for seed in seeds:
        for i in range(len(prompts)):
            print(f"Processing {i}th image...")
            ## 判断当前image是否已经被处理过了.
            if img_ids[i] in os.listdir(f"./edited-image/omnigen/{json_name}/testResults_{seed}"):
                print(f"skip {img_ids[i]}...")
                continue
            images = pipe(
                prompt=prompts[i], 
                input_images=input_images[i],
                height=512, # 1024.
                width=512,
                guidance_scale=2.5, 
                img_guidance_scale=1.6,
                seed=seed)  # 0.
            # img_dir = f"./eval/OIR_Bench/testResults_{seed}/{img_ids[i]}"

            img_dir = f"./edited-image/omnigen/{json_name}/testResults_{seed}/{img_ids[i]}"
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            images[0].save(os.path.join(img_dir, f"{img_ids[i]}_1.png"))

                # Image.open(input_images[i][0]).convert('RGB').save(os.path.join(img_dir, f"{input_images[i][0].split('/')[-1]}"))  # 以防万一把gt也保存一次把.

print('finished.')