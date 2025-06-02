from google import genai
from google.genai import types
import PIL
import json
import os
from io import BytesIO


api_key = 'YOUR API KEY'
client = genai.Client(api_key=api_key)

seeds= [42]
import time

def is_time_out(limit, s_time):
    # global start_time  # Declare start_time as a global variable
    if limit <= 0:
        elapsed_time = time.time() - s_time
        if elapsed_time < 61:
            remaining_time = 61 - elapsed_time
            print("waiting...")
            time.sleep(remaining_time)
            print("restart.")
            limit = 10
            s_time = time.time()
        else:  # 如果使用时间超过1min,直接重置即可.
            limit = 10
            s_time = time.time()
            print("No wait, restart!")

    return limit, s_time


dir_name = r'.\data\more-object-no-multi3'
json_names = ['three-chain', 'COCO-obj-attr-global', 'COCO-three-obj', 'COCO-two-obj-one-attr', 'two-chain']  # 'two-object-one-attr'


limits = 10
start_time = time.time()

error_lists = []
for seed in seeds:
    for json_name in json_names:
        file_name = os.path.join(r'data\instructions', json_name, 'final_update_v2.json' )

        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ## 判断当前json name的image是否被处理过了，只判断是否处理了当前json
        dirs_name = f'./cot-gemini/{json_name}/testResults_{seed}'
        if not os.path.exists(dirs_name):  # 不存在就创建一下文件夹
            os.makedirs(dirs_name)
        if len(os.listdir(f'./cot-gemini/{json_name}/testResults_{seed}')) == len(data):
            print(f"Skip {json_name}...")
            continue
        else:
            print(f"Start {json_name}...")

            ### prepare the prompt and image.

            for key, value in data.items():
                if key in os.listdir(f'./cot-gemini/{json_name}/testResults_{seed}'):  # 判断当前json name下的key还有哪些每处理
                    print(f"Skip {key}...")
                    continue

                ins = ",".join(data[key]['new_ins'].split('\r\n')) + '.'
                
                # 将thinking内容拼接到指令前面
                with open(os.path.join(f"./Gemini-thought/{json_name}/testResults_{seed}/{key}", f'{key}_1.txt'), 'r', encoding='utf-8') as f:
                    try:
                        thought = f.read().strip()
                    except Exception as e:
                        print(f"读取{key}_1.txt出错: {e}")
                        exit()
                
                ins = ins + thought
                img = PIL.Image.open(os.path.join(dir_name, key + '-' + data[key]['img_path']))
                # img_dir = os.path.join('./eval/OIR_Bench', value['image_path'])
                
                num = 0 
                while True:
                    try:
                        response = client.models.generate_content(
                            model="models/gemini-2.0-flash-exp-image-generation",
                            contents = [ins, img],
                            config=types.GenerateContentConfig(
                                response_modalities=['Text', 'Image']
                            )
                        )
                        time.sleep(4)
                        for part in response.candidates[0].content.parts:
                            if part.text is not None:
                                print(part.text)
                            if part.inline_data is not None:
                                image = PIL.Image.open(BytesIO(part.inline_data.data))
                            
                            # time.sleep(4)
                            limits -= 1
                            limits, start_time = is_time_out(limits, start_time)
                        
                            img_dir = f"./cot-gemini/{json_name}/testResults_{seed}/{key}"

                            if not os.path.exists(img_dir):
                                os.makedirs(img_dir)
                            image.save(os.path.join(img_dir, f"{key}_1.png"))
                        break
                    except Exception as e:
                        print(e, f"continue:{key}, prompt is :{ins}")
                        num += 1
                        if num == 3:
                            error_lists.append(key)
                            break
                        continue


print(error_lists)
print('finished.')