import os
import json
from pathlib import Path
import numpy as np

def main(results_base_dir):
    results_base_dir = Path(results_base_dir)

    all_score, num = 0, 0
    # --- 判断 prompt_type ---
    prompt_type = None
    if 'two-chain' in results_base_dir.parent.name:
        print('two-chain')
        prompt_type = 'two-chain'
    elif 'three-chain' in results_base_dir.parent.name:
        print('three-chain')
        prompt_type = 'three-chain'
    # 遍历所有子文件夹
    for item in results_base_dir.iterdir():
        # if item.is_dir():
        #     eval_json_path = item / 'eval_results.json'
        #     if not eval_json_path.is_file():
        #         continue
        #     with open(eval_json_path, 'r', encoding='utf-8') as f:
        #         eval_results = json.load(f)
        if item.is_file() and item.name.endswith('eval_results.json'): # 直接读取当前文件夹.
            eval_json_path = item
            with open(eval_json_path, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)

            scores = []
            edit_types = []
            avg_score_dict = {}
            for res in eval_results:
                # 跳过异常项
                if 'final_score' not in res or 'edit type' not in res:
                    continue
                try:
                    
                    score = float(res['final_score'])
            
                except Exception:
                    continue
                scores.append(score)
                edit_types.append(res['edit type'])
            if scores and edit_types:
                if prompt_type == 'two-chain': 
                    try:
                        avg_score = ((scores[0]/5.0) * (scores[1]/5.0) + scores[2]/5.0) / 2
                    except Exception as e:
                        print(e, item)


                elif prompt_type == 'three-chain': 
                    try:
                        avg_score = (scores[0]/5.0) * (scores[1]/5.0) * (scores[2]/5.0)
                    except Exception as e:
                        print(e, item)


                else:
                    avg_score = sum(scores) / (len(scores)*5.0)
               
                key = ','.join(edit_types)
                avg_score_dict[key] = avg_score
            # 避免重复append, 如果有直接覆盖掉值即可
                already_exists = False
                for idx, item in enumerate(eval_results):
                    if isinstance(item, dict) and key in item:
                        eval_results[idx] = avg_score_dict  # 覆盖
                        already_exists = True
                        break
                if not already_exists:
                    eval_results.append(avg_score_dict)

                all_score += avg_score
                num += 1
    # print(avg_score_dict)
        # 如需保存到文件，重新写回去.
            with open(eval_json_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    avg_score_path = results_base_dir.parent / 'all_score.txt'
    with open(avg_score_path, 'w', encoding='utf-8') as f:
        f.write(f'all score: {all_score / num}\n')
    
    print(f"评估目录: {results_base_dir.parts[-2]}; \t 平均得分: {all_score / num:.4f};\t 个数：{num}")

    print("="*40)


if __name__ == "__main__":
    '''修改为你的结果文件夹路径'''
    '''统计文件夹下面的所有结果,得到一个txt总分'''

    results_folders = [
        r'edited-image\Gemini\COCO-obj-attr-global\testResults_42_eval_v3_thinking_01_21',
        r'edited-image\Gemini\COCO-three-obj\testResults_42_eval_v3_thinking_01_21',
        r'edited-image\Gemini\COCO-two-obj-one-attr\testResults_42_eval_v3_thinking_01_21',
        r'edited-image\Gemini\three-chain\testResults_42_eval_v3_thinking_01_21',
        r'edited-image\Gemini\two-chain\testResults_42_eval_v3_thinking_01_21'
    ]


    for results_folder in results_folders:
        main(results_folder)