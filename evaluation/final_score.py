import json
from pathlib import Path

def collect_edit_type_scores(folder_paths):
    type_score_sum = {}  # 累积每种类型得分
    type_count = {}  # 累积每种类型出现次数

    for folder_path in folder_paths:
        folder = Path(folder_path)
        if not folder.is_dir():
            continue
        for item in folder.iterdir():
            # if not subfolder.is_dir():
            #     continue
            # eval_json = subfolder / 'eval_results.json'
            # if not eval_json.is_file():
            #     continue
            if item.is_file() and item.name.endswith('eval_results.json'):
                eval_json = item

            with open(eval_json, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)
            for res in eval_results:
                if 'edit type' in res and 'final_score' in res:
                    try:
                        score = float(res['final_score'])
                        etype = res['edit type']
                        type_score_sum[etype] = type_score_sum.get(etype, 0) + score
                        type_count[etype] = type_count.get(etype, 0) + 1
                    except Exception:
                        continue

    # 计算平均分
    type_avg_score = {}
    for etype in type_score_sum:
        count = type_count[etype]
        avg = type_score_sum[etype] / (count * 5)
        type_avg_score[etype] = avg

    return type_avg_score

if __name__ == "__main__":

    folders = [
        r'edited-image\Gemini\COCO-obj-attr-global\testResults_42_eval_v3_thinking_01_21',
        r'edited-image\Gemini\COCO-three-obj\testResults_42_eval_v3_thinking_01_21',
        r'edited-image\Gemini\COCO-two-obj-one-attr\testResults_42_eval_v3_thinking_01_21',
        r'edited-image\Gemini\three-chain\testResults_42_eval_v3_thinking_01_21',
        r'edited-image\Gemini\two-chain\testResults_42_eval_v3_thinking_01_21'
    ]

    avg_scores = collect_edit_type_scores(folders)
    print("每个编辑类型的平均得分：")
    for etype, avg in avg_scores.items():
        print(f"{etype}: {avg:.4f}")