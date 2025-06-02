from google import genai
from google.genai import types
import argparse
import os
from pathlib import Path
import mimetypes # 用于猜测文件类型
import json
import re
import time
from tqdm import trange
# --- Gemini API ---
api_key = 'YOUR Gemini API KEY'
try:
    client = genai.Client(api_key=api_key)
except KeyError:
    print("错误：请设置 API_KEY 环境变量。")
    exit(1)
except Exception as e:
    print(f"配置 Gemini API 时出错: {e}")
    exit(1)

# --- 全局变量 ---
# 基础路径
# 获取当前脚本的绝对路径
BASE_PROJECT_PATH = Path(__file__).resolve()
# 项目根目录 (evaluation目录的上两级)

EVAL_PROMPT_DIR = BASE_PROJECT_PATH / "evaluation" / "eval_prompt" 
ORIGINAL_IMAGE_BASE_PATH = BASE_PROJECT_PATH / "data" / "more-object-no-multi3" 

# 支持的图片扩展名
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

# --- 函数定义 ---

def load_prompt(prompt_type: str) -> str:
    """根据类型加载指定的提示文件内容"""
    if prompt_type == 'chain two':
        prompt_type = 'add object'

    prompt_file_path = EVAL_PROMPT_DIR / f"{prompt_type}.txt"
    if not prompt_file_path.is_file():
        raise FileNotFoundError(f"提示文件未找到: {prompt_file_path}")
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise IOError(f"读取提示文件时出错 {prompt_file_path}: {e}")

def evaluate_image_with_gemini(prompt: str, instruction: str, image_path: Path, ori_image_path: Path, all_ins: list):
    """使用 Gemini API 评估单个指令和图片"""
    print(f"\n--- 正在评估图片: {image_path.name} ---")
    print(f"--- 使用指令: {instruction} ---")

    max_retries = 5
    retry_delay_seconds = 5
    
    for attempt in range(max_retries):

        try:
            # 检查文件是否存在且可读
            if not image_path.is_file():
                print(f"错误：图片文件未找到或无法访问: {image_path}")
                return None

            # 猜测 MIME 类型
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith("image/"):
                print(f"警告：无法确定图片文件的有效 MIME 类型: {image_path}. 将尝试作为 application/octet-stream 上传。")
                # 可以选择一个默认的或者跳过，这里我们尝试用通用类型
                # mime_type = "application/octet-stream"
                # 或者直接返回错误
                print(f"错误：不支持的文件类型 {mime_type} for {image_path}")
                return None


            # 上传文件到 Gemini API (如果需要，API 会处理)
            # 对于本地文件，可以直接传递路径或字节流
            '''读取原始图像和编辑后图像'''
            uploaded_file = client.files.upload(file=ori_image_path)
            with open(image_path, 'rb') as f:
                img2_bytes = f.read()  # 编辑后图像

            
            # 结合基础提示、单条指令和图片
            ins_dict = {"Editing Instructions": all_ins, "Instruction To Evaluate": instruction}
            full_prompt = f"{ins_dict}\n\n{prompt}"

            # Create the prompt with text and multiple images
            response = client.models.generate_content(
                model="models/gemini-2.0-flash-thinking-exp-01-21", # models/gemini-2.0-flash-thinking-exp-1219  models/gemini-2.0-flash-thinking-exp-01-21
                # models/gemini-2.0-pro-exp-02-05    models/gemini-2.5-pro-exp-03-25
                contents=[
                    full_prompt,  # 传递prompt;
                    uploaded_file,  # Use the uploaded file reference
                    types.Part.from_bytes(
                        data=img2_bytes,
                        mime_type='image/png'
                    )
                ]
            )

            # 打印结果
            print("--- Gemini 评估结果 ---")
            print(response.text)
            print("------------------------")
            return response.text

        except FileNotFoundError:
            print(f"错误：图片文件未找到: {image_path}")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"{e}, 将在 {retry_delay_seconds} 秒后重试 (尝试 {attempt + 2}/{max_retries})...")
                time.sleep(retry_delay_seconds)
            else:
                print(f"已达到最大重试次数 ({max_retries})，评估失败。正在退出程序...")
                exit()
    return None

def process_images(prompt_type: str, results_base_dir_path: Path, json_instructions_path: Path, output_dir_path: Path): # 参数名和类型修改
    """处理指定基础结果文件夹中的所有图片子文件夹"""

    # 确保输出目录存在
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"评估结果将保存到: {output_dir_path}")

    # 检查并加载 JSON 指令文件
    if not json_instructions_path.is_file():
        print(f"错误：JSON 指令文件未找到: {json_instructions_path}")
        return
    try:
        with open(json_instructions_path, 'r', encoding='utf-8') as f:
            instructions_data = json.load(f)
        print(f"已加载指令文件: {json_instructions_path.name}")
    except json.JSONDecodeError:
        print(f"错误：无法解析 JSON 文件: {json_instructions_path}")
        return
    except Exception as e:
        print(f"读取 JSON 文件时出错 {json_instructions_path}: {e}")
        return

    # 检查基础结果文件夹是否存在
    if not results_base_dir_path.is_dir():
        print(f"错误：结果文件夹未找到或不是一个目录: {results_base_dir_path}")
        return

    print(f"开始处理基础结果文件夹中的子文件夹: {results_base_dir_path}")

    # 统计需要处理的文件夹（未评估过的）
    all_folders = [item for item in results_base_dir_path.iterdir() if item.is_dir()]
    folders_to_process = []
    for item in all_folders:
        eval_results_path = output_dir_path / f"{item.name}_eval_results.json"
        if not eval_results_path.exists():
            folders_to_process.append(item)
    total = len(folders_to_process)
    idx = 0

    # 遍历基础结果文件夹下的所有项目
    for item in results_base_dir_path.iterdir():
        # 检查是否是子文件夹 (以 JSON key 命名)
        if item.is_dir(): 
            image_subfolder_path = item
            folder_key = image_subfolder_path.name # 子文件夹名-(image-id), 即为 JSON key

            print(f"\n>>> 正在处理第 {idx}/{total} 个图片文件夹，还剩 {total - idx} 个待处理 <<<")
            # 检查 key 是否存在于 JSON 数据中
            if folder_key not in instructions_data:
                print(f"警告：在 JSON 文件中未找到 key '{folder_key}' 对应的指令，跳过文件夹。")
                continue
            
            '''如果重新运行的话需要把已经有的评估都删掉.'''
            # 检查是否评估过了
            eval_results_path = output_dir_path / f'{folder_key}_eval_results.json'
            if eval_results_path.exists():
                print(f"跳过 '{folder_key}'，已存在评估结果文件: {eval_results_path.name}")
                continue

            # 在子文件夹中查找图片文件
            image_path = None
            found_images = []  # 文件夹下只有一张edited image.[考虑多张？]
            for sub_item in image_subfolder_path.iterdir():  # 遍历子文件夹下的所有文件， 带路径.
                if sub_item.is_file() and sub_item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    found_images.append(sub_item)

            if not found_images:
                print(f"警告：在文件夹 '{image_subfolder_path.name}' 中未找到支持的图片文件，跳过。")
                continue
            elif len(found_images) > 1:
                print(f"警告：在文件夹 '{image_subfolder_path.name}' 中找到多个图片文件，将使用第一个找到的: {found_images[0].name}")
                image_path = found_images[0]
            else:
                image_path = found_images[0]  # windowsPath, 带路径的, 用name取最后的名字

            # 从 JSON 获取并处理指令
            try:
                # 获取 "new_ins" 字段
                if "new_ins" not in instructions_data[folder_key]:
                     print(f"警告：在 JSON key '{folder_key}' 下未找到 'new_ins' 字段，跳过文件夹。")
                     continue

                multiple_instructions = instructions_data[folder_key]["new_ins"]  # 读取多指令。

                if not isinstance(multiple_instructions, str) or not multiple_instructions.strip():
                     print(f"警告：JSON key '{folder_key}' 的 'new_ins' 字段为空或不是字符串，跳过。")
                     continue

                # 按行分割指令，并去除空行，最多取前三条
                single_instructions = [line.strip() for line in multiple_instructions.splitlines() if line.strip()]

                if not single_instructions:
                    print(f"警告：从 JSON key '{folder_key}' 的 'new_ins' 解析后未找到有效指令，跳过图片 '{image_path.name}'。")
                    continue

                # 只取前三条指令进行评估
                instructions_to_evaluate = single_instructions[:3]
                if len(single_instructions) > 3:
                     print(f"注意：JSON key '{folder_key}' 包含超过3条指令，仅评估前3条。")

                print(f"\n找到 {len(instructions_to_evaluate)} 条指令用于图片: {image_path.name} (来自 key: {folder_key})")
                
                # TODO: 读取edit pair, 包括每条instruction的type以及editing前后的object. 定义每个类型prompt的模板;
                multiple_pairs = instructions_data[folder_key]["edit pair"]
                ori_img_path_name =  instructions_data[folder_key]['img_path']
                ori_img_path = ORIGINAL_IMAGE_BASE_PATH / f"{folder_key}-{ori_img_path_name}"

                ## 定义一个变量存储评估结果用, 多加一个type字段
                results_list = []
                for i in trange(len(multiple_pairs)):
                    print(f"pair {i+1}: {multiple_pairs[i]}")
                    
                    if i > 0:
                        time.sleep(4)  # 控制速率
                    
                    base_prompt = load_prompt(multiple_pairs[i]['edit_type'])
                    ori_obj = multiple_pairs[i]['ori obj']
                    edit_obj = multiple_pairs[i]['new_obj']

                    res = evaluate_image_with_gemini(base_prompt, instructions_to_evaluate[i], image_path, ori_img_path, instructions_to_evaluate)

                    # TODO: 处理结果,保存并统计分数.;
                    try:
                        match = re.search("```json(.*?)```", res, re.DOTALL)

                        res_json = json.loads(match.group(1).strip())  # select the first.
                        res_json['edit type'] = multiple_pairs[i]['edit_type']
                        res_json['instruction'] = instructions_to_evaluate[i]
                        results_list.append(res_json)
                    except Exception as e:
                        print(f"处理 key '{folder_key}' (图片: '{image_path.name}') 时出错: {e}")
                        results_list.append({'error': str(e), 'final_score': "0", "edit type": multiple_pairs[i]['edit_type'], "instruction": instructions_to_evaluate[i]})
                # 将当前image的评估结果保存到当前editing image文件夹下
                
                with open(eval_results_path, 'w', encoding='utf-8') as f:
                    json.dump(results_list, f, ensure_ascii=False, indent=2)
                print(f"评估结果已保存到: {eval_results_path}")
                
                idx += 1  # Next.

            except KeyError as e:
                 print(f"处理 JSON 数据 key '{folder_key}' 时出错：缺少键 {e}，跳过。")
            except Exception as e:
                print(f"处理 key '{folder_key}' (图片: '{image_path.name}') 时出错: {e}")

    print("\n所有子文件夹处理完毕。")


# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Gemini API 评估图片编辑效果。")

    # 获取所有可用的 prompt 类型
    available_prompts = [p.stem for p in EVAL_PROMPT_DIR.glob("*.txt")]  # stem 用于获取文件名（不含扩展名）

    parser.add_argument(
        "--type",  #作为parallel, two-chain  three-chain的区分把
        default='parallel',
        choices=available_prompts, # 限制选择为存在的文件名（不含扩展名） 
        help=f"选择评估使用的提示类型。可用选项: {', '.join(available_prompts)}"
    )
    parser.add_argument(
        "--results_folder",
        required=True,
        help="包含以 JSON key 命名的子文件夹的基础结果文件夹路径，每个子文件夹内含一张编辑后的图片。"
    )
    parser.add_argument(
        "--json_path",
        required=True,
        help=f"包含指令 ('new_ins' 字段) 的 JSON 文件路径"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="输出文件夹路径，用于保存评估结果。"
    )

    args = parser.parse_args()

    # 将相对路径或绝对路径转换为 Path 对象
    results_folder = Path(args.results_folder)
    json_path = Path(args.json_path)
    output_dir = Path(args.output_dir)

    # 检查路径是否为绝对路径
    if not results_folder.is_absolute():
            print(f"提供的结果文件夹路径 '{args.results_folder}' 不是绝对路径。将相对于当前工作目录解析: '{results_folder.resolve()}'")
    
    if not json_path.is_absolute():
            print(f"提供的 JSON 文件路径 '{args.json_path}' 不是绝对路径。将相对于当前工作目录解析: '{json_path.resolve()}'")

    if not output_dir.is_absolute():
            print(f"提供的输出文件夹路径 '{args.output_dir}' 不是绝对路径。将相对于当前工作目录解析: '{output_dir.resolve()}'")

    process_images(args.type, results_folder, json_path, output_dir)