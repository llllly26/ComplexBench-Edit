import json
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# --- 请将下面的路径替换成你的路径 ---
JSON_PARENT_DIR = './evaluation/L1L2/output'
ORIGINAL_IMAGE_PARENT_DIR = './data/more-object-no-multi3'
EDITED_IMAGE_BASE_DIR = './edited-image/Gemini'

def create_mask_from_bboxes(image_size, bboxes):
    """
    根据边界框列表为给定尺寸的图像创建掩码。
    掩码中，边界框外的区域为1，边界框内的区域为0。
    """
    h, w = image_size
    # 初始掩码全为1 (保留所有像素)
    mask = torch.ones((1, h, w), dtype=torch.float32)
    for bbox in bboxes:
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                # 确保坐标是整数且在图像范围内
                x1 = int(round(max(0, bbox[0])))
                y1 = int(round(max(0, bbox[1])))
                x2 = int(round(min(w, bbox[2])))
                y2 = int(round(min(h, bbox[3])))

                # 确保 x1 < x2 and y1 < y2
                if x1 < x2 and y1 < y2:
                    # 将边界框内的区域设置为0 (排除这些像素)
                    mask[:, y1:y2, x1:x2] = 0
            except (ValueError, TypeError) as e:
                print(f"Skipping invalid bbox {bbox}: {e}")
        # else: # "nothing" or invalid format, ignore
        #     pass
    return mask

def eval_masked_distance(image_pairs_with_bboxes, metric='l1', device='cpu'):
    """
    使用pytorch评估L1或L2距离，忽略指定边界框内的区域。
    """
    if metric == 'l1':
        # 使用 reduction='none' 获取元素级别的损失
        criterion = nn.L1Loss(reduction='none')
    elif metric == 'l2':
        criterion = nn.MSELoss(reduction='none')
    else:
        raise ValueError("Metric must be 'l1' or 'l2'")

    eval_score = 0
    total_valid_pairs = 0

    for item in tqdm(image_pairs_with_bboxes):
        gen_img_path = item['gen_img_path']
        gt_img_path = item['gt_img_path']
        bboxes = item['bboxes']

        try:
            gen_img = Image.open(gen_img_path).convert('RGB')
            gt_img = Image.open(gt_img_path).convert('RGB')

            # 调整生成图像大小以匹配GT图像
            gen_img = gen_img.resize(gt_img.size)

            # 转换为张量
            transform = transforms.ToTensor()
            gen_img_tensor = transform(gen_img).to(device)
            gt_img_tensor = transform(gt_img).to(device)

            # 创建掩码 (边界框外为1，内为0)
            # TODO: 这里掩码只计算了gt-image的object的bbox, edited image的也可以考虑计算.
            mask = create_mask_from_bboxes(gt_img.size[::-1], bboxes).to(device) # PIL size is (w, h), need (h, w)

            # 计算元素级别的损失
            loss_map = criterion(gen_img_tensor, gt_img_tensor)

            # 应用掩码 (只保留边界框外的损失)
            # loss_map 已经是 3 通道，mask 是 1 通道，在定义的时候定义了1通道了。不用额外广播
            masked_loss_map = loss_map * mask

            # 计算掩码区域外的总损失 (所有通道)
            total_loss = torch.sum(masked_loss_map)

            # 计算掩码区域外的像素数量 (乘以通道数)
            # mask 是 [1, H, W], gen_img_tensor 是 [C, H, W]
            num_channels = gen_img_tensor.shape[0]
            num_unmasked_pixels = torch.sum(mask) * num_channels

            if num_unmasked_pixels > 0:
                per_score = total_loss / num_unmasked_pixels
                eval_score += per_score.detach().cpu().numpy().item()
                total_valid_pairs += 1
            else:
                # 如果掩码覆盖了整个图像，则计算此图像对的整体的得分.
                # print(f"Warning: Mask covers the entire image for pair {os.path.basename(gt_img_path)}. calculating full image loss.")
                # 计算整张图的总损失, 不做MASK
                full_image_total_loss = torch.sum(loss_map)
                # 计算整张图的总像素数
                total_pixels_in_image = gt_img_tensor.numel() # 返回元素个数: C * H * W
                if total_pixels_in_image > 0:
                    per_score = full_image_total_loss / total_pixels_in_image
                    eval_score += per_score.detach().cpu().numpy().item()
                    total_valid_pairs += 1
                else:
                    print(f"Warning: Image {os.path.basename(gt_img_path)} has zero pixels. Skipping score calculation.")


        except FileNotFoundError:  # Gemini 可能针对某些指令的编辑，无法实现，导致有些editing 图像可能不存在
            print(f"Warning: Image file not found for pair involving {os.path.basename(gt_img_path)}. Skipping.")
        except Exception as e:
            print(f"Error processing pair involving {os.path.basename(gt_img_path)}: {e}")

    if total_valid_pairs == 0:
        return 0.0
    return eval_score / total_valid_pairs  # 整个文件夹下, 所有图像的L1/L2结果.

def main():
    # 检查父目录是否已设置
    if JSON_PARENT_DIR == '/path/to/your/original/images' or \
       EDITED_IMAGE_BASE_DIR == '/path/to/your/edited/images':
        print("错误：请在脚本中设置 ORIGINAL_IMAGE_PARENT_DIR 和 EDITED_IMAGE_PARENT_DIR 的实际路径。")
        return

    all_image_pairs_with_bboxes = []
    json_subdirs = [d for d in os.listdir(JSON_PARENT_DIR) if os.path.isdir(os.path.join(JSON_PARENT_DIR, d))]
    
    for json_subdir_name in json_subdirs: # 依次遍历处理.
        current_json_file_path = os.path.join(JSON_PARENT_DIR, json_subdir_name, 'updated_final_update_v2.json') 

        # 加载JSON数据
        try:
            with open(current_json_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：JSON 文件未找到于 {current_json_file_path}")
            return
        except json.JSONDecodeError:
            print(f"错误：无法解析 JSON 文件 {current_json_file_path}")
            return

        image_pairs_with_bboxes = []
        for key, value in data.items():
            original_img_filename = value.get("img_path")
            edit_pairs = value.get("edit pair", [])

            if not original_img_filename:
                print(f"Warning: Skipping key {key} due to missing 'img_path'.")
                continue

            # 构建原始图像路径
            gt_img_path = os.path.join(ORIGINAL_IMAGE_PARENT_DIR, f"{key}-{original_img_filename}")

            # 构建编辑后图像路径
            # 假设编辑后的图像在以 key 命名的子目录下，文件名为 key_1.png
            edited_img_filename = f"{key}_1.png"
            gen_img_path = os.path.join(EDITED_IMAGE_BASE_DIR, json_subdir_name, "testResults_42", key, edited_img_filename)

            # 收集所有有效的边界框
            current_bboxes = []
            for edit in edit_pairs:
                bbox = edit.get("bbox")
                # 只添加非 "nothing" 且格式正确的边界框
                if isinstance(bbox, list) and len(bbox) == 4:
                    current_bboxes.append(bbox)

            # 只有当编辑后的图像存在时才添加到列表中
            # (你可能需要根据实际情况调整此检查)
            # if os.path.exists(gen_img_path): # 暂时注释掉，因为用户可能还没生成所有图像
            image_pairs_with_bboxes.append({
                'gen_img_path': gen_img_path,
                'gt_img_path': gt_img_path,
                'bboxes': current_bboxes
            })
            # else:
            #     print(f"Warning: Edited image not found at {gen_img_path}. Skipping pair for key {key}.")


        if not image_pairs_with_bboxes:
            print("错误：没有找到有效的图像对进行评估。请检查路径和JSON文件。")
            return

        # 选择设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 计算 L1 距离 (忽略bbox区域)
        l1_score_masked = eval_masked_distance(image_pairs_with_bboxes, metric='l1', device=device)
        print(f"{json_subdir_name} Masked L1 Distance (outside bboxes): {l1_score_masked}")

        # 计算 L2 距离 (忽略bbox区域)
        l2_score_masked = eval_masked_distance(image_pairs_with_bboxes, metric='l2', device=device)
        print(f"{json_subdir_name} Masked L2 Distance (outside bboxes): {l2_score_masked}")

if __name__ == "__main__":
    main()