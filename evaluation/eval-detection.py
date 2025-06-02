from enum import unique
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import re
import json
import os
from matplotlib.patches import Rectangle

def extract_objects_and_mask(image_path, text_list, processor, model, device, threshold=0.3):
    """
    从文本列表中提取唯一的目标对象，获取它们的边界框，然后为每个文本创建掩码图像
    
    参数:
        image_path: 图像路径
        text_list: 文本列表，每个元素描述一个目标
        processor: DINO处理器
        model: DINO模型
        device: 设备（CPU或GPU）
        threshold: 检测阈值
        
    返回:
        masked_images: 掩码图像列表，与text_list长度相同
        object_to_box: 对象到边界框的映射
    """
    # 加载图像
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # 提取唯一的目标对象
    '''将text_list变为index， unique_objects存储index为key,object为value'''
    unique_objects, indexs, index_lists = {}, 0, []
    for text in text_list:
        if text not in unique_objects.values():
            unique_objects[indexs] = text
            index_lists.append(indexs)
            indexs = indexs + 1 
        else:  # 如果text已经在unique-objects里，直接读取index
            for k, v in unique_objects.items():
                if text == v:
                    index_lists.append(k)
    
    # 构建用于DINO的文本提示
    text_prompt = ". ".join([f"{obj}" for obj in unique_objects.values()]) + "." # 舍弃了重复的obj
    
    # 使用DINO进行目标检测
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 处理检测结果
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold
    )[0]
    
    # 创建对象到边界框的映射
    object_to_box = {}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # 获取标签文本
        if "text_labels" in results:
            label_text = results["text_labels"][results["labels"].index(label) if isinstance(results["labels"], list) else results["labels"].tolist().index(label)]
        else:
            label_id = label.item() if torch.is_tensor(label) else label
            label_text = model.config.id2label.get(label_id, "unknown")
        
        '''提取object对应的bbox'''
        if label_text in unique_objects.values() and label_text not in object_to_box:
            object_to_box[label_text] = box.tolist() if torch.is_tensor(box) else box

    print(object_to_box)
    # 为每个文本创建掩码图像
    masked_images = []
    for ind in index_lists:
        # 创建一个全黑掩码
        mask = np.zeros_like(image_array)
        
        # 查找文本中的对象并应用对应的边界框
        text_objects = [unique_objects[ind]]
        for obj in text_objects:
            if obj in object_to_box:
                box = object_to_box[obj]
                # 在边界框区域内设置掩码为原始图像
                x1, y1, x2, y2 = map(int, box)
                mask[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
        
        # 将掩码转换为PIL图像并添加到列表
        masked_image = Image.fromarray(mask)
        masked_images.append(masked_image)
    
    return masked_images, object_to_box

def save_annotated_image(image_path, object_to_box, ori_objs, output_path):
    """
    保存标注了ori obj的图片
    
    参数:
        image_path: 图像路径
        object_to_box: 对象到边界框的映射
        ori_objs: 原始对象列表
        output_path: 输出路径
    """
    # 加载图像
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # 创建图形
    plt.figure(figsize=(10, 10))
    plt.imshow(image_array)
    
    # 在图像上绘制边界框
    for obj, box in object_to_box.items():
        if obj in ori_objs:
            x1, y1, x2, y2 = map(int, box)
            rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            plt.text(x1, y1-10, obj, color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # 设置标题
    plt.title(", ".join(ori_objs), fontsize=14)
    plt.axis('off')
    
    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def process_json_file_with_stats(json_path, image_dir, output_dir, processor, model, device, threshold=0.3):
    """
    处理JSON文件，为每个图片中的ori obj检测边界框，并保存结果，同时统计检测情况
    
    参数:
        json_path: JSON文件路径
        image_dir: 图像目录
        output_dir: 输出目录
        processor: DINO处理器
        model: DINO模型
        device: 设备（CPU或GPU）
        threshold: 检测阈值
        
    返回:
        images_count: 处理的图像总数
        fully_detected_count: 所有目标对象都被检测到的图像数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images_count = 0
    fully_detected_count = 0
    
    # 处理每张图片
    for key, image_info in data.items():
        image_id = image_info["img_path"]
        image_path = os.path.join(image_dir, f"{key}-{image_id}")
        
        if not os.path.exists(image_path):
            print(f"图片 {image_path} 不存在，跳过")
            continue
        
        images_count += 1
        
        # 收集所有非"nothing"的ori obj
        ori_objs = []
        for edit_pair in image_info["edit pair"]:
            if edit_pair["ori obj"] != "nothing":
                ori_objs.append(edit_pair["ori obj"])
        
        if not ori_objs:
            print(f"图片 {image_id} 没有有效的ori obj，跳过")
            continue
        
        # 获取唯一的ori obj列表
        unique_ori_objs = list(set(ori_objs))
        
        # 检测边界框
        _, object_to_box = extract_objects_and_mask(image_path, ori_objs, processor, model, device, threshold)
        
        # 检查是否所有唯一的ori obj都被检测到
        all_detected = True
        for obj in unique_ori_objs:
            if obj not in object_to_box:
                all_detected = False
                break
        
        if all_detected:
            fully_detected_count += 1
        
        # 更新JSON数据, 有识别出来的bbox就更新.
        for edit_pair in image_info["edit pair"]:
            if edit_pair["ori obj"] != "nothing":
                if edit_pair["ori obj"] in object_to_box:
                    edit_pair["bbox"] = object_to_box[edit_pair["ori obj"]]
                else:
                    edit_pair["bbox"] = "not detected"
            else:
                edit_pair["bbox"] = "nothing"
        
        # 保存标注了ori obj的图片
        save_annotated_image(image_path, object_to_box, ori_objs, os.path.join(output_dir, f"{image_id[:-4]}_annotated.jpg"))
    
    # 保存更新后的JSON文件
    output_json_path = os.path.join(output_dir, "updated_" + os.path.basename(json_path))  # 更新后的json文件，和带bbox标注的image存放到一起.
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"处理完成，更新后的JSON文件已保存到 {output_json_path}")
    print(f"图像总数: {images_count}, 所有目标都被检测到的图像数: {fully_detected_count}")
    
    return images_count, fully_detected_count


if __name__ == "__main__":

    # 处理editing-json-file下的所有子文件夹
    json_base_dir = "./data/instructions"  # json文件夹
    image_dir = "./data/more-object-no-multi3"  # 原始图像所在路径
    output_base_dir = "./evaluation/L1L2/output"  # 输出的bbox图像所在路径
    
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载Grounding DINO模型
    processor = AutoProcessor.from_pretrained("./dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("./dino-base").to(device)
    
    # 确保输出基础目录存在
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(json_base_dir) if os.path.isdir(os.path.join(json_base_dir, d))]
    
    total_images = 0
    total_fully_detected_images = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(json_base_dir, subdir)
        output_subdir = os.path.join(output_base_dir, subdir)
        
        # 确保输出子目录存在
        os.makedirs(output_subdir, exist_ok=True)
        
        # 获取子目录中的所有JSON文件
        json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
        
        for json_file in json_files:
            json_file_path = os.path.join(subdir_path, json_file)
            
            print(f"处理文件: {json_file_path}")
            
            # 处理JSON文件并统计检测情况
            images_count, fully_detected_count = process_json_file_with_stats(
                json_file_path, 
                image_dir, 
                output_subdir, 
                processor, 
                model, 
                device
            )
            
            total_images += images_count
            total_fully_detected_images += fully_detected_count
    
    print(f"\n统计结果:")
    print(f"总图像数: {total_images}")
    print(f"所有目标对象都被检测到的图像数: {total_fully_detected_images}")
    print(f"检测成功率: {(total_fully_detected_images / total_images * 100) if total_images > 0 else 0:.2f}%")