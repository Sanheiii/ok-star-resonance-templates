import os
import json
import uuid
import shutil
from PIL import Image
import numpy as np


def process_coco_dataset(work_dir):
    json_file = 'coco_annotations.json'
    json_path = os.path.join(work_dir, json_file)

    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    annotations_by_img = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)

    padding = 2

    temp_dir_name = f"temp_{uuid.uuid4().hex}"
    temp_dir_path = os.path.join(work_dir, temp_dir_name)
    os.makedirs(temp_dir_path, exist_ok=True)
    print(f"创建临时处理目录: {temp_dir_name}")

    for index, img_info in enumerate(coco_data['images']):
        old_filename = img_info['file_name']
        old_filepath = os.path.join(work_dir, old_filename)

        name_part, ext = os.path.splitext(old_filename)
        target_ext = ext.lower()
        if target_ext in ['.jpg', '.jpeg']:
            target_ext = '.png'

        new_filename = f"{index}{target_ext}"
        new_filepath = os.path.join(temp_dir_path, new_filename)

        img_info['file_name'] = new_filename

        if not os.path.exists(old_filepath):
            print(f"跳过: {old_filename}")
            continue

        try:
            with Image.open(old_filepath) as orig:
                img = orig.convert("RGB")

            width, height = img.size
            img_data = np.array(img)

            white_ratio = np.mean(np.all(img_data == [255, 255, 255], axis=-1))

            if white_ratio <= 0.5:
                white_bg = Image.new("RGB", (width, height), "white")
                anns = annotations_by_img.get(img_info['id'], [])

                for ann in anns:
                    x, y, w, h = ann['bbox']

                    x_min = int(max(0, x - padding))
                    y_min = int(max(0, y - padding))
                    x_max = int(min(width, x + w + padding))
                    y_max = int(min(height, y + h + padding))

                    region = img.crop((x_min, y_min, x_max, y_max))
                    white_bg.paste(region, (x_min, y_min))
                img = white_bg

            img.save(new_filepath)

            if os.path.exists(old_filepath):
                os.remove(old_filepath)

            print(f"完成: {old_filename} -> {new_filename}")

        except Exception as e:
            print(f"错误 {old_filename}: {e}")

    print("开文件移出临时目录...")
    for temp_file in os.listdir(temp_dir_path):
        src_path = os.path.join(temp_dir_path, temp_file)
        dst_path = os.path.join(work_dir, temp_file)
        shutil.move(src_path, dst_path)
    
    # 删除空的临时目录
    os.rmdir(temp_dir_path)
    print(f"已清理临时目录: {temp_dir_name}")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print("全部完成")


if __name__ == "__main__":
    process_coco_dataset(r'./')