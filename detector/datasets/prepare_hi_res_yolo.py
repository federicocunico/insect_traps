import json
import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm


def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] normalized."""
    x, y, w, h = coco_bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return x_center, y_center, w_norm, h_norm


def convert_coco_to_yolo(data_root, output_root, train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    """Convert COCO format dataset to YOLO format with train/val/test splits."""
    
    data_root = Path(data_root)
    output_root = Path(output_root)
    
    random.seed(seed)
    
    annotations_file = data_root / 'annotations' / 'instances_default.json'
    images_dir = data_root / 'images' / 'default'
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_id_to_yolo = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
    
    print(f"Categories found: {categories}")
    print(f"Category ID mapping: {category_id_to_yolo}")
    
    images_dict = {img['id']: img for img in coco_data['images']}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    image_ids = list(images_dict.keys())
    random.shuffle(image_ids)
    
    n_train = int(len(image_ids) * train_split)
    n_val = int(len(image_ids) * val_split)
    
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Total images: {len(image_ids)}")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/len(image_ids)*100:.1f}%)")
    print(f"  Val: {len(val_ids)} ({len(val_ids)/len(image_ids)*100:.1f}%)")
    print(f"  Test: {len(test_ids)} ({len(test_ids)/len(image_ids)*100:.1f}%)")
    
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    for split_name, split_ids in splits.items():
        split_images_dir = output_root / 'images' / split_name
        split_labels_dir = output_root / 'labels' / split_name
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split_name} split...")
        for img_id in tqdm(split_ids):
            img_info = images_dict[img_id]
            img_filename = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']
            
            src_img_path = images_dir / img_filename
            dst_img_path = split_images_dir / img_filename
            
            if src_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
            else:
                print(f"Warning: Image {img_filename} not found")
                continue
            
            label_filename = Path(img_filename).stem + '.txt'
            label_path = split_labels_dir / label_filename
            
            if img_id in annotations_by_image:
                with open(label_path, 'w') as f:
                    for ann in annotations_by_image[img_id]:
                        category_id = ann['category_id']
                        yolo_class_id = category_id_to_yolo[category_id]
                        coco_bbox = ann['bbox']
                        
                        x_center, y_center, w_norm, h_norm = coco_to_yolo_bbox(
                            coco_bbox, img_width, img_height
                        )
                        
                        f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            else:
                with open(label_path, 'w') as f:
                    pass
    
    train_txt = output_root / 'train.txt'
    val_txt = output_root / 'val.txt'
    test_txt = output_root / 'test.txt'
    
    with open(train_txt, 'w') as f:
        for img_id in train_ids:
            img_filename = images_dict[img_id]['file_name']
            img_path = f"./images/train/{img_filename}"
            f.write(f"{img_path}\n")
    
    with open(val_txt, 'w') as f:
        for img_id in val_ids:
            img_filename = images_dict[img_id]['file_name']
            img_path = f"./images/val/{img_filename}"
            f.write(f"{img_path}\n")
    
    with open(test_txt, 'w') as f:
        for img_id in test_ids:
            img_filename = images_dict[img_id]['file_name']
            img_path = f"./images/test/{img_filename}"
            f.write(f"{img_path}\n")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_root}")
    print(f"  Train file: {train_txt}")
    print(f"  Val file: {val_txt}")
    print(f"  Test file: {test_txt}")
    
    return categories, category_id_to_yolo


if __name__ == "__main__":
    data_root = "data/hi_res"
    output_root = "data/hi_res"
    
    categories, category_mapping = convert_coco_to_yolo(
        data_root=data_root,
        output_root=output_root,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=42
    )
    
    print(f"\nClass names for YAML file:")
    for yolo_id in sorted(category_mapping.values()):
        coco_id = [k for k, v in category_mapping.items() if v == yolo_id][0]
        print(f"  {yolo_id}: {categories[coco_id]}")
