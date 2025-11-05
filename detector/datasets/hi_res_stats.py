from pathlib import Path
import glob
import random
import json
from PIL import Image, ImageDraw


def load_coco_data(annotations_path):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    images_dict = {img['file_name']: img for img in coco_data['images']}
    
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    return coco_data, images_dict, annotations_by_image


def generate_samples(dataset_path, annotations_path, output_dir, num_samples=4, seed=42):
    coco_data, images_dict, annotations_by_image = load_coco_data(annotations_path)
    
    image_files = glob.glob(f"{dataset_path}/**/*.jpg", recursive=True)
    
    random.seed(seed)
    random.shuffle(image_files)
    
    selected_images = []
    for img_path in image_files:
        img_filename = Path(img_path).name
        
        if img_filename in images_dict:
            img_info = images_dict[img_filename]
            img_id = img_info['id']
            
            if img_id in annotations_by_image and len(annotations_by_image[img_id]) > 0:
                selected_images.append(img_path)
                if len(selected_images) == num_samples:
                    break
    
    print(f"\n=== Generating Samples ===")
    print(f"Found {len(selected_images)} images with bounding boxes")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, img_path in enumerate(selected_images):
        img = Image.open(img_path)
        img_filename = Path(img_path).name
        img_info = images_dict[img_filename]
        img_id = img_info['id']
        
        border = 10
        bordered_img = Image.new('RGB', (img.width + 2*border, img.height + 2*border), color='white')
        bordered_img.paste(img, (border, border))
        
        draw = ImageDraw.Draw(bordered_img)
        
        num_bboxes = len(annotations_by_image[img_id])
        print(f"  Sample {idx+1}: {img_filename} - {num_bboxes} bbox(es)")
        
        for ann in annotations_by_image[img_id]:
            x, y, w, h = ann['bbox']
            x1, y1 = x + border, y + border
            x2, y2 = x1 + w, y1 + h
            draw.rectangle([x1, y1, x2, y2], outline='red', width=10)
        
        dest_path = output_path / f"sample_{idx+1}.jpg"
        bordered_img.save(dest_path)
    
    print(f"Samples saved to {output_path}")


def calculate_statistics(dataset_path, annotations_path, output_dir):
    coco_data, images_dict, annotations_by_image = load_coco_data(annotations_path)
    
    image_files = glob.glob(f"{dataset_path}/**/*.jpg", recursive=True)
    
    total_images = len(image_files)
    images_with_bboxes = 0
    images_without_bboxes = 0
    total_bboxes = 0
    bboxes_per_image = []
    
    for img_path in image_files:
        img_filename = Path(img_path).name
        
        if img_filename in images_dict:
            img_info = images_dict[img_filename]
            img_id = img_info['id']
            
            if img_id in annotations_by_image:
                num_bboxes = len(annotations_by_image[img_id])
                if num_bboxes > 0:
                    images_with_bboxes += 1
                    total_bboxes += num_bboxes
                    bboxes_per_image.append(num_bboxes)
                else:
                    images_without_bboxes += 1
            else:
                images_without_bboxes += 1
    
    avg_bboxes = total_bboxes / images_with_bboxes if images_with_bboxes > 0 else 0
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total images: {total_images}")
    print(f"\nAnnotated images (with bboxes):")
    print(f"  Count: {images_with_bboxes}")
    print(f"  Percentage: {images_with_bboxes/total_images*100:.2f}%")
    print(f"\nBackground images (no bboxes):")
    print(f"  Count: {images_without_bboxes}")
    print(f"  Percentage: {images_without_bboxes/total_images*100:.2f}%")
    print(f"\nBounding boxes:")
    print(f"  Total: {total_bboxes}")
    print(f"  Average per annotated image: {avg_bboxes:.2f}")
    if bboxes_per_image:
        print(f"  Min per image: {min(bboxes_per_image)}")
        print(f"  Max per image: {max(bboxes_per_image)}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats_file = output_path / "dataset_statistics.txt"
    with open(stats_file, 'w') as f:
        f.write("=== Dataset Statistics ===\n\n")
        f.write(f"Total images: {total_images}\n\n")
        f.write(f"Annotated images (with bboxes):\n")
        f.write(f"  Count: {images_with_bboxes}\n")
        f.write(f"  Percentage: {images_with_bboxes/total_images*100:.2f}%\n\n")
        f.write(f"Background images (no bboxes):\n")
        f.write(f"  Count: {images_without_bboxes}\n")
        f.write(f"  Percentage: {images_without_bboxes/total_images*100:.2f}%\n\n")
        f.write(f"Bounding boxes:\n")
        f.write(f"  Total: {total_bboxes}\n")
        f.write(f"  Average per annotated image: {avg_bboxes:.2f}\n")
        if bboxes_per_image:
            f.write(f"  Min per image: {min(bboxes_per_image)}\n")
            f.write(f"  Max per image: {max(bboxes_per_image)}\n")
    
    print(f"\nStatistics saved to {stats_file}")


def main():
    dataset_path = "detector/data/hi_res/images/default"
    annotations_path = "detector/data/hi_res/annotations/instances_default.json"
    output_dir = "stats"
    
    generate_samples(dataset_path, annotations_path, output_dir, num_samples=4, seed=42)
    calculate_statistics(dataset_path, annotations_path, output_dir)


if __name__ == "__main__":
    main()