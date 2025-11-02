from pathlib import Path
import PIL
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Tuple

from detector.engine import train_one_epoch, evaluate
from detector.datasets.literature_dataset import InsectDetectionDataset
from detector.visualize import draw_gt_and_preds, GT, Pred


def convert_target_to_gts(target: dict) -> list:
    """Convert target dictionary to list of GT objects."""
    gts = []
    boxes = target["boxes"].cpu() if target["boxes"].is_cuda else target["boxes"]
    labels = target["labels"].cpu() if target["labels"].is_cuda else target["labels"]

    for box, label in zip(boxes, labels):
        # Boxes are already in [x1, y1, x2, y2] format
        x1, y1, x2, y2 = box.tolist()
        gts.append(GT(bbox=(x1, y1, x2, y2), label=str(label.item())))

    return gts


def convert_output_to_preds(output: dict, score_thresh: float = 0.5) -> list:
    """Convert model output dictionary to list of Pred objects."""
    preds = []
    boxes = output["boxes"].cpu() if output["boxes"].is_cuda else output["boxes"]
    scores = output["scores"].cpu() if output["scores"].is_cuda else output["scores"]
    labels = output["labels"].cpu() if output["labels"].is_cuda else output["labels"]

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        # Output is already in [x1, y1, x2, y2] format
        x1, y1, x2, y2 = box.tolist()
        preds.append(
            Pred(bbox=(x1, y1, x2, y2), score=score.item(), label=str(label.item()))
        )

    return preds


def create_fasterrcnn_model(num_classes, device):
    """
    Create a Faster R-CNN model with a ResNet-50 backbone.
    Ensure num_classes includes background class.
    Args:
        num_classes: Number of output classes (including background)
        device: Device to load the model onto ('cpu' or 'cuda')
    Returns:
        model: Configured Faster R-CNN model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    return model


def visualize_samples(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_folder: Path,
    epoch: int,
    split_name: str,
    num_samples: int = 4,
    score_thresh: float = 0.3,
):
    """Visualize predictions on fixed samples and save to disk."""
    model.eval()
    output_folder.mkdir(exist_ok=True, parents=True)

    # Get or create fixed sample indices for this split
    cache_attr = f"fixed_sample_indices_{split_name}"
    if not hasattr(visualize_samples, cache_attr):
        indices = []
        for i in range(len(dataloader.dataset)):
            _, target = dataloader.dataset[i]
            if target["boxes"].shape[0] > 0:
                indices.append(i)
                if len(indices) >= num_samples:
                    break
        setattr(visualize_samples, cache_attr, indices)

    fixed_indices = getattr(visualize_samples, cache_attr)

    with torch.no_grad():
        for idx, sample_i in enumerate(fixed_indices):
            image, target = dataloader.dataset[sample_i]
   
            actual_idx = dataloader.dataset.indices[idx]
            real_image = dataloader.dataset.image_files[actual_idx]
   
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)

            # vis_image = torch.clamp(image_tensor[0].cpu(), 0, 1)
            vis_image = Image.open(real_image).convert("RGB")

            gts = convert_target_to_gts(target)
            preds = convert_output_to_preds(output[0], score_thresh=score_thresh)
            draw_gt_and_preds(
                vis_image,
                gts,
                preds,
                out_path=output_folder / f"epoch_{epoch:03d}_sample_{idx}.png",
            )


def collate_fn(batch):
    """Custom collate function for variable-sized annotations."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def main_literature():
    seed_all(42)

    # Transforms - only color transforms, geometric transforms handled by ResizeWithBoxes in dataset
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            torchvision.transforms.ToTensor(),
        ]
    )
    val_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    train_dataset = InsectDetectionDataset(
        split="train",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
        transform=train_transform,
        max_size=IMG_SIZE,
    )
    val_dataset = InsectDetectionDataset(
        split="val",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
        transform=val_transform,
        max_size=IMG_SIZE,
    )
    test_dataset = InsectDetectionDataset(
        split="test",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
        transform=test_transform,
        max_size=IMG_SIZE,
    )

    print(
        f"Dataset splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = create_fasterrcnn_model(NUM_CLASSES, DEVICE)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Device: {DEVICE}")
    print(f"Learning rate: {LR}")
    print(f"Batch size: 8\n")

    best_val_map = 0.0

    for epoch in range(EPOCHS):
        train_one_epoch(
            model, optimizer, train_loader, DEVICE, epoch, print_freq=10
        )
        evaluator = evaluate(model, val_loader, device=DEVICE)
        
        # get validation mAP
        val_map = evaluator.coco_eval["bbox"].stats[0]
        print(f"Validation mAP at epoch {epoch}: {val_map:.4f}")

        if val_map > best_val_map:
            best_val_map = val_map
            # Save best model
            torch.save(
                model.state_dict(), RESULTS_FOLDER / "best_fasterrcnn_model.pth"
            )
            print(f"New best model saved with mAP: {best_val_map:.4f}")

        # Visualize validation samples
        visualize_samples(
            model,
            val_loader,
            DEVICE,
            RESULTS_FOLDER / "val_samples",
            epoch,
            "val",
            num_samples=4,
            score_thresh=0.3,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Faster R-CNN Training for Insect Detection")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Maximum image size")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use for training (e.g., 'cpu' or 'cuda:0')")
    args = parser.parse_args()

    EPOCHS = args.epochs
    LR = args.lr
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 3  # 2 insect classes + background
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.img_size
    EXP_NAME = f"fasterrcnn_literature_img={IMG_SIZE}"

    RESULTS_FOLDER = Path("results") / EXP_NAME
    RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)

    main_literature()
