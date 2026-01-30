import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm.autonotebook import tqdm
import os
from refactor.segmentation.datasets import Coralscapes
import albumentations as A
from albumentations.pytorch import ToTensorV2

def compute_class_weights(dataset, num_classes):
    class_values = [0, 10, 20, 30, 40, 50]
    # class_values = list(range(40))
    class_counts = torch.zeros(num_classes)

    print("Computing class weights...")
    for _, mask in tqdm(dataset, desc="Analyzing dataset"):
        for i, c in enumerate(class_values):
            class_counts[i] += (mask == c).sum().item()

    total = class_counts.sum()
    weights = total / (num_classes * class_counts)

    print("\nClass distribution:")
    for i, (val, count, weight) in enumerate(zip(class_values, class_counts, weights)):
        print(f"  Class {i} (value={val}): {count:.0f} pixels (weight: {weight:.4f})")

    return weights.float()

if __name__ == '__main__':
    coralscapes_root_dir = "D:\\AI_ENGINEER\\Hust\\GR2_UOD\\dataset\\segmentation\\coral_jpg"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = Coralscapes(root=coralscapes_root_dir, split='train', img_ext=".jpg", refactor_flag="new_")
    weights = compute_class_weights(train_dataset, 6)
    ce_weights = torch.clamp(weights, max=5.0).to(device)

    model = deeplabv3_resnet50(pretrained=False, aux_loss=True, num_classes=6)
    checkpoint_path = 'D:\\AI_ENGINEER\\Hust\\GR2_UOD\\train\\result\\segmentation\\adam_3_stage_model_best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    seg_model = model
    seg_model.load_state_dict(checkpoint["model"])
    seg_model.to(device)

    test_transform = A.Compose([
        A.Resize(513, 513),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    test_dataset = Coralscapes(root=coralscapes_root_dir, split='val', refactor_flag="new_", transform=test_transform,
                              transform_target=False, img_ext=".jpg")

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        num_workers=0,
        drop_last=False
    )
    criterion = CrossEntropyLoss(
        weight=ce_weights.to(device),
    )

    validation_metric = MulticlassJaccardIndex(num_classes=len(test_dataset.classes)).to(device)

    validation_metric_per_class = MulticlassJaccardIndex(num_classes=len(test_dataset.classes), average=None).to(
        device)
    seg_model.eval()

    validation_metric.reset()
    validation_metric_per_class.reset()
    progress_bar = tqdm(test_dataloader, colour="cyan")
    for images, targets in progress_bar:
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            output = model(images)["out"]
            loss = criterion(output, targets)

            # IOU metric
            preds = output.argmax(dim=1)
            validation_metric.update(preds, targets)
            validation_metric_per_class.update(preds, targets)
            progress_bar.set_description("Test: Loss {:0.4f}".format(loss.item()))

    test_iou = validation_metric.compute()
    val_iou_per_class = validation_metric_per_class.compute()
    print(f"\n mIOU : {test_iou}")
    print("IoU per class:")
    class_names = test_dataset.classes
    for i, (class_name, iou) in enumerate(zip(class_names, val_iou_per_class)):
        print(f"  Class {i} ({class_name}): {iou:.4f}")