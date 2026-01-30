import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm.autonotebook import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from refactor.segmentation.datasets import Coralscapes
import argparse
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

def get_args():
    parser = argparse.ArgumentParser(description = "Train Deeplabv3")
    parser.add_argument("-n", "--num_epochs", type = int, default = 100)
    parser.add_argument("-d","--coral_dataset_root", type = str, default = "D:\\AI_ENGINEER\\Hust\\GR2_UOD\\dataset\\segmentation\\coral_jpg")
    parser.add_argument("-b", "--batch_size", type = int, default = 2)
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="1e-3 for SGD, 1e-4 for Adam")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for Adam")
    parser.add_argument("--checkpoint_folder", "-c", type = str, default = "D:\\AI_ENGINEER\\Hust\\GR2_UOD\\train\\result\\segmentation")
    parser.add_argument("-r", "--resume", type=bool, default=False) # Train tiếp từ last checkpoint hay là lại từ đầu
    args = parser.parse_args()
    return args

def get_transforms(stage='stage1'):
    """
    stage1 (epoch 1-10): Light Augmentation
    stage2 (epoch 10-30): Add geometric transforms + RandomCrop
    stage3 (epoch 30+): Full augmentation
    """
    if stage == 'stage1':
        # Epoch 1-10: Chỉ flip cơ bản
        train_transform = A.Compose([
            A.Resize(513, 513),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    elif stage == 'stage2':
        # Epoch 10-30: Thêm RandomCrop + rotation nhẹ
        train_transform = A.Compose([
            A.OneOf([
                A.RandomCrop(513, 513),
                A.Resize(513, 513)
            ], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Affine(
                translate_percent=0.03,
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                interpolation=1,
                mask_interpolation=0,
                border_mode=0,
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    else:  # stage3
        # Epoch 30+: Full augmentation
        train_transform = A.Compose([
            A.OneOf([
                A.RandomCrop(513, 513),
                A.CenterCrop(513, 513)
            ], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                interpolation=1,
                mask_interpolation=0,
                border_mode=0,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    val_transform = A.Compose([
        A.Resize(513, 513),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

    return train_transform, val_transform

def train(class_weights):

    #Get CLI arguments
    params = get_args()
    num_epochs = params.num_epochs
    batch_size = params.batch_size
    # momentum = params.momentum
    lr = params.lr
    weight_decay = params.weight_decay
    resume = params.resume
    coralscapes_root_dir = params.coral_dataset_root
    ckpt_folder = params.checkpoint_folder
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)

    #Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, val_transform = get_transforms(stage='stage1')

    train_dataset = Coralscapes(root=coralscapes_root_dir, split='train', refactor_flag="new_", transform = train_transform, transform_target = True, img_ext = ".jpg" )
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
        drop_last = True
    )

    val_dataset = Coralscapes(root=coralscapes_root_dir, split='val', refactor_flag="new_", transform = val_transform, transform_target = False, img_ext = ".jpg")
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        num_workers = 0,
        drop_last = False
    )

    # Load pre_trained model Deeplabv3
    model = deeplabv3_resnet50(weights="DEFAULT")
    num_classes = len(train_dataset.classes)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
    model = model.to(device)
    # optimizer = torch.optim.SGD(params = model.parameters() ,lr = lr, momentum = momentum)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr = lr,
        weight_decay = weight_decay,
        betas=(0.9, 0.999),  # Default Adam betas
        eps=1e-8
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        threshold=0.005,
        threshold_mode='abs',
        min_lr=1e-6,
    )

    criterion = CrossEntropyLoss(
        weight = class_weights.to(device),
    )

    validation_metric = MulticlassJaccardIndex(num_classes=len(train_dataset.classes)).to(device)

    validation_metric_per_class = MulticlassJaccardIndex(num_classes=len(train_dataset.classes), average=None).to(device)
    # resume or not with latest epoch
    if resume:
        checkpoint_path = os.path.join(ckpt_folder, "adam_3_stage_model_last.pt")  # Train tiếp -> last, inference -> best
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_mIOU = checkpoint["mIOU"]
        current_lr = optimizer.param_groups[0]['lr']
        print("Now the current best mIOU is : {}".format(best_mIOU))
        print(f"Current LR from optimizer: {current_lr:.6f}")
        print(f"Scheduler best mIOU: {scheduler.best:.4f}")
        print(f"Scheduler num_bad_epochs: {scheduler.num_bad_epochs}")
    else:
        start_epoch = 0
        best_mIOU = -1


    for epoch in range(start_epoch,num_epochs):

        current_stage = None
        if epoch == 10:
            current_stage = 'stage2'
            print("\n" + "=" * 60)
            print("STAGE 2: Adding RandomCrop + Geometric transforms (epoch 11-30)")
            print("=" * 60 + "\n")
        elif epoch == 30:
            current_stage = 'stage3'
            print("\n" + "=" * 60)
            print("STAGE 3: Full augmentation with color transforms (epoch 30+)")
            print("=" * 60 + "\n")

        if current_stage:
            train_transform, _ = get_transforms(stage=current_stage)
            train_dataset.transform = train_transform
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )

        # Train
        model.train()
        progress_bar = tqdm(train_dataloader, colour = "yellow")
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            result = model(images)
            output = result["out"]
            aux_loss = criterion(result["aux"],targets)
            loss = criterion(output,targets) + aux_loss * 0.4
            progress_bar.set_description("Train: Epoch {}/{}. Loss {:0.4f}".format(epoch+1,num_epochs, loss.item()))
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        validation_metric.reset()
        validation_metric_per_class.reset()
        progress_bar = tqdm(val_dataloader, colour="cyan")
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad() :
                output = model(images)["out"]
                loss = criterion(output,targets)

                # IOU metric
                preds = output.argmax(dim=1)
                validation_metric.update(preds,targets)
                validation_metric_per_class.update(preds, targets)
                progress_bar.set_description("Val: Epoch {}/{}. Loss {:0.4f}".format(epoch+1,num_epochs, loss.item()))

        val_iou = validation_metric.compute()
        val_iou_per_class = validation_metric_per_class.compute()
        print(f"\nEpoch {epoch+1}/{num_epochs} mIOU : {val_iou}")
        print("IoU per class:")
        class_names = train_dataset.classes
        for i, (class_name, iou) in enumerate(zip(class_names, val_iou_per_class)):
            print(f"  Class {i} ({class_name}): {iou:.4f}")

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_iou) # key
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        else:
            print(f"Current LR: {new_lr:.6f}")

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),  # Lưu optimizer state (Adam) để resume training
            "scheduler": scheduler.state_dict(),
            "mIOU": val_iou,
        }
        torch.save(checkpoint, os.path.join(ckpt_folder, "adam_3_stage_model_last.pt"))

        if val_iou > best_mIOU:
            best_mIOU = val_iou
            torch.save(checkpoint, os.path.join(ckpt_folder, "adam_3_stage_model_best.pt"))


if __name__ == "__main__" :
    coralscapes_root_dir = "D:\\AI_ENGINEER\\Hust\\GR2_UOD\\dataset\\segmentation\\coral_jpg"
    train_dataset = Coralscapes(root=coralscapes_root_dir, split='train', img_ext=".jpg", refactor_flag="new_")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = compute_class_weights(train_dataset,6)
    ce_weights = torch.clamp(weights, max=5.0).to(device)
    train(ce_weights)









