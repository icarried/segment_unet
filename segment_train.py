import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from torchvision import transforms
from transformers import get_scheduler
import random
import numpy as np
import shutil
import os
import yaml
from torchmetrics import JaccardIndex, Dice, Accuracy, Specificity, Recall
from archs_unet import unet as UNet
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm  # 引入tqdm
import logging
import tempfile

logger = get_logger(__name__, log_level="INFO")

# Custom dataset class
# dataset/
# ├── img/
# │   ├── train/
# │   ├── validation/
# │   └── test/
# └── mask/
#     ├── train/
#     ├── validation/
#     └── test/
# enlarge_dataset/
# ├── img/
# │   └── train/
# └── mask/
#     └── train/
# Custom dataset class for loading images and masks from file structure
class CustomHFDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None, color_mode="RGB"):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.color_mode = color_mode
        self.img_files = sorted(os.listdir(self.img_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        mask_file = self.mask_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(img_path).convert(self.color_mode)
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return {'image': image, 'label': mask}

# Training function
def train(model, dataloader, optimizer, lr_scheduler, accelerator, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
        inputs, targets = batch['image'].to(accelerator.device), batch['label'].to(accelerator.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()

    # Log average loss for the epoch
    accelerator.log({"train_loss": total_loss / len(dataloader)}, step=epoch)

# Validation function
def validate(model, dataloader, accelerator, metrics, epoch):
    model.eval()
    val_loss = 0
    for metric in metrics.values():
        metric.reset()

    batch_count = 0  # 添加一个计数器, 用于上传首个batch中的图像和mask(循环里的batch为字典对象不能直接用)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch}"):
            inputs, targets = batch['image'].to(accelerator.device), batch['label'].to(accelerator.device)
            outputs = model(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
            val_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            targets_int = targets.int()  # 转换为整数张量
            metrics["iou"](preds, targets_int)
            metrics["dice"](preds, targets_int)
            metrics["acc"](preds, targets_int)
            metrics["se"](preds, targets_int)
            metrics["sp"](preds, targets_int)

            # 上传首个batch中前4个图像和mask
            if batch_count == 0:
                images_cell = inputs[:4].detach().cpu().numpy()
                images_mask = targets[:4].detach().cpu().numpy()

                # Ensure the temporary directory exists
                temp_dir = tempfile.gettempdir()
                wandb_media_dir = os.path.join(temp_dir, 'wandb-media')
                if not os.path.exists(wandb_media_dir):
                    os.makedirs(wandb_media_dir)

                accelerator.log({
                    "val_image": [wandb.Image(image) for image in images_cell],
                    "val_mask": [wandb.Image(image) for image in images_mask],
                },
                step=epoch)
            batch_count += 1

    accelerator.log({
        "val_loss": val_loss / len(dataloader),
        "val_iou": metrics["iou"].compute().item(),
        "val_dice": metrics["dice"].compute().item(),
        "val_acc": metrics["acc"].compute().item(),
        "val_se": metrics["se"].compute().item(),
        "val_sp": metrics["sp"].compute().item(),
    },
    step=epoch)

# Test function
def test(model, dataloader, accelerator, metrics):
    model.eval()
    test_loss = 0
    for metric in metrics.values():
        metric.reset()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            inputs, targets = batch['image'].to(accelerator.device), batch['label'].to(accelerator.device)
            outputs = model(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
            test_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5
            targets_int = targets.int()  # 转换为整数张量
            metrics["iou"](preds, targets_int)
            metrics["dice"](preds, targets_int)
            metrics["acc"](preds, targets_int)
            metrics["se"](preds, targets_int)
            metrics["sp"](preds, targets_int)

    accelerator.log({
        "test_loss": test_loss / len(dataloader),
        "test_iou": metrics["iou"].compute().item(),
        "test_dice": metrics["dice"].compute().item(),
        "test_acc": metrics["acc"].compute().item(),
        "test_se": metrics["se"].compute().item(),
        "test_sp": metrics["sp"].compute().item(),
    }
    )

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Main function
def main(args):
    set_seed(args.seed)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(log_with="wandb", project_config=accelerator_project_config)  # 使用WandB日志记录器
    # 初始化wandb追踪器
    accelerator.init_trackers(project_name=args.project, config=dict(vars(args)))

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # 设置color_mode
    color_mode = "RGB" if args.color_mode == "rgb" else "L"
    
    # Create custom datasets
    if args.enlarge_train_dataset is not None:
        if args.enlarge_and_org_choice == "org":
            train_dataset = CustomHFDataset(
                img_dir=os.path.join(args.dataset, 'img/train'),
                mask_dir=os.path.join(args.dataset, 'mask/train'),
                transform=transform,
                mask_transform=mask_transform,
                color_mode=color_mode
            )
        elif args.enlarge_and_org_choice == "enlarge":
            train_dataset = CustomHFDataset(
                img_dir=os.path.join(args.enlarge_train_dataset, 'img/train'),
                mask_dir=os.path.join(args.enlarge_train_dataset, 'mask/train'),
                transform=transform,
                mask_transform=mask_transform,
                color_mode=color_mode
            )
        elif args.enlarge_and_org_choice == "both":
            train_dataset = CustomHFDataset(
                img_dir=os.path.join(args.dataset, 'img/train'),
                mask_dir=os.path.join(args.dataset, 'mask/train'),
                transform=transform,
                mask_transform=mask_transform,
                color_mode=color_mode
            )
            enlarge_train_dataset = CustomHFDataset(
                img_dir=os.path.join(args.enlarge_train_dataset, 'img/train'),
                mask_dir=os.path.join(args.enlarge_train_dataset, 'mask/train'),
                transform=transform,
                mask_transform=mask_transform,
                color_mode=color_mode
            )
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, enlarge_train_dataset])
    else:
        train_dataset = CustomHFDataset(
            img_dir=os.path.join(args.dataset, 'img/train'),
            mask_dir=os.path.join(args.dataset, 'mask/train'),
            transform=transform,
            mask_transform=mask_transform,
            color_mode=color_mode
        )
    val_dataset = CustomHFDataset(
        img_dir=os.path.join(args.dataset, 'img/validation'),
        mask_dir=os.path.join(args.dataset, 'mask/validation'),
        transform=transform,
        mask_transform=mask_transform,
        color_mode=color_mode
    )
    test_dataset = CustomHFDataset(
        img_dir=os.path.join(args.dataset, 'img/test'),
        mask_dir=os.path.join(args.dataset, 'mask/test'),
        transform=transform,
        mask_transform=mask_transform,
        color_mode=color_mode
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model, optimizer, and scheduler
    in_channels = 3 if args.color_mode == "rgb" else 1  # 根据color_mode设置输入通道数
    model = UNet(in_channels=in_channels, out_channels=1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.num_epochs * len(train_dataloader)) # cosine

    # Prepare model and dataloaders with accelerator
    model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, test_dataloader, optimizer, lr_scheduler
    )

    # Initialize metrics
    metrics = {
        "iou": JaccardIndex(task="binary", num_classes=2).to(accelerator.device),
        "dice": Dice(num_classes=2).to(accelerator.device),
        "acc": Accuracy(task="binary").to(accelerator.device),
        "se": Recall(task="binary").to(accelerator.device),  # 使用 Recall 替换 Sensitivity
        "sp": Specificity(task="binary").to(accelerator.device)
    }

    logger.info(f"Training {args.project} for {args.num_epochs} epochs")
    logger.info(f"Using {args.color_mode} images")
    logger.info(f"Enlarging train dataset: {args.enlarge_and_org_choice}")
    logger.info(f"Using {len(train_dataset)} training samples")
    logger.info(f"Using {len(val_dataset)} validation samples")
    logger.info(f"Using {len(test_dataset)} test samples")

    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            first_epoch = int(path.split("-")[1])

    args.resume_from_checkpoint

    # Training loop
    progress_bar = tqdm(range(args.num_epochs), desc="Training", unit="epoch")
    progress_bar.update(first_epoch)
    for epoch in range(first_epoch, args.num_epochs):
        train(model, train_dataloader, optimizer, lr_scheduler, accelerator, epoch)
        validate(model, val_dataloader, accelerator, metrics, epoch)
        progress_bar.update(1)

        if accelerator.is_main_process:
            # 保存模型
            if (epoch + 1) % args.checkpoint_epochs == 0:
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch + 1}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Test the model
        test(model, test_dataloader, accelerator, metrics)

        # Save the model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        model_to_save = accelerator.unwrap_model(model)  # 获取原始模型
        model_save_path = os.path.join(args.save_dir, f'{args.project}_of_{args.enlarge_and_org_choice}_epoch{args.num_epochs}.pt')
        torch.save(model_to_save.state_dict(), model_save_path)
        logger.info(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a UNet model with WandB and Accelerate")
    parser.add_argument('--config', type=str, default="./seg_train_config.yaml", help='config file path, replace the default config, 用于指定配置文件路径')
    parser.add_argument("--project", type=str, default="nameless", help="WandB project name")
    parser.add_argument("--logging_dir", type=str, default="./log", help="Directory to log WandB logs")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save WandB logs")
    parser.add_argument("--dataset", type=str, required=None, help="Dataset directory path")
    parser.add_argument("--enlarge_train_dataset", type=str, default=None, help="使用的扩充数据集的路径")
    parser.add_argument("--enlarge_and_org_choice", type=str, default="org", choices=["org", "enlarge", "both"], help="使用原始数据集、扩充数据集还是两者都用")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--color_mode", type=str, default="rgb", choices=["rgb", "grayscale"], help="Color mode of the images")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training, latest checkpoint will be used if value is 'latest'")
    parser.add_argument("--checkpoint_epochs", type=int, default=5, help="Epoch to resume training from")
    parser.add_argument("--checkpoints_total_limit" , type=int, default=5, help="Total number of checkpoints to keep")

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # 用config字典中的参数替换args
            for key, value in config.items():
                setattr(args, key, value)

    if args.dataset is None:
        raise ValueError("Please provide a dataset directory path")
    
    main(args)
