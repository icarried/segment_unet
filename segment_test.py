import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from torchvision import transforms
from torchmetrics import JaccardIndex, Dice, Accuracy, Specificity, Recall
from archs_unet import unet as UNet
from PIL import Image
import os
import torch.nn as nn
import yaml
import numpy as np
from tqdm import tqdm  # 引入tqdm

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

# Test function
def test(model, dataloader, accelerator, metrics):
    model.eval()
    test_loss = 0
    for metric in metrics.values():
        metric.reset()

    with torch.no_grad():
        batch_now = 0
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            inputs, targets = batch['image'].to(accelerator.device), batch['label'].to(accelerator.device)
            outputs = model(inputs)
            loss = nn.BCEWithLogitsLoss()(outputs, targets.float())
            test_loss += loss.item()

            if batch_now == 0:
                print(f"output mean: {outputs.mean()}, output max: {outputs.max()}, output min: {outputs.min()}")
                sigmoid_outputs = torch.sigmoid(outputs)
                print(f"sigmoid output mean: {sigmoid_outputs.mean()}, sigmoid output max: {sigmoid_outputs.max()}, sigmoid output min: {sigmoid_outputs.min()}")

            preds = torch.sigmoid(outputs) > 0.5
            targets_int = targets.int()  # 转换为整数张量
            metrics["iou"](preds, targets_int)
            metrics["dice"](preds, targets_int)
            metrics["acc"](preds, targets_int)
            metrics["se"](preds, targets_int)
            metrics["sp"](preds, targets_int)

            if batch_now == 0:
                # 保存图像到指定文件夹
                images_cell = inputs[:4].detach().cpu().numpy()
                images_mask = targets[:4].detach().cpu().numpy()
                images_pred = preds[:4].detach().cpu().numpy()
                for i in range(4):
                    if images_cell[i].shape[0] == 1:
                        img_cell = Image.fromarray((images_cell[i].squeeze(0) * 255).astype(np.uint8), mode='L')
                    else:
                        img_cell = Image.fromarray((images_cell[i].transpose(1, 2, 0) * 255).astype(np.uint8))

                    img_mask = Image.fromarray((images_mask[i].squeeze(0) * 255).astype(np.uint8), mode='L')
                    img_pred = Image.fromarray((images_pred[i].squeeze(0) * 255).astype(np.uint8), mode='L')

                    # images_cell和images_mask通道堆叠形成4通道图像
                    if images_cell[i].shape[0] == 1:
                        images_cell_3c = np.concatenate((images_cell[i], images_cell[i], images_cell[i]), axis=0)
                    else:
                        images_cell_3c = images_cell[i]
                    
                    img_mask_concat = np.concatenate((images_cell_3c, np.repeat(images_mask[i], 3, axis=0)), axis=0)
                    img_pred_concat = np.concatenate((images_cell_3c, np.repeat(images_pred[i], 3, axis=0)), axis=0)

                    reversed_mask = np.where(images_mask[i] > 0.5, 0, 1)
                    img_mask_reverse = np.concatenate((images_cell_3c, np.repeat(reversed_mask, 3, axis=0)), axis=0)
                    reversed_pred = np.where(images_pred[i] > 0.5, 0, 1)
                    img_pred_reverse = np.concatenate((images_cell_3c, np.repeat(reversed_pred, 3, axis=0)), axis=0)

                    img_mask_concat = Image.fromarray((img_mask_concat.transpose(1, 2, 0) * 255).astype(np.uint8), mode='RGBA')
                    img_pred_concat = Image.fromarray((img_pred_concat.transpose(1, 2, 0) * 255).astype(np.uint8), mode='RGBA')
                    img_mask_reverse = Image.fromarray((img_mask_reverse.transpose(1, 2, 0) * 255).astype(np.uint8), mode='RGBA')
                    img_pred_reverse = Image.fromarray((img_pred_reverse.transpose(1, 2, 0) * 255).astype(np.uint8), mode='RGBA')

                    img_cell.save(f"./output_img/cell_{i}.png")
                    img_mask.save(f"./output_img/mask_{i}.png")
                    img_pred.save(f"./output_img/pred_{i}.png")
                    img_mask_concat.save(f"./output_img/mask_concat_{i}.png")
                    img_pred_concat.save(f"./output_img/pred_concat_{i}.png")
                    img_mask_reverse.save(f"./output_img/mask_reverse_{i}.png")
                    img_pred_reverse.save(f"./output_img/pred_reverse_{i}.png")


            batch_now += 1

    accelerator.log({
        "test_loss": test_loss / len(dataloader),
        "test_iou": metrics["iou"].compute().item(),
        "test_dice": metrics["dice"].compute().item(),
        "test_acc": metrics["acc"].compute().item(),
        "test_se": metrics["se"].compute().item(),
        "test_sp": metrics["sp"].compute().item(),
    }
    )

    print(f"Test Loss: {test_loss / len(dataloader)}")
    print(f"IoU: {metrics['iou'].compute().item()}")
    print(f"Dice: {metrics['dice'].compute().item()}")
    print(f"Accuracy: {metrics['acc'].compute().item()}")
    print(f"Sensitivity: {metrics['se'].compute().item()}")
    print(f"Specificity: {metrics['sp'].compute().item()}")
    print("Testing complete")

# Main function
def main(args):
    # accelerator = Accelerator(log_with="wandb")  # 使用WandB日志记录器
    accelerator = Accelerator() # 测试时不使用WandB日志记录器
    # 初始化wandb追踪器
    accelerator.init_trackers(project_name=args.project, config=dict(vars(args)))

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
    test_dataset = CustomHFDataset(
        img_dir=os.path.join(args.dataset, 'img/test'),
        mask_dir=os.path.join(args.dataset, 'mask/test'),
        transform=transform,
        mask_transform=mask_transform,
        color_mode=color_mode
    )

    # Create dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    in_channels = 3 if args.color_mode == "rgb" else 1  # 根据color_mode设置输入通道数
    model = UNet(in_channels=in_channels, out_channels=1)
    model.load_state_dict(torch.load(args.model_path, map_location=accelerator.device))

    # Prepare model and dataloader with accelerator
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    # Initialize metrics
    metrics = {
        "iou": JaccardIndex(task="binary", num_classes=2).to(accelerator.device),
        "dice": Dice(num_classes=2).to(accelerator.device),
        "acc": Accuracy(task="binary").to(accelerator.device),
        "se": Recall(task="binary").to(accelerator.device),  # 使用 Recall 替换 Sensitivity
        "sp": Specificity(task="binary").to(accelerator.device)
    }

    print(f"Testing {args.project}")
    print(f"Using {args.color_mode} images")
    print(f"Using {len(test_dataset)} test samples")

    # Test the model
    test(model, test_dataloader, accelerator, metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a UNet model with WandB and Accelerate")
    parser.add_argument('--config', type=str, default="./seg_test_config.yaml", help='config file path, replace the default config, 用于指定配置文件路径')
    parser.add_argument("--project", type=str, default="seg_test", help="WandB project name")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved model")
    parser.add_argument("--color_mode", type=str, default="rgb", choices=["rgb", "grayscale"], help="Color mode of the images")

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # 用config字典中的参数替换args
            for key, value in config.items():
                setattr(args, key, value)
    main(args)
