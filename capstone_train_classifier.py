import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
import sys
import ast

class BicycleStateDataset:
    """Dataset with expanded crop region (20% larger than bbox)"""
    def __init__(self, feather_file, img_dir="", transform=None, expand_factor=0.2):
        self.data = pd.read_feather(feather_file).dropna()
        self.data = self.data[self.data['new_label']!=2].reset_index(drop = True)
        self.img_dir = img_dir
        self.transform = transform
        self.expand_factor = expand_factor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["filename"]
        if self.img_dir:
            img_path = os.path.join(self.img_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # bbox_str = row["bbox_2d"]
        # if isinstance(bbox_str, str):
        #     bbox_list = ast.literal_eval(bbox_str)
        # else:
        #     bbox_list = bbox_str 
        # x1, y1, x2, y2 = map(float, bbox_list)
        bbox = row['bbox_2d']
        if isinstance(bbox, str):
            x1, y1, x2, y2 = ast.literal_eval(bbox)
        else:
            x1, y1, x2, y2 = bbox
        
        # Calculate bbox dimensions
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Expand by expand_factor (20% means 10% on each side)
        expansion_w = bbox_w * self.expand_factor / 2
        expansion_h = bbox_h * self.expand_factor / 2
        
        # Calculate expanded coordinates
        exp_x1 = x1 - expansion_w
        exp_y1 = y1 - expansion_h
        exp_x2 = x2 + expansion_w
        exp_y2 = y2 + expansion_h
        
        # Clamp to image boundaries
        exp_x1 = max(0, exp_x1)
        exp_y1 = max(0, exp_y1)
        exp_x2 = min(img_width, exp_x2)
        exp_y2 = min(img_height, exp_y2)
        
        # Crop the expanded region
        roi = image.crop((exp_x1, exp_y1, exp_x2, exp_y2))
        
        if self.transform:
            image = self.transform(image)
            roi = self.transform(roi)
        
        label = torch.tensor(int(row["new_label"]), dtype=torch.long)
        return image, roi, label


class ResizeWithPadding:
    """Resize image while preserving aspect ratio and padding to square"""
    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill
    
    def __call__(self, img):
        # Get original dimensions
        w, h = img.size
        
        # Calculate scaling factor to fit within target_size
        scale = self.target_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize maintaining aspect ratio
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Create new image with padding
        new_img = Image.new('RGB', (self.target_size, self.target_size), 
                           color=(self.fill, self.fill, self.fill))
        
        # Calculate padding to center the image
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        
        # Paste resized image onto padded canvas
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img

class TeeLogger:
    """Redirects stdout to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# class BicycleStateDataset(Dataset):
#     def __init__(self, feather_file, img_dir="", transform=None):
#         self.data = pd.read_feather(feather_file).dropna()
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         img_path = row["filename"]
#         if self.img_dir:
#             img_path = os.path.join(self.img_dir, img_path)
#         image = Image.open(img_path).convert("RGB")
#         bbox_str = row["bbox_2d"]
#         bbox_list = ast.literal_eval(bbox_str)
#         x1, y1, x2, y2 = map(float, bbox_list)
#         roi = image.crop((x1, y1, x2, y2))
#         if self.transform:
#             image = self.transform(image)
#             roi = self.transform(roi)
#         # label = torch.tensor(int(row["target"]), dtype=torch.long)
#         label = torch.tensor(int(row["new_label"]), dtype=torch.long)
#         return image, roi, label

class DualStreamBicycleClassifier(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, num_classes=2, 
                 fc_hidden_dims=[256], dropout=0.3):
        super().__init__()
        
        # Weight mapping
        weights_map = {
            "resnet18": ResNet18_Weights.DEFAULT if pretrained else None,
            "resnet34": ResNet34_Weights.DEFAULT if pretrained else None,
            "resnet50": ResNet50_Weights.DEFAULT if pretrained else None,
        }
        
        weights = weights_map.get(backbone_name)
        if backbone_name == "resnet18":
            backbone_scene = resnet18(weights=weights)
            backbone_roi = resnet18(weights=weights)
        elif backbone_name == "resnet34":
            backbone_scene = resnet34(weights=weights)
            backbone_roi = resnet34(weights=weights)
        elif backbone_name == "resnet50":
            backbone_scene = resnet50(weights=weights)
            backbone_roi = resnet50(weights=weights)
        
        # Remove classification heads
        self.scene_encoder = nn.Sequential(*list(backbone_scene.children())[:-1])
        self.roi_encoder = nn.Sequential(*list(backbone_roi.children())[:-1])
        
        # Get feature dimension
        feat_dim = backbone_scene.fc.in_features
        
        self.fc_scene = nn.Linear(feat_dim, fc_hidden_dims[0])
        self.fc_roi = nn.Linear(feat_dim, fc_hidden_dims[0])
        
        # Build classifier with variable depth
        classifier_layers = []
        input_dim = fc_hidden_dims[0] * 2  # Concatenated features
        
        for hidden_dim in fc_hidden_dims:
            classifier_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, full_img, roi):
        scene_feat = self.scene_encoder(full_img).flatten(1)
        roi_feat = self.roi_encoder(roi).flatten(1)
        scene_feat = F.relu(self.fc_scene(scene_feat))
        roi_feat = F.relu(self.fc_roi(roi_feat))
        fused = torch.cat([scene_feat, roi_feat], dim=1)
        logits = self.classifier(fused)
        return logits
    
    def freeze_backbone(self):
        """Freeze backbone encoders"""
        for param in self.scene_encoder.parameters():
            param.requires_grad = False
        for param in self.roi_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone encoders"""
        for param in self.scene_encoder.parameters():
            param.requires_grad = True
        for param in self.roi_encoder.parameters():
            param.requires_grad = True

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(config, train_loader, val_loader, device, run_dir):
    """Train a single model with given hyperparameters"""
    
    model = DualStreamBicycleClassifier(
        backbone_name=config['backbone'],
        pretrained=True,
        num_classes=2,
        fc_hidden_dims=config['fc_dims'],
        dropout=config['dropout']
    ).to(device)
    
    # Freeze backbone initially if specified
    if config.get('freeze_backbone_epochs', 0) > 0:
        model.freeze_backbone()
        print(f"Backbone frozen for first {config['freeze_backbone_epochs']} epochs")
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0)
    )
    
    # Add label smoothing
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get('label_smoothing', 0.0)
    )
    
    # Learning rate scheduler
    if config['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    elif config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                 factor=0.5, patience=5)
    else:
        scheduler = None
    
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    best_val_acc = 0
    history = []
    
    for epoch in range(config['epochs']):
        # Unfreeze backbone after specified epochs
        if config.get('freeze_backbone_epochs', 0) > 0 and epoch == config['freeze_backbone_epochs']:
            model.unfreeze_backbone()
            print(f"Backbone unfrozen at epoch {epoch+1}")
            # Optionally reduce learning rate when unfreezing
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", 
                         file=sys.stdout, ncols=100)
        
        for imgs, rois, labels in train_pbar:
            imgs, rois, labels = imgs.to(device), rois.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs, rois)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]  ", 
                       file=sys.stdout, ncols=100)
        
        with torch.no_grad():
            for imgs, rois, labels in val_pbar:
                imgs, rois, labels = imgs.to(device), rois.to(device), labels.to(device)
                logits = model(imgs, rois)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        # Update scheduler
        if scheduler is not None:
            if config['scheduler'] == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(run_dir, 'best_model.pth'))
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(run_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return best_val_acc, history

def main():
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"hyperparam_search_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "training.log")
    sys.stdout = TeeLogger(log_file)
    
    print(f"Starting hyperparameter search at {timestamp}")
    print(f"Logs saved to: {log_dir}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # UPDATED TRAINING TRANSFORM - AUGMENTATION FOR DRIVING DATA
    train_transform = transforms.Compose([
        ResizeWithPadding(target_size=224, fill=0),  # Preserve aspect ratio with black padding
        
        # Geometric transformations (position/orientation variations)
        transforms.RandomHorizontalFlip(p=0.5),  # Bikes can appear from either direction
        transforms.RandomRotation(20),  # Slightly more rotation for camera angles
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.15, 0.15),  # More translation - ROI position varies
            scale=(0.85, 1.15),  # Scale variation - distance from camera
            shear=5  # Small shear for perspective variations
        ),
        
        # Color/lighting augmentations (day/night, weather conditions)
        transforms.ColorJitter(
            brightness=0.4,  # Day to dusk to night transitions
            contrast=0.4,    # Overexposed to underexposed
            saturation=0.3,  # Weather/lighting conditions
            hue=0.1          # Color temperature variations
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))  # Motion blur, focus issues
        ], p=0.3),
        
        # Simulate low-light/night conditions
        transforms.RandomApply([
            transforms.Lambda(lambda img: transforms.functional.adjust_brightness(img, 0.5))
        ], p=0.2),
        
        # Random grayscale (night vision, poor lighting)
        transforms.RandomGrayscale(p=0.1),
        
        # Normalize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # VALIDATION TRANSFORM WITHOUT AUGMENTATION
    val_transform = transforms.Compose([
        ResizeWithPadding(target_size=224, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # train_dataset = BicycleStateDataset("cnn_classifier_train_data_restricted.feather", 
    #                                   "data/nuscenes/datasets/v1.0-trainval", 
    #                                   transform=train_transform)
    # val_dataset = BicycleStateDataset("cnn_classifier_val_data_restricted.feather", 
    #                                  "data/nuscenes/datasets/v1.0-trainval", 
    #                                  transform=val_transform)
                                     
    train_dataset = BicycleStateDataset("hand_labeled_train_data_5k.feather", 
                                       "data/nuscenes/datasets/v1.0-trainval", 
                                       transform=train_transform)
    val_dataset = BicycleStateDataset("hand_labeled_data.feather", 
                                     "data/nuscenes/datasets/v1.0-trainval", 
                                     transform=val_transform)
    
    # Hyperparameter configurations to try
    configs = [
        # Anti-overfitting configuration
        {
            'name': 'anti_overfit',
            'backbone': 'resnet18',
            'fc_dims': [128],           # Simpler architecture
            'dropout': 0.6,              # Higher dropout
            'lr': 5e-5,                  # Lower learning rate
            'scheduler': 'cosine',
            'batch_size': 32,            # Larger batch size
            'epochs': 50,
            'weight_decay': 1e-3,        # L2 regularization
            'label_smoothing': 0.1,      # Label smoothing
            'freeze_backbone_epochs': 5  # Freeze backbone for first 5 epochs
        },
        # Baseline - anti - overfit - hybrid
        {
            'name': 'anti_overfit',
            'backbone': 'resnet18',
            'fc_dims': [256,128],           # Simpler architecture
            'dropout': 0.5,              # Higher dropout
            'lr': 1e-5,                  # Lower learning rate
            'scheduler': 'step',
            'batch_size': 16,            # Larger batch size
            'epochs': 50,
            'weight_decay': 0,        # L2 regularization
            'label_smoothing': 0.1,      # Label smoothing
            'freeze_backbone_epochs': 6  # Freeze backbone for first 5 epochs
        },
        # Baseline
        {
            'name': 'baseline',
            'backbone': 'resnet18',
            'fc_dims': [256, 128],
            'dropout': 0.3,
            'lr': 1e-4,
            'scheduler': 'step',
            'batch_size': 16,
            'epochs': 50,
            'weight_decay': 0,
            'label_smoothing': 0.0,
            'freeze_backbone_epochs': 0
        },
        # Deeper FC layers
        {
            'name': 'deep_fc',
            'backbone': 'resnet18',
            'fc_dims': [512, 256, 128],
            'dropout': 0.4,
            'lr': 1e-4,
            'scheduler': 'step',
            'batch_size': 16,
            'epochs': 50,
            'weight_decay': 0,
            'label_smoothing': 0.0,
            'freeze_backbone_epochs': 0
        },
        # Larger backbone
        {
            'name': 'resnet34',
            'backbone': 'resnet34',
            'fc_dims': [256, 128],
            'dropout': 0.3,
            'lr': 1e-4,
            'scheduler': 'cosine',
            'batch_size': 16,
            'epochs': 50,
            'weight_decay': 0,
            'label_smoothing': 0.0,
            'freeze_backbone_epochs': 0
        },
        # Higher learning rate with plateau scheduler
        {
            'name': 'high_lr_plateau',
            'backbone': 'resnet18',
            'fc_dims': [256, 128],
            'dropout': 0.3,
            'lr': 5e-4,
            'scheduler': 'plateau',
            'batch_size': 16,
            'epochs': 50,
            'weight_decay': 0,
            'label_smoothing': 0.0,
            'freeze_backbone_epochs': 0
        },
        # Wider FC layers
        {
            'name': 'wide_fc',
            'backbone': 'resnet18',
            'fc_dims': [512, 256],
            'dropout': 0.5,
            'lr': 1e-4,
            'scheduler': 'cosine',
            'batch_size': 16,
            'epochs': 50,
            'weight_decay': 0,
            'label_smoothing': 0.0,
            'freeze_backbone_epochs': 0
        },
        # Lower learning rate
        {
            'name': 'low_lr',
            'backbone': 'resnet18',
            'fc_dims': [256, 128],
            'dropout': 0.3,
            'lr': 5e-5,
            'scheduler': 'step',
            'batch_size': 16,
            'epochs': 50,
            'weight_decay': 0,
            'label_smoothing': 0.0,
            'freeze_backbone_epochs': 0
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Running configuration {i+1}/{len(configs)}: {config['name']}")
        print(f"Config: {json.dumps(config, indent=2)}")
        print(f"{'='*80}\n")
        
        run_dir = os.path.join(log_dir, f"run_{i+1}_{config['name']}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                 shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                               shuffle=False, num_workers=4)
        
        # Train model
        try:
            best_acc, history = train_model(config, train_loader, val_loader, device, run_dir)
            
            # Save history
            with open(os.path.join(run_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
            
            results.append({
                'config': config,
                'best_val_acc': best_acc,
                'final_epoch': len(history)
            })
            
            print(f"\nCompleted {config['name']}: Best Val Acc = {best_acc:.4f}\n")
            
        except Exception as e:
            print(f"Error training {config['name']}: {str(e)}")
            continue
    
    # Save summary
    results_sorted = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
    
    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH RESULTS")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results_sorted):
        print(f"{i+1}. {result['config']['name']}: Val Acc = {result['best_val_acc']:.4f} "
              f"(Epochs: {result['final_epoch']})")
    
    with open(os.path.join(log_dir, 'summary.json'), 'w') as f:
        json.dump(results_sorted, f, indent=2)
    
    print(f"\nAll results saved to: {log_dir}")

if __name__ == "__main__":
    main()