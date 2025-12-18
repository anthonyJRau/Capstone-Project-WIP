import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import argparse
from tqdm import tqdm
import ast

class BicycleStateDataset(Dataset):
    def __init__(self, feather_file, img_dir="", transform=None):
        self.data = pd.read_feather(feather_file).dropna()
        self.data = self.data[self.data['new_label']!=2].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["filename"]
        if self.img_dir:
            img_path = os.path.join(self.img_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        bbox = row["bbox_2d"]
        # Only do literal_eval if bbox is a string
        if isinstance(bbox, str):
            bbox_list = ast.literal_eval(bbox)
        else:
            bbox_list = bbox
        x1, y1, x2, y2 = map(float, bbox_list)
        roi = image.crop((x1, y1, x2, y2))
        if self.transform:
            image = self.transform(image)
            roi = self.transform(roi)
        
        # Return original row data along with tensors
        return image, roi, idx

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

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = DualStreamBicycleClassifier(
        backbone_name=config['backbone'],
        pretrained=False,  # Not needed for inference
        num_classes=2,
        fc_hidden_dims=config['fc_dims'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.4f}")
    print(f"Model config: {config['name']}")
    
    return model, config

def run_inference(model, dataloader, device, dataset):
    """Run inference on dataset and return predictions with probabilities"""
    predictions = []
    probabilities = []
    indices = []
    
    print("Running inference...")
    with torch.no_grad():
        for imgs, rois, idx_batch in tqdm(dataloader, desc="Processing batches"):
            imgs, rois = imgs.to(device), rois.to(device)
            logits = model(imgs, rois)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            predictions.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            indices.extend(idx_batch.numpy().tolist())
    
    return predictions, probabilities, indices

def main():
    parser = argparse.ArgumentParser(description='Run inference on bicycle state dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to input feather file')
    parser.add_argument('--img_dir', type=str, default='',
                        help='Directory containing images')
    parser.add_argument('--output', type=str, default=None,
                        help='Output feather file path (default: auto-generated from model name)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    
    # Setup transform (no augmentation for inference)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"\nLoading data from: {args.data}")
    dataset = BicycleStateDataset(args.data, args.img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    print(f"Dataset size: {len(dataset)} images")
    
    # Run inference
    predictions, probabilities, indices = run_inference(model, dataloader, device, dataset)
    
    # Create results dataframe
    print("\nCreating results dataframe...")
    results_df = dataset.data.copy()
    results_df['predicted_class'] = predictions
    results_df['prob_class_0'] = [prob[0] for prob in probabilities]
    results_df['prob_class_1'] = [prob[1] for prob in probabilities]
    results_df['model_name'] = config['name']
    results_df['checkpoint_path'] = args.checkpoint
    
    # Calculate accuracy if ground truth exists
    if 'new_label' in results_df.columns:
        accuracy = (results_df['predicted_class'] == results_df['new_label']).mean()
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Print confusion matrix info
        tp = ((results_df['predicted_class'] == 1) & (results_df['new_label'] == 1)).sum()
        tn = ((results_df['predicted_class'] == 0) & (results_df['new_label'] == 0)).sum()
        fp = ((results_df['predicted_class'] == 1) & (results_df['new_label'] == 0)).sum()
        fn = ((results_df['predicted_class'] == 0) & (results_df['new_label'] == 1)).sum()
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            print(f"  Precision: {precision:.4f}")
        if (tp + fn) > 0:
            recall = tp / (tp + fn)
            print(f"  Recall: {recall:.4f}")
    
    # Generate output filename if not provided
    # python inference.py --checkpoint "hyperparam_search_20251024_125918/run_1_anti_overfit.pth" --data hand_labeled_data.feather --img_dir "data/nuscenes/datasets/v1.0-trainval"
    if args.output is None:
        input_basename = os.path.splitext(os.path.basename(args.data))[0]
        input_basename_path_head = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f"{input_basename_path_head}_{input_basename}_predictions_{config['name']}.feather"
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    results_df.reset_index(drop=True).to_feather(args.output)
    print("Done!")

if __name__ == "__main__":
    main()