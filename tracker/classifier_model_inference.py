import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from nuscenes.nuscenes import NuScenes
from torchvision import transforms
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)


def _default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class DualStreamBicycleClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        num_classes: int = 2,
        fc_hidden_dims=None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if fc_hidden_dims is None:
            fc_hidden_dims = [256]

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
        else:
            raise ValueError(f"Unsupported backbone '{backbone_name}'")

        self.scene_encoder = nn.Sequential(*list(backbone_scene.children())[:-1])
        self.roi_encoder = nn.Sequential(*list(backbone_roi.children())[:-1])

        feat_dim = backbone_scene.fc.in_features
        self.fc_scene = nn.Linear(feat_dim, fc_hidden_dims[0])
        self.fc_roi = nn.Linear(feat_dim, fc_hidden_dims[0])

        classifier_layers = []
        input_dim = fc_hidden_dims[0] * 2
        for hidden_dim in fc_hidden_dims:
            classifier_layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            input_dim = hidden_dim
        classifier_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, full_img: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
        scene_feat = self.scene_encoder(full_img).flatten(1)
        roi_feat = self.roi_encoder(roi).flatten(1)
        scene_feat = torch.relu(self.fc_scene(scene_feat))
        roi_feat = torch.relu(self.fc_roi(roi_feat))
        fused = torch.cat([scene_feat, roi_feat], dim=1)
        return self.classifier(fused)


class MotionClassifierInference:
    """Wraps the dual-stream bicycle classifier for on-demand inference."""

    def __init__(
        self,
        checkpoint_path: str,
        nusc: NuScenes,
        device: Optional[Union[str, torch.device]] = None,
        transform: Optional[transforms.Compose] = None,
        stationary_class_index: int = 1,
    ) -> None:
        if nusc is None:
            raise ValueError("NuScenes handle must be provided for image retrieval")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.nusc = nusc
        self.transform = transform or _default_transform()
        self.stationary_class_index = stationary_class_index

        self.model, self.model_config = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> Tuple[DualStreamBicycleClassifier, Dict]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "config" not in checkpoint:
            raise KeyError("Checkpoint missing configuration payload under 'config'")
        config = checkpoint["config"]

        model = DualStreamBicycleClassifier(
            backbone_name=config.get("backbone", "resnet18"),
            pretrained=False,
            num_classes=config.get("num_classes", 2),
            fc_hidden_dims=config.get("fc_dims", [256]),
            dropout=config.get("dropout", 0.3),
        )

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError("Checkpoint missing 'model_state_dict'")

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model, config

    def predict_bbox(
        self,
        bbox: Dict,
        sample_token: str,
        annotate: bool = False,
        expand_pixels: int = 0,
    ) -> Dict:
        if bbox is None:
            raise ValueError("bbox payload is required")
        if "bbox_image" not in bbox:
            raise KeyError("bbox must contain 'bbox_image' entry")
        camera_type = bbox["bbox_image"].get("camera_type")
        if not camera_type:
            raise KeyError("bbox['bbox_image'] must include 'camera_type'")

        full_image, image_path = self._load_image(sample_token, camera_type)
        roi = self._crop_roi(full_image, bbox["bbox_image"].get("x1y1x2y2"), expand_pixels)

        if annotate:
            annotated = full_image.copy()
            draw = ImageDraw.Draw(annotated)
            draw.rectangle(self._to_rectangle(bbox["bbox_image"].get("x1y1x2y2")), outline="red", width=3)
        else:
            annotated = None

        full_tensor = self.transform(full_image).unsqueeze(0).to(self.device)
        roi_tensor = self.transform(roi).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(full_tensor, roi_tensor)
            probs = F.softmax(logits, dim=1).cpu()[0]
            pred_idx = int(logits.argmax(dim=1).cpu().item())

        if not 0 <= self.stationary_class_index < probs.shape[0]:
            raise IndexError(
                f"stationary_class_index {self.stationary_class_index} out of range for probabilities"
            )
        stationary_prob = float(probs[self.stationary_class_index])
        response = {
            "stationary_probability": stationary_prob,
            "raw_probabilities": probs.tolist(),
            "prediction": pred_idx,
            "image_path": image_path,
        }

        if annotate:
            response["annotated_image"] = annotated
            response["roi_image"] = roi

        return response

    def _load_image(self, sample_token: str, camera_type: str) -> Tuple[Image.Image, str]:
        sample = self.nusc.get("sample", sample_token)
        sample_data_token = sample["data"].get(camera_type)
        if sample_data_token is None:
            raise KeyError(f"Sample {sample_token} has no data for camera {camera_type}")

        image_path, _, _ = self.nusc.get_sample_data(sample_data_token)
        image = Image.open(image_path).convert("RGB")
        return image, image_path

    @staticmethod
    def _crop_roi(image: Image.Image, coords, expand_pixels: int) -> Image.Image:
        if coords is None:
            raise ValueError("ROI coordinates are required")
        if len(coords) != 4:
            raise ValueError("ROI coordinates must be of length 4")

        width, height = image.size
        x1, y1, x2, y2 = coords
        x1 = math.floor(max(0, x1 - expand_pixels))
        y1 = math.floor(max(0, y1 - expand_pixels))
        x2 = math.ceil(min(width, x2 + expand_pixels))
        y2 = math.ceil(min(height, y2 + expand_pixels))
        if x1 >= x2 or y1 >= y2:
            raise ValueError("Invalid ROI after clipping")
        return image.crop((x1, y1, x2, y2))

    @staticmethod
    def _to_rectangle(coords) -> Tuple[int, int, int, int]:
        if coords is None or len(coords) != 4:
            raise ValueError("Bounding box coordinates must contain four values")
        x1, y1, x2, y2 = coords
        return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
