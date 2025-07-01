"""
Inference module for the Drone Detection Flask App.

This script contains the core logic for:
- Defining the application's configuration.
- Loading the pre-trained Faster R-CNN model.
- Pre-processing an input image.
- Running the model to get predictions.
- Drawing bounding boxes with clear, readable labels using Pillow.
"""
import torch
import torchvision.ops as ops
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet18
# --- Import Pillow for custom drawing ---
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataclasses import dataclass, field
import numpy as np

@dataclass
class AppConfig:
    """Configuration class for the inference application."""
    # Model Hyperparameters
    NUM_CLASSES: int = 4  # 3 drone classes + 1 background
    IMG_SIZE: int = 600

    # Prediction Thresholds
    PREDICTION_SCORE_THRESHOLD: float = 0.5
    PREDICTION_IOU_THRESHOLD: float = 0.4

    # Class Mappings (ID to Name)
    ID_TO_CAT_MAP: dict = field(default_factory=lambda: {
        1: 'drone',
        2: 'small_drone',
        3: 'large_drone'
    })

def create_detection_model(num_classes: int) -> FasterRCNN:
    """Creates the same Faster R-CNN model structure used for training."""
    backbone = resnet18(pretrained=False)
    return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
    in_channels_list = [128, 256, 512]
    out_channels = 256
    backbone_fpn = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )

    model = FasterRCNN(
        backbone_fpn,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )
    return model

def get_inference_transform(img_size: int) -> A.Compose:
    """Returns the transformation pipeline for inference."""
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2()
    ])

# --- THIS IS THE UPDATED FUNCTION ---
def run_prediction(image_path: str, model: FasterRCNN, device: torch.device, config: AppConfig) -> np.ndarray:
    """
    Loads an image, runs prediction, and returns an image with clear,
    professional-looking bounding boxes drawn using Pillow.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    transform = get_inference_transform(config.IMG_SIZE)
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].float() / 255.0
    image_tensor = image_tensor.to(device).unsqueeze(0)

    # Run inference
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # Filter predictions
    keep = prediction['scores'] >= config.PREDICTION_SCORE_THRESHOLD
    boxes = prediction['boxes'][keep]
    scores = prediction['scores'][keep]
    labels = prediction['labels'][keep]

    nms_keep_indices = ops.nms(boxes, scores, config.PREDICTION_IOU_THRESHOLD)
    boxes = boxes[nms_keep_indices].cpu()
    labels = labels[nms_keep_indices].cpu()
    scores = scores[nms_keep_indices].cpu()

    label_names = [f"{config.ID_TO_CAT_MAP.get(l.item(), 'Unknown')}: {s:.2f}" for l, s in zip(labels, scores)]

    # --- NEW CUSTOM DRAWING LOGIC ---
    # Convert tensor back to a Pillow Image to draw on
    img_to_draw = Image.fromarray((image_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_to_draw)

    # Use Pillow's default font. For a custom font, you'd use:
    # font = ImageFont.truetype("path/to/your/font.ttf", size=20)
    try:
        # On some systems, a default sans-serif font can be found.
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        # If not, fall back to the basic default font.
        font = ImageFont.load_default()

    for box, label in zip(boxes, label_names):
        x1, y1, x2, y2 = box.tolist()
        
        # Draw the main bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        
        # Draw text with a background rectangle for readability
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        rect_y1 = y1 - text_height - 5
        if rect_y1 < 0: # If the box would go off-screen, place it inside
            rect_y1 = y1 + 5
        
        rect_x1 = x1
        rect_x2 = rect_x1 + text_width + 10
        rect_y2 = rect_y1 + text_height + 5

        draw.rectangle([(rect_x1, rect_y1), (rect_x2, rect_y2)], fill="red")
        draw.text((rect_x1 + 5, rect_y1), label, fill="white", font=font)

    # Return the final image as a NumPy array for Flask to save
    return np.array(img_to_draw)