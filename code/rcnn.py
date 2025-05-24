"""
Module for RCNN-based object detection.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import FasterRCNN, MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import cv2
import sys
sys.path.append('..')
from database.db_setup import DatabaseManager
from code.utils import preprocess_image, save_model, ensure_dir, format_time

class ObjectDetectionDataset(Dataset):
    """
    Dataset class for object detection.
    """
    def __init__(self, data_dir, annotation_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory with all the images
            annotation_file (str): Path to JSON annotation file
            transform (callable, optional): Transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get class names
        self.classes = sorted(set(ann['category'] for img in self.annotations.values() for ann in img))
        self.class_to_idx = {cls_name: i + 1 for i, cls_name in enumerate(self.classes)}  # +1 because 0 is background
        
        # Get image paths
        self.image_paths = sorted(list(self.annotations.keys()))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image_id = torch.tensor([idx])
        
        # Get annotations for this image
        img_anns = self.annotations[self.image_paths[idx]]
        
        # Convert annotations to tensors
        boxes = []
        labels = []
        masks = []
        
        for ann in img_anns:
            # Get box coordinates [x1, y1, x2, y2]
            x1, y1, w, h = ann['bbox']
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            
            # Get class label
            labels.append(self.class_to_idx[ann['category']])
            
            # Add mask if it exists
            if 'segmentation' in ann:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
                for seg in ann['segmentation']:
                    poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly], 1)
                masks.append(mask)
        
        # Convert to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            if masks:
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
                
                target = {
                    'boxes': boxes,
                    'labels': labels,
                    'masks': masks,
                    'image_id': image_id,
                    'area': area,
                    'iscrowd': iscrowd
                }
            else:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
                
                target = {
                    'boxes': boxes,
                    'labels': labels,
                    'image_id': image_id,
                    'area': area,
                    'iscrowd': iscrowd
                }
        else:
            # Return empty targets
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': image_id,
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
            if masks:
                target['masks'] = torch.zeros((0, image.height, image.width), dtype=torch.uint8)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, target


def get_faster_rcnn_model(num_classes):
    """
    Create a Faster R-CNN model with a ResNet-50-FPN backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        
    Returns:
        torchvision.models.detection.FasterRCNN: Faster R-CNN model
    """
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the pre-trained head with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def get_mask_rcnn_model(num_classes):
    """
    Create a Mask R-CNN model with a ResNet-50-FPN backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        
    Returns:
        torchvision.models.detection.MaskRCNN: Mask R-CNN model
    """
    # Load a pre-trained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the pre-trained head with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model


def train_rcnn_model(data_dir, annotation_file, model_save_dir, model_type='faster_rcnn', 
                     epochs=10, batch_size=2, learning_rate=0.005, val_split=0.2):
    """
    Train an RCNN model for object detection.
    
    Args:
        data_dir (str): Directory containing the dataset
        annotation_file (str): Path to JSON annotation file
        model_save_dir (str): Directory to save the model
        model_type (str): Type of model ('faster_rcnn' or 'mask_rcnn')
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        val_split (float): Validation split ratio
        
    Returns:
        dict: Training results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create dataset
    dataset = ObjectDetectionDataset(data_dir=data_dir, annotation_file=annotation_file, transform=transform)
    
    # Split into train and validation sets
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))  # This is needed for variable-sized objects like images and targets
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Create model
    num_classes = len(dataset.classes) + 1  # +1 for background
    
    if model_type == 'faster_rcnn':
        model = get_faster_rcnn_model(num_classes)
    elif model_type == 'mask_rcnn':
        model = get_mask_rcnn_model(num_classes)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'faster_rcnn' or 'mask_rcnn'.")
    
    model = model.to(device)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Define learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += losses.item()
            
            # Update progress bar
            pbar.set_postfix(loss=losses.item())
        
        # Update learning rate
        lr_scheduler.step()
        
        # Calculate average loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, targets in pbar:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Update statistics
                epoch_loss += losses.item()
                
                # Update progress bar
                pbar.set_postfix(loss=losses.item())
        
        # Calculate average loss
        avg_val_loss = epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save the best model
    ensure_dir(model_save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_save_dir, f"{model_type}_{timestamp}.pth")
    
    # If best model was found, load it before saving
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model with metadata
    metadata = {
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx,
        'num_classes': num_classes,
        'model_type': model_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'val_split': val_split,
        'best_val_loss': best_val_loss,
        'training_time': training_time
    }
    
    model_dict = {
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }
    
    torch.save(model_dict, model_path)
    
    # Record in database
    db_manager = DatabaseManager()
    model_id = db_manager.add_model(
        name=f"{model_type.upper()}_{timestamp}",
        model_type=model_type.upper(),
        file_path=model_path,
        parameters={"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate},
        loss=best_val_loss,
        description=f"{model_type.upper()} model for object detection"
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(model_save_dir, f"{model_type}_training_plot_{timestamp}.png")
    plt.savefig(plot_path)
    
    # Return results
    results = {
        'model_path': model_path,
        'model_id': model_id,
        'classes': dataset.classes,
        'training_time': format_time(training_time),
        'best_val_loss': best_val_loss,
        'final_val_loss': val_losses[-1],
        'plot_path': plot_path
    }
    
    return results


def detect_objects(model_path, image_path, confidence_threshold=0.5):
    """
    Detect objects in an image using a trained RCNN model.
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the image
        confidence_threshold (float): Confidence threshold for detections
        
    Returns:
        dict: Detection results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get metadata
    metadata = checkpoint.get('metadata', {})
    classes = metadata.get('classes', [])
    num_classes = metadata.get('num_classes', 0)
    model_type = metadata.get('model_type', 'faster_rcnn')
    
    # Create model and load weights
    if model_type == 'faster_rcnn':
        model = get_faster_rcnn_model(num_classes)
    elif model_type == 'mask_rcnn':
        model = get_mask_rcnn_model(num_classes)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'faster_rcnn' or 'mask_rcnn'.")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    
    # Extract predictions above threshold
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    # If mask_rcnn, also get masks
    masks = None
    if model_type == 'mask_rcnn' and 'masks' in prediction:
        masks = prediction['masks'].cpu().numpy()
    
    # Filter by confidence threshold
    keep = scores >= confidence_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    if masks is not None:
        masks = masks[keep]
    
    # Map labels to class names
    class_names = ['background'] + classes  # Add background class at index 0
    label_names = [class_names[label] for label in labels]
    
    # Prepare results
    detections = []
    for i in range(len(boxes)):
        detection = {
            'box': boxes[i].tolist(),
            'label': label_names[i],
            'score': float(scores[i])
        }
        if masks is not None:
            detection['mask'] = masks[i][0].tolist()  # Take the first channel of the mask
        detections.append(detection)
    
    # Create results dictionary
    results = {
        'image_path': image_path,
        'detections': detections,
        'num_detections': len(detections)
    }
    
    # Record prediction in database
    db_manager = DatabaseManager()
    model_info = db_manager.get_models()
    
    # Find the model_id based on the model_path
    model_id = None
    for model in model_info:
        if model[4] == model_path:  # Assuming file_path is at index 4
            model_id = model[0]  # Assuming id is at index 0
            break
    
    if model_id:
        db_manager.add_prediction(
            model_id=model_id,
            input_data=image_path,
            output_result=json.dumps(detections),
            confidence=float(np.mean(scores)) if len(scores) > 0 else 0.0
        )
    
    return results


def evaluate_rcnn_model(model_path, data_dir, annotation_file, iou_threshold=0.5):
    """
    Evaluate a trained RCNN model on a test dataset.
    
    Args:
        model_path (str): Path to the trained model
        data_dir (str): Directory containing the test dataset
        annotation_file (str): Path to JSON annotation file
        iou_threshold (float): IoU threshold for correct detection
        
    Returns:
        dict: Evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get metadata
    metadata = checkpoint.get('metadata', {})
    num_classes = metadata.get('num_classes', 0)
    model_type = metadata.get('model_type', 'faster_rcnn')
    
    # Create model and load weights
    if model_type == 'faster_rcnn':
        model = get_faster_rcnn_model(num_classes)
    elif model_type == 'mask_rcnn':
        model = get_mask_rcnn_model(num_classes)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'faster_rcnn' or 'mask_rcnn'.")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = ObjectDetectionDataset(data_dir=data_dir, annotation_file=annotation_file, transform=transforms.ToTensor())
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize metrics
    total_gt = 0
    total_pred = 0
    total_correct = 0
    class_metrics = {cls: {'gt': 0, 'pred': 0, 'correct': 0} for cls in dataset.classes}
    
    # Evaluate
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get ground truth boxes and labels
            gt_boxes = targets[0]['boxes'].cpu().numpy()
            gt_labels = targets[0]['labels'].cpu().numpy()
            
            # Make prediction
            prediction = model(images)[0]
            
            # Get predicted boxes, labels, and scores
            pred_boxes = prediction['boxes'].cpu().numpy()
            pred_labels = prediction['labels'].cpu().numpy()
            pred_scores = prediction['scores'].cpu().numpy()
            
            # Count ground truth objects
            total_gt += len(gt_boxes)
            for i, label in enumerate(gt_labels):
                class_name = dataset.classes[label - 1]  # -1 because label 0 is background
                class_metrics[class_name]['gt'] += 1
            
            # Count predicted objects
            total_pred += len(pred_boxes)
            for i, label in enumerate(pred_labels):
                if label <= len(dataset.classes):  # Ensure label is valid
                    class_name = dataset.classes[label - 1]
                    class_metrics[class_name]['pred'] += 1
            
            # Calculate IoU for each prediction with ground truth
            for i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
                if pred_score < 0.5:  # Skip low confidence predictions
                    continue
                
                for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if pred_label == gt_label:
                        # Calculate IoU
                        x1 = max(pred_box[0], gt_box[0])
                        y1 = max(pred_box[1], gt_box[1])
                        x2 = min(pred_box[2], gt_box[2])
                        y2 = min(pred_box[3], gt_box[3])
                        
                        if x2 < x1 or y2 < y1:
                            continue
                        
                        intersection = (x2 - x1) * (y2 - y1)
                        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                        union = pred_area + gt_area - intersection
                        
                        iou = intersection / union
                        
                        if iou >= iou_threshold:
                            total_correct += 1
                            class_name = dataset.classes[pred_label - 1]
                            class_metrics[class_name]['correct'] += 1
                            break  # Each ground truth can only be matched once
    
    # Calculate precision, recall, and F1 score
    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-class metrics
    class_results = {}
    for cls, metrics in class_metrics.items():
        cls_precision = metrics['correct'] / metrics['pred'] if metrics['pred'] > 0 else 0
        cls_recall = metrics['correct'] / metrics['gt'] if metrics['gt'] > 0 else 0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
        
        class_results[cls] = {
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1,
            'gt_count': metrics['gt'],
            'pred_count': metrics['pred'],
            'correct_count': metrics['correct']
        }
    
    # Return results
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'total_correct': total_correct,
        'class_results': class_results,
        'iou_threshold': iou_threshold
    }
    
    return results


def visualize_detection(image_path, detection_results, output_path=None, show_masks=True):
    """
    Visualize object detection results on an image.
    
    Args:
        image_path (str): Path to the original image
        detection_results (dict): Detection results from detect_objects
        output_path (str, optional): Path to save the output image
        show_masks (bool): Whether to show masks (if available)
        
    Returns:
        PIL.Image: Annotated image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Define colors for each class
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
    ]
    
    # Draw detections
    detections = detection_results['detections']
    unique_labels = set(d['label'] for d in detections)
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    for i, detection in enumerate(detections):
        box = detection['box']
        label = detection['label']
        score = detection['score']
        color = label_to_color[label]
        
        # Draw bounding box
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=3)
        
        # Draw label and score
        text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1] - 10), text, fill=color)
        
        # Draw mask if available and requested
        if show_masks and 'mask' in detection:
            mask = np.array(detection['mask'])
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.convert('L')
            
            # Create a colored mask
            colored_mask = Image.new('RGBA', image.size, (0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(colored_mask)
            
            # Find mask contours
            mask_np = np.array(mask_img)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            for contour in contours:
                contour = contour.reshape(-1, 2)
                coords = [(point[0], point[1]) for point in contour]
                if len(coords) > 2:  # Need at least 3 points for a polygon
                    mask_draw.polygon(coords, fill=(*color, 64))  # Semi-transparent fill
            
            # Composite the mask with the image
            image = Image.alpha_composite(image.convert('RGBA'), colored_mask).convert('RGB')
            draw = ImageDraw.Draw(image)  # Create a new draw object for the composited image
    
    # Save output image if path is provided
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        image.save(output_path)
    
    return image
