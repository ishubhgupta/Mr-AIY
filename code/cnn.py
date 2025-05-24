"""
Module for CNN-based image classification.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('..')
from database.db_setup import DatabaseManager
from code.utils import preprocess_image, save_model, ensure_dir, format_time

class ImageDataset(Dataset):
    """
    Dataset class for loading images for classification.
    """
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class CNNModel(nn.Module):
    """
    CNN model for image classification.
    """
    def __init__(self, num_classes, pretrained=True):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super(CNNModel, self).__init__()
        
        # Use ResNet18 as the base model
        self.model = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)


def train_cnn_model(data_dir, model_save_dir, epochs=10, batch_size=32, learning_rate=0.001, 
                    val_split=0.2, pretrained=True):
    """
    Train a CNN model for image classification.
    
    Args:
        data_dir (str): Directory containing the dataset
        model_save_dir (str): Directory to save the model
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        val_split (float): Validation split ratio
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        dict: Training results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = ImageDataset(root_dir=data_dir, transform=train_transform)
    
    # Split into train and validation sets
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    num_classes = len(full_dataset.classes)
    model = CNNModel(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save the best model
    ensure_dir(model_save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_save_dir, f"cnn_model_{timestamp}.pth")
    
    # If best model was found, load it before saving
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model with metadata
    metadata = {
        'classes': full_dataset.classes,
        'class_to_idx': full_dataset.class_to_idx,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'val_split': val_split,
        'pretrained': pretrained,
        'best_val_loss': best_val_loss,
        'best_val_acc': val_accuracies[val_losses.index(best_val_loss)],
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
        name=f"CNN_{timestamp}",
        model_type="CNN",
        file_path=model_path,
        parameters={"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate},
        accuracy=val_accuracies[-1],
        loss=val_losses[-1],
        description="CNN model for image classification"
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training')
    plt.plot(val_accuracies, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(model_save_dir, f"cnn_training_plot_{timestamp}.png")
    plt.savefig(plot_path)
    
    # Return results
    results = {
        'model_path': model_path,
        'model_id': model_id,
        'classes': full_dataset.classes,
        'training_time': format_time(training_time),
        'best_val_loss': best_val_loss,
        'best_val_acc': val_accuracies[val_losses.index(best_val_loss)],
        'final_val_loss': val_losses[-1],
        'final_val_acc': val_accuracies[-1],
        'plot_path': plot_path
    }
    
    return results


def classify_image(model_path, image_path, top_k=5):
    """
    Classify an image using a trained model.
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the image
        top_k (int): Number of top predictions to return
        
    Returns:
        dict: Classification results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get metadata
    metadata = checkpoint.get('metadata', {})
    classes = metadata.get('classes', [])
    num_classes = len(classes)
    
    # Create model and load weights
    model = CNNModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    # Convert to lists
    top_probs = top_probs.cpu().numpy().tolist()
    top_indices = top_indices.cpu().numpy().tolist()
    
    # Map indices to class names
    top_classes = [classes[idx] for idx in top_indices]
    
    # Create results dictionary
    results = {
        'image_path': image_path,
        'top_classes': top_classes,
        'top_probabilities': top_probs,
        'prediction': top_classes[0],
        'confidence': top_probs[0]
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
            output_result=results['prediction'],
            confidence=results['confidence']
        )
    
    return results


def evaluate_model(model_path, test_data_dir):
    """
    Evaluate a trained model on a test dataset.
    
    Args:
        model_path (str): Path to the trained model
        test_data_dir (str): Directory containing the test dataset
        
    Returns:
        dict: Evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get metadata
    metadata = checkpoint.get('metadata', {})
    classes = metadata.get('classes', [])
    class_to_idx = metadata.get('class_to_idx', {})
    num_classes = len(classes)
    
    # Create model and load weights
    model = CNNModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    test_dataset = ImageDataset(root_dir=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                class_correct[label] += (pred == label).item()
                class_total[label] += 1
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[classes[i]] = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracy[classes[i]] = 0
    
    # Return results
    results = {
        'accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'num_test_samples': total
    }
    
    return results
