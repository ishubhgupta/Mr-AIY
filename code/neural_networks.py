"""
Module for deep learning neural networks.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from tqdm import tqdm
import json
import sys
sys.path.append('..')
from database.db_setup import DatabaseManager
from code.utils import save_model, ensure_dir, format_time

class TabularDataset(Dataset):
    """
    Dataset class for tabular data.
    """
    def __init__(self, X, y=None):
        """
        Initialize the dataset.
        
        Args:
            X (numpy.ndarray): Feature data
            y (numpy.ndarray, optional): Target data
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class MLPModel(nn.Module):
    """
    Multi-layer Perceptron model.
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        """
        Initialize the model.
        
        Args:
            input_size (int): Number of input features
            hidden_sizes (list): List of hidden layer sizes
            output_size (int): Number of output units
            dropout_rate (float): Dropout rate
        """
        super(MLPModel, self).__init__()
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # If output_size is 1, this is a binary classification or regression problem
        if output_size == 1:
            layers.append(nn.Sigmoid())
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)


class LSTMModel(nn.Module):
    """
    LSTM model for sequence data.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        """
        Initialize the model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of the hidden state
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output units
            dropout_rate (float): Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # If output_size is 1, this is a binary classification or regression problem
        if output_size == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        # Apply activation if needed
        if self.activation:
            out = self.activation(out)
        
        return out


def prepare_tabular_data(data_path, target_column, test_size=0.2, random_state=42):
    """
    Prepare tabular data for training.
    
    Args:
        data_path (str): Path to the data file
        target_column (str): Name of the target column
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, scaler, label_encoder)
    """
    # Load data based on file extension
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV, Excel, or JSON.")
    
    # Handle missing values
    df = df.dropna()
    
    # Extract features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Convert categorical features to numeric
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        label_encoder = None
        y_encoded = y.values
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_names, scaler, label_encoder


def train_mlp_model(X_train, y_train, X_val, y_val, hidden_sizes=[64, 32], 
                   epochs=100, batch_size=32, learning_rate=0.001, model_save_dir=None):
    """
    Train a Multi-layer Perceptron model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_val (numpy.ndarray): Validation features
        y_val (numpy.ndarray): Validation targets
        hidden_sizes (list): Sizes of hidden layers
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        model_save_dir (str, optional): Directory to save the model
        
    Returns:
        dict: Training results
    """
    # Import streamlit for real-time visualization
    try:
        import streamlit as st
        use_streamlit = True
    except ImportError:
        use_streamlit = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get input and output sizes
    input_size = X_train.shape[1]
    
    # Check if it's binary classification, multi-class classification, or regression
    unique_classes = np.unique(np.concatenate([y_train, y_val]))
    
    # Determine task type and prepare targets
    if len(unique_classes) <= 2:
        # Binary classification
        output_size = 1
        criterion = nn.BCELoss()
        task_type = 'binary_classification'
        
        # For binary classification, ensure targets are 0 or 1
        y_train_processed = np.where(y_train == unique_classes[0], 0, 1).astype(np.float32)
        y_val_processed = np.where(y_val == unique_classes[0], 0, 1).astype(np.float32)
        
    elif len(unique_classes) > 2:
        # Multi-class classification
        output_size = len(unique_classes)
        criterion = nn.CrossEntropyLoss()
        task_type = 'multi_class_classification'
        
        # Create label mapping to ensure targets are in range [0, n_classes-1]
        label_map = {label: idx for idx, label in enumerate(sorted(unique_classes))}
        
        # Apply label mapping
        y_train_processed = np.array([label_map[label] for label in y_train], dtype=np.int64)
        y_val_processed = np.array([label_map[label] for label in y_val], dtype=np.int64)
        
        print(f"Label mapping: {label_map}")
        print(f"Unique training targets: {np.unique(y_train_processed)}")
        print(f"Output size: {output_size}")
        
    else:
        # Regression (continuous values)
        output_size = 1
        criterion = nn.MSELoss()
        task_type = 'regression'
        y_train_processed = y_train.astype(np.float32)
        y_val_processed = y_val.astype(np.float32)
    
    # Create datasets
    train_dataset = TabularDataset(X_train, y_train_processed)
    val_dataset = TabularDataset(X_val, y_val_processed)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = MLPModel(input_size, hidden_sizes, output_size)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Setup real-time visualization
    if use_streamlit:
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create placeholders for metrics and plots
        metrics_cols = st.columns(3)
        plot_placeholder = st.empty()
        
        # Create metric placeholders that will be updated in place
        with metrics_cols[0]:
            train_loss_metric = st.empty()
        with metrics_cols[1]:
            val_loss_metric = st.empty()
        with metrics_cols[2]:
            accuracy_metric = st.empty()

    
    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Starting training on {device}...")
    print(f"Task type: {task_type}")
    print(f"Input size: {input_size}, Output size: {output_size}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            
            # Adjust outputs and targets for loss calculation based on task type
            if task_type == 'binary_classification':
                loss = criterion(outputs.view(-1), y_batch)
            elif task_type == 'multi_class_classification':
                # Ensure targets are long integers and within valid range
                y_batch = y_batch.long()
                if torch.max(y_batch) >= output_size:
                    print(f"Warning: Target {torch.max(y_batch).item()} >= output_size {output_size}")
                    continue  # Skip this batch
                loss = criterion(outputs, y_batch)
            else:  # regression
                loss = criterion(outputs.view(-1), y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                
                # Adjust outputs and targets for loss calculation based on task type
                if task_type == 'binary_classification':
                    loss = criterion(outputs.view(-1), y_batch)
                    # Calculate accuracy for binary classification
                    preds = (outputs.view(-1) > 0.5).float()
                    epoch_correct += (preds == y_batch).sum().item()
                elif task_type == 'multi_class_classification':
                    y_batch = y_batch.long()
                    if torch.max(y_batch) >= output_size:
                        continue  # Skip this batch
                    loss = criterion(outputs, y_batch)
                    # Calculate accuracy for multi-class classification
                    preds = torch.argmax(outputs, dim=1)
                    epoch_correct += (preds == y_batch).sum().item()
                else:  # regression
                    loss = criterion(outputs.view(-1), y_batch)
                
                epoch_loss += loss.item() * X_batch.size(0)
                epoch_total += X_batch.size(0)
        
        # Calculate average validation loss and accuracy
        avg_val_loss = epoch_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Calculate accuracy (only for classification tasks)
        if task_type in ['binary_classification', 'multi_class_classification']:
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        else:
            accuracy = 0.0  # For regression, we don't calculate accuracy
        accuracies.append(accuracy)
        
        # Update real-time visualization
        if use_streamlit:
            # Update progress bar
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            
            # Update status text
            status_text.text(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Accuracy: {accuracy:.4f}')
            
            # Update metrics every 5 epochs or on the last epoch
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                # Update metrics in place using the placeholders
                with train_loss_metric.container():
                    st.metric("Training Loss", f"{avg_train_loss:.4f}")
                with val_loss_metric.container():
                    st.metric("Validation Loss", f"{avg_val_loss:.4f}")
                with accuracy_metric.container():
                    if task_type in ['binary_classification', 'multi_class_classification']:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    else:
                        st.metric("RMSE", f"{np.sqrt(avg_val_loss):.4f}")
                
                # Update plots
                with plot_placeholder.container():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Loss plot
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=train_losses,
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            y=val_losses,
                            mode='lines',
                            name='Validation Loss',
                            line=dict(color='red')
                        ))
                        fig.update_layout(
                            title='Training Progress',
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"loss_plot_{epoch}")
                    
                    with col2:
                        # Accuracy or additional metric plot
                        if task_type in ['binary_classification', 'multi_class_classification']:
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                y=accuracies,
                                mode='lines',
                                name='Accuracy',
                                line=dict(color='green')
                            ))
                            fig2.update_layout(
                                title='Model Accuracy',
                                xaxis_title='Epoch',
                                yaxis_title='Accuracy',
                                height=400
                            )
                            st.plotly_chart(fig2, use_container_width=True, key=f"acc_plot_{epoch}")
                        else:
                            # For regression, show RMSE
                            rmse_values = [np.sqrt(loss) for loss in val_losses]
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                y=rmse_values,
                                mode='lines',
                                name='RMSE',
                                line=dict(color='purple')
                            ))
                            fig2.update_layout(
                                title='Root Mean Square Error',
                                xaxis_title='Epoch',
                                yaxis_title='RMSE',
                                height=400
                            )
                            st.plotly_chart(fig2, use_container_width=True, key=f"rmse_plot_{epoch}")

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # Final update for Streamlit
    if use_streamlit:
        progress_bar.progress(1.0)
        status_text.text(f'✅ Training completed! Best validation loss: {best_val_loss:.4f}')
        st.balloons()
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate model on validation set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            
            # Convert outputs to predictions based on task type
            if task_type == 'binary_classification':
                preds = (outputs.view(-1) > 0.5).float()
            elif task_type == 'multi_class_classification':
                preds = torch.argmax(outputs, dim=1)
            else:  # regression
                preds = outputs.view(-1)
            
            # Collect predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    if task_type == 'binary_classification' or task_type == 'multi_class_classification':
        metrics['accuracy'] = accuracy_score(all_targets, all_preds)
        if len(np.unique(all_targets)) > 1:  # Avoid warning for single class
            metrics['precision'] = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    else:  # regression
        metrics['mse'] = ((np.array(all_targets) - np.array(all_preds)) ** 2).mean()
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.abs(np.array(all_targets) - np.array(all_preds)).mean()
    
    # Save model if directory is provided
    model_path = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_save_dir:
        ensure_dir(model_save_dir)
        model_path = os.path.join(model_save_dir, f"mlp_model_{timestamp}.pth")
        
        # Save model with metadata
        metadata = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'output_size': output_size,
            'task_type': task_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_val_loss': best_val_loss,
            'label_map': label_map if task_type == 'multi_class_classification' else None
        }
        
        model_dict = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }
        
        torch.save(model_dict, model_path)
        print(f"Model saved to: {model_path}")
    
    # Create static training plot for final results
    plt.figure(figsize=(15, 5))
    
    # Loss subplot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy subplot (for classification)
    if task_type in ['binary_classification', 'multi_class_classification']:
        plt.subplot(1, 3, 2)
        plt.plot(accuracies, label='Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.subplot(1, 3, 2)
        rmse_values = [np.sqrt(loss) for loss in val_losses]
        plt.plot(rmse_values, label='RMSE', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Root Mean Square Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Learning curve
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation')
    plt.fill_between(range(1, len(train_losses) + 1), train_losses, alpha=0.3, color='blue')
    plt.fill_between(range(1, len(val_losses) + 1), val_losses, alpha=0.3, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if model_save_dir is provided
    plot_path = None
    if model_save_dir:
        plot_path = os.path.join(model_save_dir, f"mlp_training_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {plot_path}")
    
    # Return results
    results = {
        'model': model,
        'model_path': model_path,
        'plot_path': plot_path,
        'training_time': format_time(training_time),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'accuracies': accuracies,
        'metrics': metrics,
        'task_type': task_type
    }
    
    return results


def train_lstm_model(X_train, y_train, X_val, y_val, seq_length=10, hidden_size=64, num_layers=2,
                    epochs=100, batch_size=32, learning_rate=0.001, model_save_dir=None):
    """
    Train an LSTM model for sequence data.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_val (numpy.ndarray): Validation features
        y_val (numpy.ndarray): Validation targets
        seq_length (int): Sequence length
        hidden_size (int): Size of the hidden state
        num_layers (int): Number of LSTM layers
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        model_save_dir (str, optional): Directory to save the model
        
    Returns:
        dict: Training results
    """
    # Import streamlit for real-time visualization
    try:
        import streamlit as st
        use_streamlit = True
    except ImportError:
        use_streamlit = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reshape data into sequences
    def create_sequences(X, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:i+seq_length])
            ys.append(y[i+seq_length])
        return np.array(Xs), np.array(ys)
    
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    
    # Get input size
    input_size = X_train.shape[1]
    
    # Check if it's binary classification, multi-class classification, or regression
    unique_classes = np.unique(np.concatenate([y_train_seq, y_val_seq]))
    
    # Determine task type and prepare targets
    if len(unique_classes) <= 2:
        # Binary classification
        output_size = 1
        criterion = nn.BCELoss()
        task_type = 'binary_classification'
        
        # For binary classification, ensure targets are 0 or 1
        y_train_processed = np.where(y_train_seq == unique_classes[0], 0, 1).astype(np.float32)
        y_val_processed = np.where(y_val_seq == unique_classes[0], 0, 1).astype(np.float32)
        
    elif len(unique_classes) > 2:
        # Multi-class classification
        output_size = len(unique_classes)
        criterion = nn.CrossEntropyLoss()
        task_type = 'multi_class_classification'
        
        # Create label mapping to ensure targets are in range [0, n_classes-1]
        label_map = {label: idx for idx, label in enumerate(sorted(unique_classes))}
        
        # Apply label mapping
        y_train_processed = np.array([label_map[label] for label in y_train_seq], dtype=np.int64)
        y_val_processed = np.array([label_map[label] for label in y_val_seq], dtype=np.int64)
        
        print(f"Label mapping: {label_map}")
        
    else:
        # Regression (continuous values)
        output_size = 1
        criterion = nn.MSELoss()
        task_type = 'regression'
        y_train_processed = y_train_seq.astype(np.float32)
        y_val_processed = y_val_seq.astype(np.float32)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_processed, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(y_val_processed, dtype=torch.float32)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Setup real-time visualization (similar to MLP)
    if use_streamlit:
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create placeholders for metrics and plots
        metrics_cols = st.columns(3)
        plot_placeholder = st.empty()
        
        # Create metric placeholders that will be updated in place
        with metrics_cols[0]:
            train_loss_metric = st.empty()
        with metrics_cols[1]:
            val_loss_metric = st.empty()
        with metrics_cols[2]:
            accuracy_metric = st.empty()

    
    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Starting LSTM training on {device}...")
    print(f"Task type: {task_type}")
    print(f"Input size: {input_size}, Output size: {output_size}")
    
    for epoch in range(epochs):
        # Training phase (similar structure to MLP but with LSTM-specific handling)
        model.train()
        epoch_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            
            # Adjust outputs and targets for loss calculation based on task type
            if task_type == 'binary_classification':
                loss = criterion(outputs.view(-1), y_batch)
            elif task_type == 'multi_class_classification':
                y_batch = y_batch.long()
                if torch.max(y_batch) >= output_size:
                    continue  # Skip this batch
                loss = criterion(outputs, y_batch)
            else:  # regression
                loss = criterion(outputs.view(-1), y_batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation phase (similar to MLP)
        model.eval()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                
                # Adjust outputs and targets for loss calculation based on task type
                if task_type == 'binary_classification':
                    loss = criterion(outputs.view(-1), y_batch)
                    preds = (outputs.view(-1) > 0.5).float()
                    epoch_correct += (preds == y_batch).sum().item()
                elif task_type == 'multi_class_classification':
                    y_batch = y_batch.long()
                    if torch.max(y_batch) >= output_size:
                        continue  # Skip this batch
                    loss = criterion(outputs, y_batch)
                    preds = torch.argmax(outputs, dim=1)
                    epoch_correct += (preds == y_batch).sum().item()
                else:  # regression
                    loss = criterion(outputs.view(-1), y_batch)
                
                epoch_loss += loss.item() * X_batch.size(0)
                epoch_total += X_batch.size(0)
        
        # Calculate average validation loss and accuracy
        avg_val_loss = epoch_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Calculate accuracy (only for classification tasks)
        if task_type in ['binary_classification', 'multi_class_classification']:
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        else:
            accuracy = 0.0
        accuracies.append(accuracy)
        
        # Update real-time visualization (similar to MLP)
        if use_streamlit:
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Accuracy: {accuracy:.4f}')
            
            # Update visualization every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                # Update metrics in place using the placeholders
                with train_loss_metric.container():
                    st.metric("Training Loss", f"{avg_train_loss:.4f}")
                with val_loss_metric.container():
                    st.metric("Validation Loss", f"{avg_val_loss:.4f}")
                with accuracy_metric.container():
                    if task_type in ['binary_classification', 'multi_class_classification']:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                    else:
                        st.metric("RMSE", f"{np.sqrt(avg_val_loss):.4f}")
                
                # Update plots (similar real-time plotting as MLP)
                with plot_placeholder.container():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Training Loss', line=dict(color='blue')))
                        fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss', line=dict(color='red')))
                        fig.update_layout(title='LSTM Training Progress', xaxis_title='Epoch', yaxis_title='Loss', height=400)
                        st.plotly_chart(fig, use_container_width=True, key=f"lstm_loss_plot_{epoch}")
                    
                    with col2:
                        if task_type in ['binary_classification', 'multi_class_classification']:
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(y=accuracies, mode='lines', name='Accuracy', line=dict(color='green')))
                            fig2.update_layout(title='LSTM Accuracy', xaxis_title='Epoch', yaxis_title='Accuracy', height=400)
                            st.plotly_chart(fig2, use_container_width=True, key=f"lstm_acc_plot_{epoch}")

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
    
    # Final update for Streamlit
    if use_streamlit:
        progress_bar.progress(1.0)
        status_text.text(f'✅ LSTM Training completed! Best validation loss: {best_val_loss:.4f}')
        st.balloons()
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Load best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model if directory is provided
    model_path = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if model_save_dir:
        ensure_dir(model_save_dir)
        model_path = os.path.join(model_save_dir, f"lstm_model_{timestamp}.pth")
        
        # Save model with metadata
        metadata = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'seq_length': seq_length,
            'task_type': task_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'best_val_loss': best_val_loss,
            'label_map': label_map if task_type == 'multi_class_classification' else None
        }
        
        model_dict = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }
        
        torch.save(model_dict, model_path)
        print(f"LSTM model saved to: {model_path}")
    
    # Create static training plot for final results
    plt.figure(figsize=(15, 5))
    
    # Loss subplot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy subplot (for classification)
    if task_type in ['binary_classification', 'multi_class_classification']:
        plt.subplot(1, 3, 2)
        plt.plot(accuracies, label='Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('LSTM Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.subplot(1, 3, 2)
        rmse_values = [np.sqrt(loss) for loss in val_losses]
        plt.plot(rmse_values, label='RMSE', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('LSTM RMSE')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Learning curve
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation')
    plt.fill_between(range(1, len(train_losses) + 1), train_losses, alpha=0.3, color='blue')
    plt.fill_between(range(1, len(val_losses) + 1), val_losses, alpha=0.3, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if model_save_dir is provided
    plot_path = None
    if model_save_dir:
        plot_path = os.path.join(model_save_dir, f"lstm_training_plot_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"LSTM training plot saved to: {plot_path}")
    
    # Return results
    results = {
        'model': model,
        'model_path': model_path,
        'plot_path': plot_path,
        'training_time': format_time(training_time),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'accuracies': accuracies,
        'metrics': metrics,
        'task_type': task_type
    }
    
    return results


def predict_with_nn_model(model_path, input_data, scaler=None, label_encoder=None, sequence_data=False, seq_length=10):
    """
    Make a prediction with a trained neural network model.
    
    Args:
        model_path (str): Path to the trained model
        input_data (numpy.ndarray): Input data for prediction
        scaler (sklearn.preprocessing.StandardScaler, optional): Scaler for standardizing input data
        label_encoder (sklearn.preprocessing.LabelEncoder, optional): Encoder for labels
        sequence_data (bool): Whether the input data is sequence data
        seq_length (int): Sequence length (for sequence data)
        
    Returns:
        dict: Prediction results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get metadata
    metadata = checkpoint.get('metadata', {})
    
    # Scale input data if scaler is provided
    if scaler:
        input_data = scaler.transform(input_data)
    
    # Create model based on metadata
    if 'num_layers' in metadata:  # LSTM model
        model = LSTMModel(
            input_size=metadata['input_size'],
            hidden_size=metadata['hidden_size'],
            num_layers=metadata['num_layers'],
            output_size=metadata['output_size']
        )
    else:  # MLP model
        model = MLPModel(
            input_size=metadata['input_size'],
            hidden_sizes=metadata['hidden_sizes'],
            output_size=metadata['output_size']
        )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Prepare input data
    if sequence_data:
        # Reshape data into sequences
        sequences = []
        for i in range(len(input_data) - seq_length + 1):
            sequences.append(input_data[i:i+seq_length])
        
        input_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
    else:
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Process outputs based on task type
    task_type = metadata.get('task_type', 'binary_classification')
    
    if task_type == 'binary_classification':
        probs = outputs.view(-1).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    elif task_type == 'multi_class_classification':
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    else:  # regression
        probs = None
        preds = outputs.view(-1).cpu().numpy()
    
    # Decode predictions if label_encoder is provided
    if label_encoder and task_type != 'regression':
        preds = label_encoder.inverse_transform(preds)
    
    # Create results dictionary
    results = {
        'predictions': preds,
        'probabilities': probs
    }
    
    return results


def evaluate_nn_model(model_path, X_test, y_test, scaler=None, label_encoder=None, sequence_data=False, seq_length=10):
    """
    Evaluate a trained neural network model on test data.
    
    Args:
        model_path (str): Path to the trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
        scaler (sklearn.preprocessing.StandardScaler, optional): Scaler for standardizing input data
        label_encoder (sklearn.preprocessing.LabelEncoder, optional): Encoder for labels
        sequence_data (bool): Whether the input data is sequence data
        seq_length (int): Sequence length (for sequence data)
        
    Returns:
        dict: Evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get metadata
    metadata = checkpoint.get('metadata', {})
    
    # Scale input data if scaler is provided
    if scaler:
        X_test = scaler.transform(X_test)
    
    # Create model based on metadata
    if 'num_layers' in metadata:  # LSTM model
        model = LSTMModel(
            input_size=metadata['input_size'],
            hidden_size=metadata['hidden_size'],
            num_layers=metadata['num_layers'],
            output_size=metadata['output_size']
        )
    else:  # MLP model
        model = MLPModel(
            input_size=metadata['input_size'],
            hidden_sizes=metadata['hidden_sizes'],
            output_size=metadata['output_size']
        )
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Prepare data for LSTM
    if sequence_data:
        # Reshape data into sequences
        def create_sequences(X, y, seq_length):
            Xs, ys = [], []
            for i in range(len(X) - seq_length):
                Xs.append(X[i:i+seq_length])
                ys.append(y[i+seq_length])
            return np.array(Xs), np.array(ys)
        
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
        
        # Create dataset and dataloader
        test_dataset = TensorDataset(
            torch.tensor(X_test_seq, dtype=torch.float32),
            torch.tensor(y_test_seq, dtype=torch.float32)
        )
    else:
        # Create dataset for MLP
        test_dataset = TabularDataset(X_test, y_test)
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get task type
    task_type = metadata.get('task_type', 'binary_classification')
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            
            # Convert outputs to predictions based on task type
            if task_type == 'binary_classification':
                preds = (outputs.view(-1) > 0.5).float()
            elif task_type == 'multi_class_classification':
                preds = torch.argmax(outputs, dim=1)
            else:  # regression
                preds = outputs.view(-1)
            
            # Collect predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    # Calculate metrics
    metrics = {}
    if task_type == 'binary_classification' or task_type == 'multi_class_classification':
        metrics['accuracy'] = accuracy_score(all_targets, all_preds)
        metrics['precision'] = precision_score(all_targets, all_preds, average='weighted')
        metrics['recall'] = recall_score(all_targets, all_preds, average='weighted')
        metrics['f1'] = f1_score(all_targets, all_preds, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # If we have a label encoder, use the class names
        if label_encoder and task_type == 'multi_class_classification':
            class_names = label_encoder.classes_
        else:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save or display the plot
        if model_path:
            model_dir = os.path.dirname(model_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cm_plot_path = os.path.join(model_dir, f"confusion_matrix_{timestamp}.png")
            plt.savefig(cm_plot_path)
            metrics['confusion_matrix_plot'] = cm_plot_path
    else:  # regression
        metrics['mse'] = ((np.array(all_targets) - np.array(all_preds)) ** 2).mean()
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.abs(np.array(all_targets) - np.array(all_preds)).mean()
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(all_targets, all_preds, alpha=0.5)
        plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        
        # Save or display the plot
        if model_path:
            model_dir = os.path.dirname(model_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(model_dir, f"actual_vs_predicted_{timestamp}.png")
            plt.savefig(plot_path)
            metrics['actual_vs_predicted_plot'] = plot_path
    
    # Return results
    results = {
        'metrics': metrics,
        'task_type': task_type,
        'num_test_samples': len(all_targets)
    }
    
    return results
