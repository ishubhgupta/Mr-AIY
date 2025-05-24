"""
Module for Tabular Generative Adversarial Networks (GANs).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import time
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.append('..')
from database.db_setup import DatabaseManager
from code.utils import save_model, ensure_dir, format_time
from sklearn.preprocessing import LabelEncoder, StandardScaler

class TabularGenerator(nn.Module):
    """
    Generator model for tabular data.
    """
    def __init__(self, latent_dim, output_dim, layers=[128, 256, 128]):
        """
        Initialize the generator.
        
        Args:
            latent_dim (int): Dimension of the latent space
            output_dim (int): Dimension of the output (number of features)
            layers (list): List of hidden layer sizes
        """
        super(TabularGenerator, self).__init__()
        
        # Create a list to hold all layers
        model_layers = []
        
        # Input layer
        model_layers.append(nn.Linear(latent_dim, layers[0]))
        model_layers.append(nn.BatchNorm1d(layers[0]))
        model_layers.append(nn.ReLU(inplace=True))
        
        # Hidden layers
        for i in range(len(layers) - 1):
            model_layers.append(nn.Linear(layers[i], layers[i+1]))
            model_layers.append(nn.BatchNorm1d(layers[i+1]))
            model_layers.append(nn.ReLU(inplace=True))
        
        # Output layer
        model_layers.append(nn.Linear(layers[-1], output_dim))
        model_layers.append(nn.Tanh())  # Output in range [-1, 1]
        
        self.model = nn.Sequential(*model_layers)
    
    def forward(self, z):
        """
        Forward pass.
        
        Args:
            z (torch.Tensor): Input noise tensor
            
        Returns:
            torch.Tensor: Generated data
        """
        return self.model(z)

class TabularDiscriminator(nn.Module):
    """
    Discriminator model for tabular data.
    """
    def __init__(self, input_dim, layers=[256, 128, 64]):
        """
        Initialize the discriminator.
        
        Args:
            input_dim (int): Dimension of the input data
            layers (list): List of hidden layer sizes
        """
        super(TabularDiscriminator, self).__init__()
        
        # Create a list to hold all layers
        model_layers = []
        
        # Input layer
        model_layers.append(nn.Linear(input_dim, layers[0]))
        model_layers.append(nn.LeakyReLU(0.2, inplace=True))
        model_layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        for i in range(len(layers) - 1):
            model_layers.append(nn.Linear(layers[i], layers[i+1]))
            model_layers.append(nn.LeakyReLU(0.2, inplace=True))
            model_layers.append(nn.Dropout(0.3))
        
        # Output layer
        model_layers.append(nn.Linear(layers[-1], 1))
        model_layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*model_layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input data tensor
            
        Returns:
            torch.Tensor: Discrimination result
        """
        return self.model(x)

class TabularDataset(Dataset):
    """
    Dataset class for tabular data.
    """
    def __init__(self, data):
        """
        Initialize the dataset.
        
        Args:
            data (numpy.ndarray): Preprocessed tabular data
        """
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class TabularPreprocessor:
    """
    Preprocessing utilities for tabular data.
    """
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.columns = []
        self.numeric_columns = []
        self.categorical_columns = []
    
    def fit_transform(self, data):
        """
        Fit and transform the data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Transformed data
        """
        self.columns = data.columns.tolist()
        transformed_data = data.copy()
        
        # Separate numeric and categorical columns
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        # Handle categorical columns
        for col in self.categorical_columns:
            encoder = LabelEncoder()
            transformed_data[col] = encoder.fit_transform(data[col].astype(str))
            self.encoders[col] = encoder
        
        # Handle numeric columns
        for col in self.numeric_columns:
            scaler = StandardScaler()
            transformed_data[col] = scaler.fit_transform(data[[col]])
            self.scalers[col] = scaler
        
        return transformed_data.values.astype(np.float32)
    
    def inverse_transform(self, data):
        """
        Inverse transform the data back to original format.
        
        Args:
            data (np.ndarray): Transformed data
            
        Returns:
            pd.DataFrame: Original format data
        """
        df = pd.DataFrame(data, columns=self.columns)
        
        # Inverse transform numeric columns
        for col in self.numeric_columns:
            if col in self.scalers:
                df[col] = self.scalers[col].inverse_transform(df[[col]])
        
        # Inverse transform categorical columns
        for col in self.categorical_columns:
            if col in self.encoders:
                # Clip values to valid range
                min_val = 0
                max_val = len(self.encoders[col].classes_) - 1
                df[col] = np.clip(df[col].round().astype(int), min_val, max_val)
                df[col] = self.encoders[col].inverse_transform(df[col])
        
        return df

def train_tabular_gan(data, model_save_dir, latent_dim=100, generator_layers=[128, 256, 128],
                     discriminator_layers=[256, 128, 64], batch_size=64, n_epochs=500, learning_rate=0.0002):
    """
    Train a Tabular GAN model.
    
    Args:
        data (pd.DataFrame): Training data
        model_save_dir (str): Directory to save the model
        latent_dim (int): Dimension of the latent space
        generator_layers (list): Generator architecture
        discriminator_layers (list): Discriminator architecture
        batch_size (int): Batch size
        n_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        
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
    
    # Preprocess data
    preprocessor = TabularPreprocessor()
    processed_data = preprocessor.fit_transform(data)
    
    print(f"Training on {device}")
    print(f"Data shape: {processed_data.shape}")
    
    # Create dataset and dataloader
    dataset = TabularDataset(processed_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    input_dim = processed_data.shape[1]
    generator = TabularGenerator(latent_dim, input_dim, generator_layers).to(device)
    discriminator = TabularDiscriminator(input_dim, discriminator_layers).to(device)
    
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # Training setup
    real_label = 1.0
    fake_label = 0.0
    
    # Setup real-time visualization
    if use_streamlit:
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create placeholders for metrics and plots
        metrics_cols = st.columns(3)
        plot_placeholder = st.empty()
        
        # Create metric placeholders
        with metrics_cols[0]:
            g_loss_metric = st.empty()
        with metrics_cols[1]:
            d_loss_metric = st.empty()
        with metrics_cols[2]:
            epoch_metric = st.empty()
    
    # Training loop
    start_time = time.time()
    g_losses = []
    d_losses = []
    
    for epoch in range(n_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        for i, real_data in enumerate(dataloader):
            batch_size_current = real_data.size(0)
            real_data = real_data.to(device)
            
            # Labels
            real_labels = torch.full((batch_size_current,), real_label, dtype=torch.float, device=device)
            fake_labels = torch.full((batch_size_current,), fake_label, dtype=torch.float, device=device)
            
            # ===============================
            # Train Discriminator
            # ===============================
            discriminator.zero_grad()
            
            # Real data loss
            output = discriminator(real_data).view(-1)
            real_loss = adversarial_loss(output, real_labels)
            real_loss.backward()
            
            # Fake data loss
            noise = torch.randn(batch_size_current, latent_dim, device=device)
            fake_data = generator(noise)
            output = discriminator(fake_data.detach()).view(-1)
            fake_loss = adversarial_loss(output, fake_labels)
            fake_loss.backward()
            
            # Update discriminator
            d_loss = real_loss + fake_loss
            optimizer_D.step()
            
            # ===============================
            # Train Generator
            # ===============================
            generator.zero_grad()
            
            # Generate fake data and get discriminator output
            output = discriminator(fake_data).view(-1)
            g_loss = adversarial_loss(output, real_labels)  # Generator wants to fool discriminator
            g_loss.backward()
            optimizer_G.step()
            
            # Update metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
        
        # Calculate average losses
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Update real-time visualization
        if use_streamlit:
            # Update progress bar
            progress = (epoch + 1) / n_epochs
            progress_bar.progress(progress)
            
            # Update status text
            status_text.text(f'Epoch {epoch+1}/{n_epochs} - G Loss: {avg_g_loss:.4f} - D Loss: {avg_d_loss:.4f}')
            
            # Update metrics every 10 epochs or on the last epoch
            if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                with g_loss_metric.container():
                    st.metric("Generator Loss", f"{avg_g_loss:.4f}")
                with d_loss_metric.container():
                    st.metric("Discriminator Loss", f"{avg_d_loss:.4f}")
                with epoch_metric.container():
                    st.metric("Epoch", f"{epoch+1}/{n_epochs}")
                
                # Update plots
                with plot_placeholder.container():
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=g_losses,
                        mode='lines',
                        name='Generator Loss',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        y=d_losses,
                        mode='lines',
                        name='Discriminator Loss',
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title='Training Progress',
                        xaxis_title='Epoch',
                        yaxis_title='Loss',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"loss_plot_{epoch}")
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
    
    # Final update for Streamlit
    if use_streamlit:
        progress_bar.progress(1.0)
        status_text.text(f'✅ Training completed! Final G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}')
        st.balloons()
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Save models and preprocessor
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(model_save_dir)
    
    generator_path = os.path.join(model_save_dir, f"generator_{timestamp}.pth")
    discriminator_path = os.path.join(model_save_dir, f"discriminator_{timestamp}.pth")
    preprocessor_path = os.path.join(model_save_dir, f"preprocessor_{timestamp}.pkl")
    
    # Save generator with architecture info
    generator_save_dict = {
        'model_state_dict': generator.state_dict(),
        'latent_dim': latent_dim,
        'output_dim': input_dim,
        'layers': generator_layers
    }
    torch.save(generator_save_dict, generator_path)
    
    # Save discriminator with architecture info
    discriminator_save_dict = {
        'model_state_dict': discriminator.state_dict(),
        'input_dim': input_dim,
        'layers': discriminator_layers
    }
    torch.save(discriminator_save_dict, discriminator_path)
    
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Create training plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss', color='blue')
    plt.plot(d_losses, label='Discriminator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(g_losses) + 1), g_losses, 'b-', label='Generator')
    plt.plot(range(1, len(d_losses) + 1), d_losses, 'r-', label='Discriminator')
    plt.fill_between(range(1, len(g_losses) + 1), g_losses, alpha=0.3, color='blue')
    plt.fill_between(range(1, len(d_losses) + 1), d_losses, alpha=0.3, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(model_save_dir, f"training_plot_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generator saved to: {generator_path}")
    print(f"Discriminator saved to: {discriminator_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Training plot saved to: {plot_path}")
    
    return {
        'generator_path': generator_path,
        'discriminator_path': discriminator_path,
        'preprocessor_path': preprocessor_path,
        'plot_path': plot_path,
        'final_g_loss': avg_g_loss,
        'final_d_loss': avg_d_loss,
        'training_time': format_time(training_time),
        'g_losses': g_losses,
        'd_losses': d_losses
    }

def generate_tabular_data(generator_path, preprocessor_path, n_samples=100):
    """
    Generate tabular data using a trained generator.
    
    Args:
        generator_path (str): Path to the saved generator model
        preprocessor_path (str): Path to the saved preprocessor
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Generated data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load preprocessor
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load generator checkpoint
    checkpoint = torch.load(generator_path, map_location=device)
    
    # Get architecture parameters from checkpoint
    latent_dim = checkpoint.get('latent_dim', 100)
    output_dim = checkpoint.get('output_dim', len(preprocessor.columns))
    layers = checkpoint.get('layers', [128, 256, 128])
    
    # Create generator with correct architecture
    generator = TabularGenerator(latent_dim, output_dim, layers)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.to(device)
    generator.eval()
    
    # Generate data
    with torch.no_grad():
        noise = torch.randn(n_samples, latent_dim, device=device)
        generated_data = generator(noise)
        generated_data = generated_data.cpu().numpy()
    
    # Inverse transform to original format
    generated_df = preprocessor.inverse_transform(generated_data)
    
    return generated_df

def load_tabular_gan(generator_path, preprocessor_path):
    """
    Load a trained tabular GAN model.
    
    Args:
        generator_path (str): Path to the saved generator model
        preprocessor_path (str): Path to the saved preprocessor
        
    Returns:
        tuple: (generator, preprocessor)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load preprocessor
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load generator checkpoint
    checkpoint = torch.load(generator_path, map_location=device)
    
    # Get architecture parameters from checkpoint
    latent_dim = checkpoint.get('latent_dim', 100)
    output_dim = checkpoint.get('output_dim', len(preprocessor.columns))
    layers = checkpoint.get('layers', [128, 256, 128])
    
    # Create generator with correct architecture
    generator = TabularGenerator(latent_dim, output_dim, layers)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.to(device)
    generator.eval()
    
    return generator, preprocessor
