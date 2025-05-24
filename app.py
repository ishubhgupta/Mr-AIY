import streamlit as st
import sqlite3
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import our modules
from database import DatabaseManager
from code import (
    load_image,
    CNNTrainer, CNNPredictor,
    RCNNTrainer, RCNNPredictor,
    NeuralNetworkTrainer, NeuralNetworkPredictor,
    GANTrainer, GANGenerator,
    ClusteringTrainer, ClusteringPredictor
)

# Import all page modules
from code.pages.gan_page import show_gan_page
from code.pages.neural_networks_page import show_neural_networks_page

# Import new pages with error handling
try:
    from code.pages.data_management_page import show_data_management_page
    DATA_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing data management page: {e}")
    DATA_MANAGEMENT_AVAILABLE = False

try:
    from code.pages.model_comparison_page import show_model_comparison_page
    MODEL_COMPARISON_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing model comparison page: {e}")
    MODEL_COMPARISON_AVAILABLE = False

try:
    from code.pages.settings_page import show_settings_page
    SETTINGS_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing settings page: {e}")
    SETTINGS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AI Vision Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .module-card {
        background-color: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .module-card h3 {
        color: #3498db;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def init_database():
    db_manager = DatabaseManager()
    db_manager.create_tables()
    return db_manager

def get_storage_info():
    """Get storage information for the current directory"""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        return {
            'used': total_size / (1024 * 1024)  # Convert to MB
        }
    except Exception:
        return {'used': 0.0}

# Sidebar navigation
def sidebar_navigation():
    st.sidebar.markdown("# 🤖 AI Vision Suite")
    st.sidebar.markdown("---")
    
    pages = {
        "🏠 Home": "home",
        "📊 Dashboard": "dashboard",
        "🖼️ CNN Image Classification": "cnn",
        "🎯 RCNN Object Detection": "rcnn",
        "🧠 Neural Networks": "neural_networks",
        "🎨 GAN Image Generation": "gan",
        "🔍 Clustering Analysis": "clustering",
        "📁 Data Management": "data_management",
        "🏆 Model Comparison": "model_comparison",
        "⚙️ Settings": "settings"
    }
    
    selected_page = st.sidebar.selectbox(
        "Choose a module:",
        list(pages.keys()),
        index=0
    )
    
    return pages[selected_page]

# Home page
def show_home():
    st.markdown('<h1 class="main-header">🤖 AI Vision Suite</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **AI Vision Suite** - a comprehensive platform that integrates five powerful AI technologies:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <h3>🖼️ CNN Image Classification</h3>
            <p>Train and deploy convolutional neural networks for image classification tasks with pre-trained models and custom architectures.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <h3>🧠 Neural Networks</h3>
            <p>Build and train various neural network architectures including MLPs and LSTMs for tabular and sequence data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <h3>🎯 RCNN Object Detection</h3>
            <p>Implement state-of-the-art object detection using Faster R-CNN and Mask R-CNN for precise object localization.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="module-card">
            <h3>🎨 GAN Image Generation</h3>
            <p>Generate realistic images using Generative Adversarial Networks with both standard and conditional variants.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="module-card">
            <h3>🔍 Clustering Analysis</h3>
            <p>Discover patterns in your data using various unsupervised clustering algorithms including K-means, DBSCAN, and more.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    db_manager = init_database()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_models = db_manager.get_total_models()
        st.metric("Total Models", total_models)
    
    with col2:
        total_predictions = db_manager.get_total_predictions()
        st.metric("Total Predictions", total_predictions)
    
    with col3:
        total_datasets = db_manager.get_total_datasets()
        st.metric("Datasets", total_datasets)
    
    with col4:
        active_models = db_manager.get_active_models()
        st.metric("Active Models", active_models)

# Dashboard page
def show_dashboard():
    st.markdown("# 📊 Dashboard")
    
    db_manager = init_database()
    
    # Model performance overview
    st.subheader("Model Performance Overview")
    
    models_df = db_manager.get_models_summary()
    
    if not models_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Model type distribution
            model_counts = models_df['model_type'].value_counts()
            fig = px.pie(values=model_counts.values, names=model_counts.index, 
                        title="Models by Type")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training progress over time
            training_data = db_manager.get_training_history()
            if not training_data.empty:
                fig = px.line(training_data, x='created_at', y='accuracy', 
                             color='model_type', title="Training Accuracy Over Time")
                st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    
    recent_predictions = db_manager.get_recent_predictions(limit=10)
    if not recent_predictions.empty:
        st.dataframe(recent_predictions, use_container_width=True)
    else:
        st.info("No recent predictions found. Start using the models to see activity here!")
    
    # System metrics
    st.subheader("System Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Storage usage
        storage_info = get_storage_info()
        st.metric("Storage Used", f"{storage_info['used']:.2f} MB")
    
    with col2:
        # Model accuracy distribution
        if not models_df.empty and 'accuracy' in models_df.columns:
            avg_accuracy = models_df['accuracy'].mean()
            st.metric("Average Model Accuracy", f"{avg_accuracy:.2%}")
    
    with col3:
        # Active sessions
        st.metric("Active Session", "1")

# Main app logic
def main():
    # Initialize database
    db_manager = init_database()
    
    # Sidebar navigation
    current_page = sidebar_navigation()
    
    # Display selected page
    if current_page == "home":
        show_home()
    elif current_page == "dashboard":
        show_dashboard()    
    elif current_page == "cnn":
        from code.pages.cnn_page import show_cnn_page
        show_cnn_page()
    elif current_page == "rcnn":
        from code.pages.rcnn_page import show_rcnn_page
        show_rcnn_page()
    elif current_page == "neural_networks":
        from code.pages.neural_networks_page import show_neural_networks_page
        show_neural_networks_page()
    elif current_page == "gan":
        from code.pages.gan_page import show_gan_page
        show_gan_page()
    elif current_page == "clustering":
        from code.pages.clustering_page import show_clustering_page
        show_clustering_page()
    elif current_page == "data_management":
        if DATA_MANAGEMENT_AVAILABLE:
            show_data_management_page()
        else:
            st.error("❌ Data Management module is not available. Please check the installation.")
            st.info("📝 Make sure the data_management_page.py file exists in the code/pages/ directory.")
    elif current_page == "model_comparison":
        if MODEL_COMPARISON_AVAILABLE:
            show_model_comparison_page()
        else:
            st.error("❌ Model Comparison module is not available. Please check the installation.")
            st.info("📝 Make sure the model_comparison_page.py file exists in the code/pages/ directory.")
    elif current_page == "settings":
        if SETTINGS_AVAILABLE:
            show_settings_page()
        else:
            st.error("❌ Settings module is not available. Please check the installation.")
            st.info("📝 Make sure the settings_page.py file exists in the code/pages/ directory.")

if __name__ == "__main__":
    main()
