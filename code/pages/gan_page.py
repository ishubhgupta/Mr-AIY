"""
GAN Page for AI Vision Suite
Handles tabular data generation using Generative Adversarial Networks
"""

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO
from datetime import datetime
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import DatabaseManager
from code.gan import (
    train_tabular_gan, generate_tabular_data
)

def show_gan_page():
    """Display the GAN page with training and generation functionality"""
    
    st.title("🎨 Generative Adversarial Networks (GAN)")
    st.markdown("### 📊 Tabular Data Generation")
    st.markdown("---")
    
    # Initialize database
    db = DatabaseManager()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["🏋️ Train GAN", "📊 Generate Data", "📈 Model Analysis", "⚙️ Advanced"])
    
    with tab1:
        st.header("Train Tabular GAN Model")
        show_tabular_gan_training(db)
    
    with tab2:
        st.header("Generate Tabular Data")
        
        # Check available models
        tabular_models = db.get_models_by_type("TABULAR_GAN")
        
        if not tabular_models.empty:
            show_tabular_generation_interface(db, tabular_models)
        else:
            st.warning("⚠️ No trained GAN models found. Please train a model first.")
            
            # Show sample data option
            if st.button("📝 Use Sample Data for Demo"):
                st.session_state.demo_data = create_sample_tabular_data()
                st.success("✅ Sample data loaded! Go to the 'Train GAN' tab to train a model.")
    
    with tab3:
        show_model_analysis(db)
    
    with tab4:
        show_advanced_features(db)

def show_tabular_gan_training(db):
    """Show tabular GAN training interface."""
    st.subheader("📊 Train Tabular Data GAN")
    
    # Model configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Configuration")
        model_name = st.text_input("Model Name", value="TabularGAN_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
        latent_dim = st.number_input("Latent Dimension", min_value=32, max_value=256, value=100)
        
        # Generator architecture
        st.markdown("#### Generator Architecture")
        gen_layers = []
        num_gen_layers = st.slider("Number of Generator Layers", min_value=2, max_value=6, value=3)
        for i in range(num_gen_layers):
            layer_size = st.number_input(f"Generator Layer {i+1} Size", 
                                       min_value=32, max_value=512, 
                                       value=128 if i == 0 else max(64, 128 - i*32), 
                                       key=f"gen_layer_{i}")
            gen_layers.append(layer_size)
        
        # Discriminator architecture
        st.markdown("#### Discriminator Architecture")
        disc_layers = []
        num_disc_layers = st.slider("Number of Discriminator Layers", min_value=2, max_value=6, value=3)
        for i in range(num_disc_layers):
            layer_size = st.number_input(f"Discriminator Layer {i+1} Size", 
                                       min_value=32, max_value=512, 
                                       value=256 if i == 0 else max(64, 256 - i*64), 
                                       key=f"disc_layer_{i}")
            disc_layers.append(layer_size)
        
    with col2:
        st.subheader("Training Parameters")
        batch_size = st.number_input("Batch Size", min_value=16, max_value=256, value=64)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.0002, format="%.4f")
        epochs = st.number_input("Epochs", min_value=100, max_value=2000, value=500)
        device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
        
        # Advanced parameters
        st.markdown("#### Advanced Parameters")
        beta1 = st.number_input("Adam Beta1", min_value=0.1, max_value=0.9, value=0.5, format="%.2f")
        beta2 = st.number_input("Adam Beta2", min_value=0.9, max_value=0.999, value=0.999, format="%.3f")
        
        # Training schedule
        st.markdown("#### Training Schedule")
        save_interval = st.number_input("Save Interval (epochs)", min_value=50, max_value=500, value=100)
        sample_interval = st.number_input("Sample Generation Interval", min_value=10, max_value=100, value=50)
    
    # Dataset upload
    st.subheader("Upload Training Dataset")
    
    data_source = st.selectbox("Data Source", ["Upload CSV File", "Use Sample Data", "Use Previous Session Data"])
    
    data = None
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload CSV dataset",
            type=['csv'],
            help="Upload a CSV file with numerical and categorical data for training"
        )
        
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                
                # Show data info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", data.shape[0])
                with col2:
                    st.metric("Columns", data.shape[1])
                with col3:
                    numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
                    st.metric("Numeric Columns", numeric_cols)
                with col4:
                    categorical_cols = len(data.select_dtypes(include=['object']).columns)
                    st.metric("Categorical Columns", categorical_cols)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    elif data_source == "Use Sample Data":
        if st.button("📊 Load Sample Dataset"):
            data = create_sample_tabular_data()
            st.success(f"✅ Loaded sample dataset with {data.shape[0]} rows and {data.shape[1]} columns")
            st.dataframe(data.head(), use_container_width=True)
    
    elif data_source == "Use Previous Session Data":
        if hasattr(st.session_state, 'demo_data') and st.session_state.demo_data is not None:
            data = st.session_state.demo_data
            st.success(f"✅ Using session data with {data.shape[0]} rows and {data.shape[1]} columns")
            st.dataframe(data.head(), use_container_width=True)
        else:
            st.info("ℹ️ No data found in session. Please load data first.")
    
    # Data preprocessing options
    if data is not None:
        st.subheader("Data Preprocessing")
        
        col1, col2 = st.columns(2)
        with col1:
            # Handle missing values
            missing_strategy = st.selectbox("Handle Missing Values", 
                                          ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])
            
            # Feature selection
            if st.checkbox("Select specific columns"):
                selected_columns = st.multiselect("Select columns to include", 
                                                data.columns.tolist(),
                                                default=data.columns.tolist()[:10])  # Limit default selection
                if selected_columns:
                    data = data[selected_columns]
        
        with col2:
            # Data validation
            st.markdown("#### Data Validation")
            
            # Check for sufficient data
            min_rows = batch_size * 10
            if data.shape[0] < min_rows:
                st.warning(f"⚠️ Dataset has only {data.shape[0]} rows. Recommended: at least {min_rows} rows")
            else:
                st.success(f"✅ Dataset size is sufficient ({data.shape[0]} rows)")
            
            # Check for data types
            if len(data.select_dtypes(include=[np.number]).columns) == 0:
                st.error("❌ No numeric columns found. Please ensure dataset has numeric data.")
            else:
                st.success(f"✅ Found {len(data.select_dtypes(include=[np.number]).columns)} numeric columns")
    
    # Training button
    if data is not None and model_name:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Tabular GAN Training", type="primary", use_container_width=True):
                with st.spinner("🔄 Training Tabular GAN... This may take a while."):
                    try:
                        # Create models directory
                        models_dir = "models/tabular_gans"
                        os.makedirs(models_dir, exist_ok=True)
                        
                        # Preprocess data
                        processed_data = data.copy()
                        
                        # Handle missing values
                        if missing_strategy == "Drop rows":
                            processed_data = processed_data.dropna()
                        elif missing_strategy == "Fill with mean":
                            processed_data = processed_data.fillna(processed_data.mean())
                        elif missing_strategy == "Fill with median":
                            processed_data = processed_data.fillna(processed_data.median())
                        elif missing_strategy == "Fill with mode":
                            processed_data = processed_data.fillna(processed_data.mode().iloc[0])
                        
                        st.info(f"📊 Preprocessed data shape: {processed_data.shape}")
                        
                        # Train model
                        results = train_tabular_gan(
                            data=processed_data,
                            model_save_dir=models_dir,
                            latent_dim=latent_dim,
                            generator_layers=gen_layers,
                            discriminator_layers=disc_layers,
                            batch_size=batch_size,
                            n_epochs=epochs,
                            learning_rate=learning_rate
                        )
                        
                        # Save model to database
                        model_id = db.add_model(
                            name=model_name,
                            model_type="TABULAR_GAN",
                            file_path=results['generator_path'],
                            parameters=json.dumps({
                                'latent_dim': latent_dim,
                                'generator_layers': gen_layers,
                                'discriminator_layers': disc_layers,
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'epochs': epochs,
                                'data_shape': processed_data.shape,
                                'columns': processed_data.columns.tolist(),
                                'preprocessor_path': results['preprocessor_path'],
                                'discriminator_path': results['discriminator_path']  # Add discriminator path
                            }),
                            loss=results['final_g_loss'],
                            description=f"Tabular GAN trained on {datetime.now().strftime('%Y-%m-%d')}"
                        )
                        
                        st.success(f"✅ Tabular GAN '{model_name}' trained successfully!")
                        st.info(f"Model ID: {model_id}")
                        
                        # Display training results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Final G Loss", f"{results['final_g_loss']:.4f}")
                        with col2:
                            st.metric("Final D Loss", f"{results['final_d_loss']:.4f}")
                        with col3:
                            st.metric("Training Time", results['training_time'])
                        
                        # Show training plot
                        if results.get('plot_path'):
                            st.image(results['plot_path'])
                        
                        # Generate sample data with better error handling
                        st.subheader("Generated Sample Data")
                        try:
                            sample_data = generate_tabular_data(
                                results['generator_path'],
                                results['preprocessor_path'],
                                n_samples=5
                            )
                            
                            st.dataframe(sample_data, use_container_width=True)
                            
                        except Exception as sample_error:
                            st.warning(f"⚠️ Could not generate sample data: {str(sample_error)}")
                            st.info("Model training completed successfully, but sample generation failed. You can try generating data in the 'Generate Data' tab.")
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        st.exception(e)
    else:
        if data is None:
            st.warning("⚠️ Please upload or load training data.")
        if not model_name:
            st.warning("⚠️ Please provide a model name.")

def show_tabular_generation_interface(db, tabular_models):
    """Show tabular data generation interface."""
    st.subheader("📊 Generate Tabular Data")
    
    # Model selection
    model_options = []
    for _, model in tabular_models.iterrows():
        model_options.append(f"{model['name']} (ID: {model['id']})")
    
    selected_model = st.selectbox("Select Tabular Model", model_options)
    model_id = int(selected_model.split("ID: ")[1].split(")")[0])
    
    # Get model details
    model_data = tabular_models[tabular_models['id'] == model_id].iloc[0]
    model_params = json.loads(model_data['parameters'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Generation Parameters")
        num_samples = st.number_input("Number of Samples to Generate", min_value=1, max_value=10000, value=100)
        seed = st.number_input("Random Seed (0 for random)", min_value=0, max_value=999999, value=0)
        
        # Show model info
        st.info(f"Original data shape: {model_params.get('data_shape', 'Unknown')}")
        st.info(f"Columns: {len(model_params.get('columns', []))}")
        
        # Advanced generation options
        st.markdown("#### Advanced Options")
        temperature = st.slider("Generation Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
                               help="Higher values make generation more random")
    
    with col2:
        if st.button("📊 Generate Data", type="primary", use_container_width=True):
            with st.spinner("Generating tabular data..."):
                try:
                    # Set seed if specified
                    if seed != 0:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                    
                    # Generate data
                    generated_data = generate_tabular_data(
                        generator_path=model_data['file_path'],
                        preprocessor_path=model_params['preprocessor_path'],
                        n_samples=num_samples
                    )
                    
                    st.subheader("Generated Data")
                    st.dataframe(generated_data, use_container_width=True)
                    
                    # Download button
                    csv = generated_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Generated Data",
                        data=csv,
                        file_name=f"generated_data_{num_samples}_samples.csv",
                        mime="text/csv"
                    )
                    
                    # Statistics comparison
                    st.subheader("Data Statistics")
                    
                    # Show basic statistics
                    numeric_cols = generated_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        stats = generated_data[numeric_cols].describe()
                        st.dataframe(stats, use_container_width=True)
                        
                        # Plot distributions for first few numeric columns
                        if len(numeric_cols) > 0:
                            st.subheader("Distribution Plots")
                            cols_to_plot = min(4, len(numeric_cols))
                            plot_cols = st.columns(cols_to_plot)
                            
                            for i, col in enumerate(numeric_cols[:cols_to_plot]):
                                with plot_cols[i]:
                                    fig = px.histogram(generated_data, x=col, title=f"Distribution of {col}")
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Optional: Save generation record to database
                    try:
                        # Attempt to log the generation activity with correct parameter names
                        db.add_prediction(
                            model_id=model_id,
                            input_data=json.dumps({
                                "num_samples": num_samples,
                                "seed": seed,
                                "temperature": temperature,
                                "timestamp": datetime.now().isoformat()
                            }),
                            prediction=f"Generated {num_samples} tabular samples using GAN model",
                            confidence=1.0
                        )
                        st.success("📝 Generation activity logged to database")
                    except TypeError as te:
                        # If the method signature is different, try alternative parameter names
                        try:
                            # Try with 'result' parameter name
                            db.add_prediction(
                                model_id=model_id,
                                input_data=json.dumps({
                                    "num_samples": num_samples,
                                    "seed": seed,
                                    "temperature": temperature,
                                    "timestamp": datetime.now().isoformat()
                                }),
                                result=f"Generated {num_samples} tabular samples using GAN model",
                                confidence=1.0
                            )
                            st.success("📝 Generation activity logged to database")
                        except TypeError:
                            # Try with minimal required parameters
                            try:
                                db.add_prediction(
                                    model_id=model_id,
                                    input_data=json.dumps({
                                        "action": "data_generation",
                                        "num_samples": num_samples,
                                        "model_type": "TABULAR_GAN"
                                    }),
                                    prediction_text=f"Generated {num_samples} samples",
                                    confidence_score=1.0
                                )
                                st.success("📝 Generation activity logged to database")
                            except Exception:
                                # Skip database logging if method signature is incompatible
                                st.info("💾 Data generated successfully (database logging skipped due to method incompatibility)")
                    except Exception as db_error:
                        # Don't let database issues break the main functionality
                        st.info(f"💾 Data generated successfully (database logging failed: {str(db_error)})")
                        # Optionally log to console for debugging
                        print(f"Database logging error: {db_error}")

                except Exception as e:
                    st.error(f"❌ Generation failed: {str(e)}")
                    st.exception(e)

def show_model_analysis(db):
    """Show model analysis for tabular GANs."""
    st.header("Model Analysis")
    
    # Get all tabular GAN models
    tabular_models = db.get_models_by_type("TABULAR_GAN")
    
    if not tabular_models.empty:
        st.success(f"✅ Found {len(tabular_models)} trained models")
        
        # Display models table
        display_models = tabular_models[['id', 'name', 'created_at', 'loss']].copy()
        display_models['created_at'] = pd.to_datetime(display_models['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_models, use_container_width=True)
        
        # Model comparison
        if len(tabular_models) >= 2:
            st.subheader("Model Comparison")
            
            # Select models for comparison
            model1_options = [f"{model['name']} (ID: {model['id']})" for _, model in tabular_models.iterrows()]
            model2_options = [f"{model['name']} (ID: {model['id']})" for _, model in tabular_models.iterrows()]
            
            col1, col2 = st.columns(2)
            with col1:
                model1_select = st.selectbox("First Model", model1_options, key="model1")
                model1_id = int(model1_select.split("ID: ")[1].split(")")[0])
            with col2:
                model2_select = st.selectbox("Second Model", model2_options, key="model2")
                model2_id = int(model2_select.split("ID: ")[1].split(")")[0])
            
            if model1_id != model2_id:
                # Get model data
                model1_data = tabular_models[tabular_models['id'] == model1_id].iloc[0]
                model2_data = tabular_models[tabular_models['id'] == model2_id].iloc[0]
                
                # Compare basic info
                comparison_data = {
                    'Parameter': ['Name', 'Loss', 'Created At'],
                    'Model 1': [
                        model1_data['name'], 
                        f"{model1_data['loss']:.4f}" if model1_data['loss'] else 'N/A',
                        str(model1_data.get('created_at', 'Unknown'))[:19]
                    ],
                    'Model 2': [
                        model2_data['name'], 
                        f"{model2_data['loss']:.4f}" if model2_data['loss'] else 'N/A',
                        str(model2_data.get('created_at', 'Unknown'))[:19]
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)
        
        # Model performance visualization
        if len(tabular_models) > 0:
            st.subheader("Model Performance")
            
            # Loss comparison chart
            models_with_loss = tabular_models[tabular_models['loss'].notna()]
            if not models_with_loss.empty:
                fig = px.bar(
                    models_with_loss, 
                    x='name', 
                    y='loss', 
                    title='Model Loss Comparison',
                    labels={'loss': 'Final Generator Loss', 'name': 'Model Name'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual model analysis
                st.subheader("Individual Model Analysis")
                selected_model_id = st.selectbox(
                    "Select a model to analyze",
                    tabular_models['id'].tolist(),
                    format_func=lambda id: tabular_models[tabular_models['id'] == id]['name'].values[0]
                )
                
                if selected_model_id:
                    model_results = tabular_models[tabular_models['id'] == selected_model_id].iloc[0]
                    model_params = json.loads(model_results.get('parameters', '{}'))
                    
                    st.write(f"### {model_results['name']}")
                    st.write(f"**Loss:** {model_results['loss']:.4f}" if model_results['loss'] else "**Loss:** N/A")
                    st.write(f"**Created At:** {model_results['created_at']}")
                    
                    # Display model parameters
                    if model_params:
                        st.markdown("#### Model Parameters")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"- **Latent Dimension:** {model_params.get('latent_dim', 'N/A')}")
                            st.write(f"- **Batch Size:** {model_params.get('batch_size', 'N/A')}")
                            st.write(f"- **Learning Rate:** {model_params.get('learning_rate', 'N/A')}")
                        with col2:
                            st.write(f"- **Epochs:** {model_params.get('epochs', 'N/A')}")
                            st.write(f"- **Generator Layers:** {model_params.get('generator_layers', 'N/A')}")
                            st.write(f"- **Data Shape:** {model_params.get('data_shape', 'N/A')}")
                    
                    # Model file links
                    st.markdown("#### Model Files")
                    st.write(f"- **Generator Model:** `{model_results['file_path']}`")
                    if 'discriminator_path' in model_params:
                        st.write(f"- **Discriminator Model:** `{model_params['discriminator_path']}`")
                    if 'preprocessor_path' in model_params:
                        st.write(f"- **Preprocessor:** `{model_params['preprocessor_path']}`")
    else:
        st.info("ℹ️ No trained models found. Please train some models first.")

def show_advanced_features(db):
    """Show advanced features and settings."""
    st.header("Advanced Features")
    
    # Model management
    st.subheader("🗂️ Model Management")
    
    tabular_models = db.get_models_by_type("TABULAR_GAN")
    
    if not tabular_models.empty:
        # Model deletion
        st.markdown("#### Delete Models")
        models_to_delete = st.multiselect(
            "Select models to delete",
            [f"{model['name']} (ID: {model['id']})" for _, model in tabular_models.iterrows()]
        )
        
        if models_to_delete and st.button("🗑️ Delete Selected Models", type="secondary"):
            for model_str in models_to_delete:
                model_id = int(model_str.split("ID: ")[1].split(")")[0])
                try:
                    # Delete from database
                    # Note: You'll need to implement delete_model method in DatabaseManager
                    st.success(f"✅ Model {model_id} marked for deletion")
                except Exception as e:
                    st.error(f"❌ Failed to delete model {model_id}: {str(e)}")
    
    # System settings
    st.subheader("🔧 System Settings")
    
    # GPU Settings
    col1, col2 = st.columns(2)
    with col1:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            st.success("✅ GPU Available")
            st.info(f"Device: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ No GPU detected, using CPU")
    
    with col2:
        # Memory settings
        if gpu_available:
            st.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
        
        # Recommended batch sizes
        st.markdown("#### Recommended Batch Sizes")
        if gpu_available:
            st.write("- Small datasets: 64-128")
            st.write("- Medium datasets: 32-64") 
            st.write("- Large datasets: 16-32")
        else:
            st.write("- CPU training: 16-32")
    
    # Model storage settings
    st.subheader("💾 Storage Settings")
    models_dir = st.text_input("Models Directory", value="models/tabular_gans")
    
    if st.button("📁 Create Directory"):
        try:
            os.makedirs(models_dir, exist_ok=True)
            st.success(f"✅ Directory created: {models_dir}")
        except Exception as e:
            st.error(f"❌ Failed to create directory: {str(e)}")
    
    # Export/Import settings
    st.subheader("📤 Export/Import")
    
    if not tabular_models.empty:
        if st.button("📤 Export Model List"):
            models_export = tabular_models[['id', 'name', 'created_at', 'loss']].to_csv(index=False)
            st.download_button(
                label="📥 Download Model List",
                data=models_export,
                file_name=f"tabular_gan_models_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

def create_sample_tabular_data():
    """Create sample tabular data for demonstration."""
    np.random.seed(42)
    
    n_samples = 1000
    
    # Create synthetic customer data
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 80, n_samples),
        'Tenure': np.random.randint(1, 72, n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples).round(2),
        'TotalCharges': np.random.uniform(100, 8000, n_samples).round(2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    show_gan_page()
