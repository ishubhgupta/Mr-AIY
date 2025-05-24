"""
CNN Page for AI Vision Suite
Handles image classification using Convolutional Neural Networks
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import DatabaseManager
from code import CNNTrainer, CNNPredictor, load_image

def show_cnn_page():
    """Display the CNN page with training and prediction functionality"""
    
    st.title("🖼️ Convolutional Neural Networks (CNN)")
    st.markdown("---")
    
    # Initialize database
    db = DatabaseManager()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Train Model", "🔍 Make Predictions", "📊 Model Performance", "📈 Analytics"])
    
    with tab1:
        st.header("Train CNN Model")
        
        # Model configuration
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Model Configuration")
            model_name = st.text_input("Model Name", value="CNN_ImageClassifier")
            architecture = st.selectbox("Architecture", ["Custom CNN", "ResNet18", "ResNet50", "VGG16"])
            num_classes = st.number_input("Number of Classes", min_value=2, max_value=1000, value=10)
            image_size = st.selectbox("Input Image Size", [32, 64, 128, 224], index=3)
            
        with col2:
            st.subheader("Training Parameters")
            batch_size = st.number_input("Batch Size", min_value=1, max_value=128, value=32)
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
            epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
            device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
        
        # Dataset upload
        st.subheader("Upload Training Dataset")
        uploaded_files = st.file_uploader(
            "Upload images (organized in folders by class)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Training CNN model..."):
                    try:
                        # Initialize trainer
                        trainer = CNNTrainer(
                            model_name=model_name,
                            architecture=architecture,
                            num_classes=num_classes,
                            image_size=image_size
                        )
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Create training callback
                        def training_callback(epoch, total_epochs, loss, accuracy):
                            progress = (epoch + 1) / total_epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch+1}/{total_epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                        
                        # Train model
                        model, history = trainer.train(
                            uploaded_files,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            device=device,
                            callback=training_callback
                        )
                        
                        # Save model to database
                        model_id = db.save_model(
                            name=model_name,
                            type="CNN",
                            file_path=f"models/{model_name}.pth",
                            accuracy=history['accuracy'][-1],
                            parameters={
                                'architecture': architecture,
                                'num_classes': num_classes,
                                'image_size': image_size,
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'epochs': epochs
                            }
                        )
                        
                        st.success(f"✅ Model '{model_name}' trained successfully!")
                        st.info(f"Model ID: {model_id}")
                        
                        # Display training history
                        st.subheader("Training History")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_loss = px.line(
                                x=range(1, len(history['loss']) + 1),
                                y=history['loss'],
                                title="Training Loss",
                                labels={'x': 'Epoch', 'y': 'Loss'}
                            )
                            st.plotly_chart(fig_loss, use_container_width=True)
                        
                        with col2:
                            fig_acc = px.line(
                                x=range(1, len(history['accuracy']) + 1),
                                y=history['accuracy'],
                                title="Training Accuracy",
                                labels={'x': 'Epoch', 'y': 'Accuracy'}
                            )
                            st.plotly_chart(fig_acc, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
            else:
                st.warning("⚠️ Please upload training images first!")
    
    with tab2:
        st.header("Make Predictions")
        
        # Model selection
        models = db.get_models_by_type("CNN")
        if models.empty:
            st.warning("⚠️ No trained CNN models found. Please train a model first.")
            return
        
        model_options = [f"{model[1]} (ID: {model[0]})" for model in models]
        selected_model = st.selectbox("Select Model", model_options)
        model_id = int(selected_model.split("ID: ")[1].split(")")[0])
        
        # Image upload for prediction
        uploaded_image = st.file_uploader(
            "Upload image for classification",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if st.button("🔍 Classify Image", type="primary"):
                    with st.spinner("Classifying image..."):
                        try:
                            # Load model details
                            model_data = db.get_model(model_id)
                            
                            # Initialize predictor
                            predictor = CNNPredictor(model_data[2])  # file_path
                            
                            # Make prediction
                            prediction, confidence, probabilities = predictor.predict(image)
                            
                            # Save prediction to database
                            db.save_prediction(
                                model_id=model_id,
                                input_data={"image_name": uploaded_image.name},
                                prediction=prediction,
                                confidence=float(confidence)
                            )
                            
                            # Display results
                            st.success(f"**Prediction:** {prediction}")
                            st.info(f"**Confidence:** {confidence:.2%}")
                            
                            # Show probability distribution
                            if probabilities is not None:
                                st.subheader("Class Probabilities")
                                prob_df = pd.DataFrame({
                                    'Class': [f'Class {i}' for i in range(len(probabilities))],
                                    'Probability': probabilities
                                })
                                fig_prob = px.bar(
                                    prob_df,
                                    x='Class',
                                    y='Probability',
                                    title="Classification Probabilities"
                                )
                                st.plotly_chart(fig_prob, use_container_width=True)
                        
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
    
    with tab3:
        st.header("Model Performance")
        
        # Model selection for performance analysis
        if not models.empty:
            model_options = [f"{model[1]} (ID: {model[0]})" for model in models]
            selected_model = st.selectbox("Select Model for Analysis", model_options, key="perf_model")
            model_id = int(selected_model.split("ID: ")[1].split(")")[0])
            
            # Get model details
            model_data = db.get_model(model_id)
            
            if model_data:
                st.subheader("Model Information")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Name", model_data[1])
                with col2:
                    st.metric("Accuracy", f"{model_data[4]:.2%}" if model_data[4] else "N/A")
                with col3:
                    st.metric("Created", model_data[6].split()[0])
                
                # Get predictions for this model
                predictions = db.get_predictions_by_model(model_id)
                
                if predictions:
                    st.subheader("Prediction History")
                    
                    # Create predictions dataframe
                    pred_df = pd.DataFrame(predictions, columns=[
                        'ID', 'Model ID', 'Input Data', 'Prediction', 'Confidence', 'Timestamp'
                    ])
                    
                    # Display predictions table
                    st.dataframe(pred_df[['Prediction', 'Confidence', 'Timestamp']], use_container_width=True)
                    
                    # Confidence distribution
                    st.subheader("Confidence Distribution")
                    fig_conf = px.histogram(
                        pred_df,
                        x='Confidence',
                        title="Prediction Confidence Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                else:
                    st.info("📝 No predictions made with this model yet.")
        else:
            st.warning("⚠️ No CNN models found.")
    
    with tab4:
        st.header("Analytics Dashboard")
        
        # Overall statistics
        total_models = len(db.get_models_by_type("CNN"))
        total_predictions = len(db.get_all_predictions())
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total CNN Models", total_models)
        with col2:
            st.metric("Total Predictions", total_predictions)
        with col3:
            avg_accuracy = np.mean([model[4] for model in models if model[4]]) if models else 0
            st.metric("Average Accuracy", f"{avg_accuracy:.2%}")
        with col4:
            recent_models = len([m for m in models if m[6] >= datetime.now().strftime('%Y-%m-%d')])
            st.metric("Models Today", recent_models)
        
        if not models.empty and total_predictions > 0:
            # Model comparison
            st.subheader("Model Comparison")
            
            model_stats = []
            for model in models:
                model_predictions = db.get_predictions_by_model(model[0])
                avg_confidence = np.mean([pred[4] for pred in model_predictions]) if model_predictions else 0
                
                model_stats.append({
                    'Model Name': model[1],
                    'Accuracy': model[4] or 0,
                    'Predictions Count': len(model_predictions),
                    'Average Confidence': avg_confidence,
                    'Created': model[6]
                })
            
            stats_df = pd.DataFrame(model_stats)
            
            if not stats_df.empty:
                # Accuracy vs Predictions scatter plot
                fig_scatter = px.scatter(
                    stats_df,
                    x='Predictions Count',
                    y='Accuracy',
                    size='Average Confidence',
                    hover_name='Model Name',
                    title="Model Performance Overview"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Model performance table
                st.subheader("Detailed Model Statistics")
                st.dataframe(stats_df, use_container_width=True)

if __name__ == "__main__":
    show_cnn_page()
