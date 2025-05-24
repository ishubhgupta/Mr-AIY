"""
RCNN module page for the Streamlit application.
Handles object detection using Faster R-CNN and Mask R-CNN.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
from datetime import datetime
import json
import io
import base64

def show_rcnn_page():
    """Main RCNN page function."""
    st.markdown("# 🎯 RCNN Object Detection")
    
    tabs = st.tabs(["Train Model", "Object Detection", "Model Management", "Detection Gallery"])
    
    with tabs[0]:
        show_rcnn_training()
    
    with tabs[1]:
        show_rcnn_detection()
    
    with tabs[2]:
        show_rcnn_management()
    
    with tabs[3]:
        show_detection_gallery()

def show_rcnn_training():
    """RCNN training interface."""
    st.subheader("Train RCNN Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Model Configuration")
        
        model_name = st.text_input(
            "Model Name", 
            value=f"rcnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        model_type = st.selectbox(
            "Model Type",
            ["Faster R-CNN", "Mask R-CNN", "RetinaNet"]
        )
        
        backbone = st.selectbox(
            "Backbone",
            ["ResNet50", "ResNet101", "MobileNetV3"]
        )
        
        num_classes = st.number_input(
            "Number of Classes (including background)", 
            min_value=2, max_value=100, value=91
        )
        
        epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=10)
        batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1)
        learning_rate = st.number_input(
            "Learning Rate", 
            min_value=0.00001, max_value=0.01, value=0.001, format="%.5f"
        )
        
        use_pretrained = st.checkbox("Use Pretrained COCO Weights", value=True)
    
    with col2:
        st.markdown("### Dataset Configuration")
        
        dataset_format = st.selectbox(
            "Annotation Format",
            ["COCO JSON", "Pascal VOC XML", "YOLO TXT"]
        )
        
        uploaded_images = st.file_uploader(
            "Upload Images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        uploaded_annotations = st.file_uploader(
            "Upload Annotations",
            type=['json', 'xml', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_images:
            st.success(f"Uploaded {len(uploaded_images)} images")
            
            # Show sample images
            if len(uploaded_images) > 0:
                st.markdown("### Sample Images")
                cols = st.columns(min(3, len(uploaded_images)))
                for i, uploaded_file in enumerate(uploaded_images[:3]):
                    with cols[i]:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=uploaded_file.name, use_column_width=True)
        
        # Class labels
        class_labels = st.text_area(
            "Class Labels (one per line)",
            value="person\ncar\nbicycle\nmotorcycle\nairplane\nbus\ntrain\ntruck\nboat",
            height=150
        )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Training", type="primary", use_container_width=True):
            if uploaded_images and model_name:
                with st.spinner("Training RCNN model... This may take a while."):
                    try:
                        # Show training simulation
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            status_text.text(f'Training Progress: {i+1}%')
                            
                        st.success("🎉 Training completed successfully!")
                        st.balloons()
                        
                        # Show training results
                        show_training_results()
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
            else:
                st.warning("Please provide model name and upload training data.")

def show_rcnn_detection():
    """Object detection interface."""
    st.subheader("Object Detection")
    
    # Model selection
    from database.db_setup import DatabaseManager
    db_manager = DatabaseManager()
    
    try:
        rcnn_models = db_manager.get_models_by_type('rcnn')
    except:
        rcnn_models = pd.DataFrame()  # Empty DataFrame if no models
    
    if not rcnn_models.empty:
        selected_model = st.selectbox(
            "Select Detection Model",
            rcnn_models['model_name'].tolist()
        )
        
        # Detection parameters
        col1, col2 = st.columns([1, 1])
        
        with col1:
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.1, max_value=1.0, value=0.5, step=0.05
            )
            
            nms_threshold = st.slider(
                "NMS Threshold", 
                min_value=0.1, max_value=1.0, value=0.5, step=0.05
            )
            
            max_detections = st.number_input(
                "Max Detections", 
                min_value=1, max_value=100, value=50
            )
        
        with col2:
            detection_mode = st.selectbox(
                "Detection Mode",
                ["Single Image", "Batch Processing", "Video Frame"]
            )
            
            show_labels = st.checkbox("Show Labels", value=True)
            show_confidence = st.checkbox("Show Confidence", value=True)
            show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        
        # Image upload
        if detection_mode == "Single Image":
            uploaded_image = st.file_uploader(
                "Upload Image for Detection",
                type=['png', 'jpg', 'jpeg']
            )
            
            if uploaded_image:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("### Original Image")
                    image = Image.open(uploaded_image)
                    st.image(image, use_column_width=True)
                
                with col2:
                    if st.button("Detect Objects", type="primary"):
                        with st.spinner("Detecting objects..."):
                            # Simulate detection
                            detection_results = simulate_object_detection(image)
                            
                            # Draw detections
                            detected_image = draw_detections(
                                image, detection_results, 
                                show_labels, show_confidence, show_boxes
                            )
                            
                            st.markdown("### Detection Results")
                            st.image(detected_image, use_column_width=True)
                            
                            # Show detection statistics
                            show_detection_statistics(detection_results)
        
        elif detection_mode == "Batch Processing":
            uploaded_images = st.file_uploader(
                "Upload Images for Batch Detection",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True
            )
            
            if uploaded_images and st.button("Process Batch", type="primary"):
                st.markdown("### Batch Processing Results")
                
                progress_bar = st.progress(0)
                results_container = st.container()
                
                for i, uploaded_file in enumerate(uploaded_images):
                    progress_bar.progress((i + 1) / len(uploaded_images))
                    
                    with results_container:
                        image = Image.open(uploaded_file)
                        detection_results = simulate_object_detection(image)
                        detected_image = draw_detections(image, detection_results)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.image(image, caption=f"Original: {uploaded_file.name}", 
                                   use_column_width=True)
                        with col2:
                            st.image(detected_image, caption=f"Detected: {uploaded_file.name}", 
                                   use_column_width=True)
    
    else:
        st.info("No RCNN models found. Please train a model first.")
        
        # Option to use demo model
        if st.button("Use Demo Model"):
            st.info("Demo model loaded! Upload an image to see object detection in action.")

def show_rcnn_management():
    """Model management interface."""
    st.subheader("RCNN Model Management")
    
    from database.db_setup import DatabaseManager
    db_manager = DatabaseManager()
    
    try:
        rcnn_models = db_manager.get_models_by_type('rcnn')
    except:
        rcnn_models = pd.DataFrame()
    
    if not rcnn_models.empty:
        st.markdown("### Available Models")
        st.dataframe(rcnn_models, use_container_width=True)
        
        # Model actions
        selected_model_id = st.selectbox(
            "Select Model for Actions",
            rcnn_models['id'].tolist(),
            format_func=lambda x: rcnn_models[rcnn_models['id'] == x]['model_name'].iloc[0]
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 View Details"):
                model_details = db_manager.get_model_details(selected_model_id)
                if model_details:
                    st.json(model_details)
        
        with col2:
            if st.button("📈 Performance"):
                show_model_performance(selected_model_id)
        
        with col3:
            if st.button("💾 Export"):
                st.info("Export functionality coming soon!")
        
        with col4:
            if st.button("🗑️ Delete", type="secondary"):
                if st.button("Confirm Delete", type="secondary"):
                    db_manager.delete_model(selected_model_id)
                    st.success("Model deleted!")
                    st.experimental_rerun()
    
    else:
        st.info("No RCNN models found.")
        
        # Quick start options
        st.markdown("### Quick Start")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📚 Load Pre-trained Model"):
                st.info("Pre-trained model loading coming soon!")
        
        with col2:
            if st.button("🎯 Start Training"):
                st.info("Redirecting to training tab...")

def show_detection_gallery():
    """Detection results gallery."""
    st.subheader("Detection Gallery")
    
    # Gallery filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_model = st.selectbox("Filter by Model", ["All Models", "Model 1", "Model 2"])
    
    with col2:
        filter_class = st.selectbox("Filter by Class", ["All Classes", "person", "car", "bicycle"])
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Date", "Confidence", "Model"])
    
    # Sample gallery
    st.markdown("### Recent Detections")
    
    # Create sample detection results
    sample_detections = create_sample_detections()
    
    for i, detection in enumerate(sample_detections):
        with st.expander(f"Detection {i+1} - {detection['model']} - {detection['date']}"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(detection['image'], caption="Detected Image", use_column_width=True)
            
            with col2:
                st.markdown("**Detection Results:**")
                for obj in detection['objects']:
                    st.write(f"- {obj['class']}: {obj['confidence']:.2%}")
                
                st.markdown("**Model Info:**")
                st.write(f"Model: {detection['model']}")
                st.write(f"Processing Time: {detection['processing_time']}")
                st.write(f"Date: {detection['date']}")

# Helper functions
def simulate_object_detection(image):
    """Simulate object detection results."""
    import random
    
    width, height = image.size
    
    # Common COCO classes
    classes = ['person', 'car', 'bicycle', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
    
    # Generate random detections
    num_detections = random.randint(2, 6)
    detections = []
    
    for _ in range(num_detections):
        x1 = random.randint(0, width // 2)
        y1 = random.randint(0, height // 2)
        x2 = random.randint(x1 + 50, width)
        y2 = random.randint(y1 + 50, height)
        
        detection = {
            'class': random.choice(classes),
            'confidence': random.uniform(0.5, 0.99),
            'bbox': [x1, y1, x2, y2]
        }
        detections.append(detection)
    
    return detections

def draw_detections(image, detections, show_labels=True, show_confidence=True, show_boxes=True):
    """Draw detection results on image."""
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    # Colors for different classes
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray']
    
    for i, detection in enumerate(detections):
        if show_boxes:
            color = colors[i % len(colors)]
            bbox = detection['bbox']
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=3)
            
            if show_labels or show_confidence:
                # Prepare label text
                label_parts = []
                if show_labels:
                    label_parts.append(detection['class'])
                if show_confidence:
                    label_parts.append(f"{detection['confidence']:.2%}")
                
                label = ' '.join(label_parts)
                
                # Draw label background
                try:
                    font = ImageFont.load_default()
                    bbox_font = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox_font[2] - bbox_font[0]
                    text_height = bbox_font[3] - bbox_font[1]
                except:
                    text_width, text_height = len(label) * 10, 15
                
                draw.rectangle([bbox[0], bbox[1] - text_height - 5, 
                              bbox[0] + text_width + 10, bbox[1]], 
                              fill=color)
                
                # Draw label text
                draw.text((bbox[0] + 5, bbox[1] - text_height - 2), label, fill='white')
    
    return image_copy

def show_detection_statistics(detections):
    """Show detection statistics."""
    st.markdown("### Detection Statistics")
    
    if detections:
        # Count by class
        class_counts = {}
        total_confidence = 0
        
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += detection['confidence']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Objects", len(detections))
            st.metric("Average Confidence", f"{total_confidence / len(detections):.2%}")
        
        with col2:
            st.metric("Unique Classes", len(class_counts))
            
            # Show class distribution
            for class_name, count in class_counts.items():
                st.write(f"**{class_name}**: {count}")
    else:
        st.info("No objects detected.")

def show_training_results():
    """Show training results visualization."""
    st.markdown("### Training Results")
    
    # Simulate training metrics
    import matplotlib.pyplot as plt
    import numpy as np
    
    epochs = list(range(1, 11))
    train_loss = [0.8, 0.6, 0.45, 0.35, 0.28, 0.22, 0.18, 0.15, 0.13, 0.11]
    val_loss = [0.85, 0.65, 0.5, 0.4, 0.33, 0.28, 0.25, 0.23, 0.21, 0.19]
    map_score = [0.3, 0.45, 0.58, 0.65, 0.71, 0.76, 0.79, 0.82, 0.84, 0.86]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        ax.plot(epochs, train_loss, label='Training Loss', color='blue')
        ax.plot(epochs, val_loss, label='Validation Loss', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        ax.plot(epochs, map_score, label='mAP Score', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP Score')
        ax.set_title('Mean Average Precision')
        ax.legend()
        st.pyplot(fig)

def show_model_performance(model_id):
    """Show detailed model performance metrics."""
    st.markdown("### Model Performance")
    
    # Simulate performance metrics
    metrics = {
        'mAP@0.5': 0.86,
        'mAP@0.75': 0.72,
        'mAP@0.5:0.95': 0.64,
        'Precision': 0.89,
        'Recall': 0.82,
        'F1-Score': 0.85
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("mAP@0.5", f"{metrics['mAP@0.5']:.2%}")
        st.metric("Precision", f"{metrics['Precision']:.2%}")
    
    with col2:
        st.metric("mAP@0.75", f"{metrics['mAP@0.75']:.2%}")
        st.metric("Recall", f"{metrics['Recall']:.2%}")
    
    with col3:
        st.metric("mAP@0.5:0.95", f"{metrics['mAP@0.5:0.95']:.2%}")
        st.metric("F1-Score", f"{metrics['F1-Score']:.2%}")

def create_sample_detections():
    """Create sample detection results for gallery."""
    return [
        {
            'model': 'Faster R-CNN ResNet50',
            'date': '2024-01-15 14:30',
            'processing_time': '0.24s',
            'image': 'https://via.placeholder.com/300x200?text=Sample+Detection+1',
            'objects': [
                {'class': 'person', 'confidence': 0.95},
                {'class': 'car', 'confidence': 0.87},
                {'class': 'bicycle', 'confidence': 0.72}
            ]
        },
        {
            'model': 'Mask R-CNN ResNet101',
            'date': '2024-01-15 13:15',
            'processing_time': '0.38s',
            'image': 'https://via.placeholder.com/300x200?text=Sample+Detection+2',
            'objects': [
                {'class': 'bus', 'confidence': 0.92},
                {'class': 'person', 'confidence': 0.88},
                {'class': 'person', 'confidence': 0.84}
            ]
        }
    ]
