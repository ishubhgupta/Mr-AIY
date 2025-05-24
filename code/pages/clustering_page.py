"""
Clustering Page for AI Vision Suite
Handles unsupervised learning using various clustering algorithms
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import sys
import json
from io import BytesIO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import DatabaseManager
from code import ClusteringTrainer, ClusteringPredictor

def show_clustering_page():
    """Display the clustering page with training and analysis functionality"""
    
    st.title("🔬 Unsupervised Clustering")
    st.markdown("---")
    
    # Initialize database
    db = DatabaseManager()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Train Clusters", "🔍 Analyze Data", "📈 Visualizations", "🎯 Predictions", "📋 Model Management"])
    
    with tab1:
        st.header("Train Clustering Models")
        
        # Data source selection
        data_source = st.radio(
            "Select Data Source",
            ["Upload CSV File", "Generate Synthetic Data", "Use Sample Dataset"]
        )
        
        data = None
        
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
                st.subheader("Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                st.write(f"Dataset shape: {data.shape}")
        
        elif data_source == "Generate Synthetic Data":
            st.subheader("Synthetic Data Parameters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_samples = st.number_input("Number of Samples", min_value=100, max_value=5000, value=1000)
                n_features = st.number_input("Number of Features", min_value=2, max_value=20, value=2)
            
            with col2:
                n_centers = st.number_input("Number of Centers", min_value=2, max_value=10, value=3)
                dataset_type = st.selectbox("Dataset Type", ["Blobs", "Circles", "Moons"])
            
            with col3:
                noise = st.slider("Noise Level", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
                random_state = st.number_input("Random State", min_value=0, max_value=999, value=42)
            
            if st.button("Generate Data"):
                if dataset_type == "Blobs":
                    X, y = make_blobs(
                        n_samples=n_samples,
                        centers=n_centers,
                        n_features=n_features,
                        random_state=random_state,
                        cluster_std=noise
                    )
                elif dataset_type == "Circles":
                    X, y = make_circles(
                        n_samples=n_samples,
                        noise=noise,
                        random_state=random_state
                    )
                elif dataset_type == "Moons":
                    X, y = make_moons(
                        n_samples=n_samples,
                        noise=noise,
                        random_state=random_state
                    )
                
                # Create DataFrame
                feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
                data = pd.DataFrame(X, columns=feature_names)
                data['True_Label'] = y
                
                st.subheader("Generated Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                
                # Visualize if 2D
                if X.shape[1] == 2:
                    fig = px.scatter(
                        data, 
                        x='Feature_1', 
                        y='Feature_2', 
                        color='True_Label',
                        title=f"Generated {dataset_type} Dataset"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif data_source == "Use Sample Dataset":
            sample_choice = st.selectbox(
                "Select Sample Dataset",
                ["Iris", "Wine", "Breast Cancer"]
            )
            
            if st.button("Load Sample Data"):
                from sklearn.datasets import load_iris, load_wine, load_breast_cancer
                
                if sample_choice == "Iris":
                    dataset = load_iris()
                elif sample_choice == "Wine":
                    dataset = load_wine()
                elif sample_choice == "Breast Cancer":
                    dataset = load_breast_cancer()
                
                data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
                data['True_Label'] = dataset.target
                
                st.subheader("Sample Data Preview")
                st.dataframe(data.head(), use_container_width=True)
                st.write(f"Dataset shape: {data.shape}")
        
        # Clustering configuration
        if data is not None:
            st.subheader("Clustering Configuration")
            
            # Feature selection
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'True_Label' in numeric_columns:
                numeric_columns.remove('True_Label')
            
            selected_features = st.multiselect(
                "Select Features for Clustering",
                numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))]
            )
            
            if selected_features:
                # Model configuration
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    model_name = st.text_input("Model Name", value="Clustering_Model")
                    algorithm = st.selectbox(
                        "Clustering Algorithm",
                        ["K-Means", "DBSCAN", "Gaussian Mixture", "BIRCH", "OPTICS"]
                    )
                    
                    # Algorithm-specific parameters
                    if algorithm == "K-Means":
                        n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=20, value=3)
                        max_iter = st.number_input("Max Iterations", min_value=100, max_value=1000, value=300)
                        random_state = st.number_input("Random State", min_value=0, max_value=999, value=42, key="kmeans_rs")
                    
                    elif algorithm == "DBSCAN":
                        eps = st.number_input("Epsilon (eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                        min_samples = st.number_input("Min Samples", min_value=2, max_value=50, value=5)
                    
                    elif algorithm == "Gaussian Mixture":
                        n_components = st.number_input("Number of Components", min_value=2, max_value=20, value=3)
                        covariance_type = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
                        random_state = st.number_input("Random State", min_value=0, max_value=999, value=42, key="gmm_rs")
                    
                    elif algorithm == "BIRCH":
                        n_clusters = st.number_input("Number of Clusters", min_value=2, max_value=20, value=3, key="birch_clusters")
                        threshold = st.number_input("Threshold", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                    
                    elif algorithm == "OPTICS":
                        min_samples = st.number_input("Min Samples", min_value=2, max_value=50, value=5, key="optics_min_samples")
                        max_eps = st.number_input("Max Epsilon", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
                
                with col2:
                    # Preprocessing options
                    st.subheader("Preprocessing")
                    scale_features = st.checkbox("Standardize Features", value=True)
                    
                    # Dimensionality reduction
                    apply_pca = st.checkbox("Apply PCA for Dimensionality Reduction", value=False)
                    if apply_pca:
                        n_components_pca = st.number_input(
                            "PCA Components",
                            min_value=2,
                            max_value=min(len(selected_features), 10),
                            value=min(3, len(selected_features))
                        )
                    
                    # Validation options
                    st.subheader("Validation")
                    calculate_metrics = st.checkbox("Calculate Clustering Metrics", value=True)
                    save_to_db = st.checkbox("Save Model to Database", value=True)
                
                # Training button
                if st.button("🚀 Start Clustering", type="primary", use_container_width=True):
                    with st.spinner("Training clustering model..."):
                        try:
                            # Prepare data
                            X = data[selected_features].values
                            
                            # Initialize trainer
                            trainer = ClusteringTrainer()
                            
                            # Set algorithm parameters
                            if algorithm == "K-Means":
                                trainer.set_algorithm("K-Means", 
                                    n_clusters=n_clusters,
                                    max_iter=max_iter,
                                    random_state=random_state
                                )
                            elif algorithm == "DBSCAN":
                                trainer.set_algorithm("DBSCAN",
                                    eps=eps,
                                    min_samples=min_samples
                                )
                            elif algorithm == "Gaussian Mixture":
                                trainer.set_algorithm("K-Means",  # Use K-Means as fallback for now
                                    n_clusters=n_components,
                                    random_state=random_state
                                )
                            elif algorithm == "BIRCH":
                                trainer.set_algorithm("K-Means",  # Use K-Means as fallback for now
                                    n_clusters=n_clusters,
                                    random_state=42
                                )
                            elif algorithm == "OPTICS":
                                trainer.set_algorithm("DBSCAN",  # Use DBSCAN as fallback for now
                                    eps=0.5,
                                    min_samples=min_samples
                                )
                            
                            # Train model
                            labels = trainer.train(X, normalize=scale_features)
                            metrics = trainer.get_metrics()
                            
                            # Save model if requested
                            if save_to_db:
                                # Create models directory if it doesn't exist
                                os.makedirs("models", exist_ok=True)
                                model_path = f"models/{model_name}.pkl"
                                trainer.save_model(model_path)
                                
                                model_id = db.add_model(
                                    name=model_name,
                                    model_type="Clustering",
                                    file_path=model_path,
                                    accuracy=metrics.get('silhouette_score') if metrics else None,
                                    parameters=json.dumps({
                                        'algorithm': algorithm,
                                        'features': selected_features,
                                        'scale_features': scale_features,
                                        'apply_pca': apply_pca,
                                        'n_clusters': trainer.n_clusters,
                                        **metrics
                                    })
                                )
                                st.success(f"✅ Model '{model_name}' trained and saved successfully!")
                                st.info(f"Model ID: {model_id}")
                            else:
                                st.success(f"✅ Model '{model_name}' trained successfully!")
                            
                            # Display results
                            st.subheader("Clustering Results")
                            
                            # Add cluster labels to data
                            result_data = data[selected_features].copy()
                            result_data['Cluster'] = labels
                            
                            # Basic statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                unique_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise points (-1)
                                st.metric("Number of Clusters", unique_clusters)
                            with col2:
                                if -1 in labels:  # Noise points exist
                                    noise_points = np.sum(labels == -1)
                                    st.metric("Noise Points", noise_points)
                                else:
                                    st.metric("Noise Points", 0)
                            with col3:
                                if metrics and 'silhouette_score' in metrics:
                                    st.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
                            
                            # Display clustering metrics
                            if metrics:
                                st.subheader("Clustering Metrics")
                                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                                st.table(metrics_df)
                            
                            # Visualization
                            st.subheader("Cluster Visualization")
                            
                            if len(selected_features) >= 2:
                                # 2D visualization
                                if len(selected_features) == 2:
                                    fig = px.scatter(
                                        result_data,
                                        x=selected_features[0],
                                        y=selected_features[1],
                                        color='Cluster',
                                        title="Clustering Results (2D)"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # PCA visualization for high-dimensional data
                                elif len(selected_features) > 2:
                                    pca_viz = PCA(n_components=2)
                                    X_pca = pca_viz.fit_transform(StandardScaler().fit_transform(X))
                                    
                                    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                                    pca_df['Cluster'] = labels
                                    
                                    fig = px.scatter(
                                        pca_df,
                                        x='PC1',
                                        y='PC2',
                                        color='Cluster',
                                        title="Clustering Results (PCA Projection)"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show explained variance
                                    st.info(f"PCA Explained Variance: {pca_viz.explained_variance_ratio_.sum():.2%}")
                            
                            # Cluster statistics
                            st.subheader("Cluster Statistics")
                            cluster_stats = result_data.groupby('Cluster').agg(['mean', 'std', 'count']).round(3)
                            st.dataframe(cluster_stats, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Clustering failed: {str(e)}")
    
    with tab2:
        st.header("Data Analysis")
        
        # Load or use existing data
        if 'data' not in locals() or data is None:
            st.info("📁 Please load data in the 'Train Clusters' tab first, or upload data here.")
            
            uploaded_file = st.file_uploader("Upload CSV file for analysis", type=['csv'], key="analysis_upload")
            if uploaded_file is not None:
                data = pd.read_csv(uploaded_file)
        
        if 'data' in locals() and data is not None:
            st.subheader("Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Numeric Features", len(data.select_dtypes(include=[np.number]).columns))
            with col4:
                missing_values = data.isnull().sum().sum()
                st.metric("Missing Values", missing_values)
            
            # Data types and missing values
            st.subheader("Data Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types**")
                dtype_df = pd.DataFrame({
                    'Column': data.columns,
                    'Type': data.dtypes,
                    'Non-Null Count': data.count(),
                    'Missing': data.isnull().sum()
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.write("**Statistical Summary**")
                st.dataframe(data.describe(), use_container_width=True)
            
            # Correlation analysis
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.subheader("Correlation Analysis")
                
                corr_matrix = data[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Highly correlated pairs
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # High correlation threshold
                            high_corr_pairs.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_val
                            ))
                
                if high_corr_pairs:
                    st.subheader("Highly Correlated Features (|r| > 0.7)")
                    high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
                    st.dataframe(high_corr_df, use_container_width=True)
            
            # Distribution analysis
            st.subheader("Feature Distributions")
            
            if len(numeric_cols) > 0:
                selected_feature = st.selectbox("Select feature to analyze", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig_hist = px.histogram(
                        data,
                        x=selected_feature,
                        title=f"Distribution of {selected_feature}",
                        nbins=30
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(
                        data,
                        y=selected_feature,
                        title=f"Box Plot of {selected_feature}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        st.header("Advanced Visualizations")
        
        if 'data' in locals() and data is not None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                st.subheader("Dimensionality Reduction Visualization")
                
                # Feature selection for visualization
                viz_features = st.multiselect(
                    "Select features for visualization",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))]
                )
                
                if len(viz_features) >= 2:
                    viz_method = st.selectbox("Visualization Method", ["PCA", "t-SNE"])
                    
                    # Prepare data
                    X_viz = data[viz_features].values
                    X_scaled = StandardScaler().fit_transform(X_viz)
                    
                    if viz_method == "PCA":
                        # PCA visualization
                        pca = PCA(n_components=2)
                        X_reduced = pca.fit_transform(X_scaled)
                        
                        viz_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
                        
                        # Color by different features if available
                        color_by = st.selectbox("Color by", ['None'] + list(data.columns))
                        
                        if color_by != 'None':
                            viz_df[color_by] = data[color_by].values
                            fig = px.scatter(
                                viz_df,
                                x='PC1',
                                y='PC2',
                                color=color_by,
                                title=f"PCA Visualization (Colored by {color_by})"
                            )
                        else:
                            fig = px.scatter(
                                viz_df,
                                x='PC1',
                                y='PC2',
                                title="PCA Visualization"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show explained variance
                        st.info(f"Explained variance ratio: PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}")
                    
                    elif viz_method == "t-SNE":
                        # t-SNE visualization
                        perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30)
                        
                        if st.button("Generate t-SNE"):
                            with st.spinner("Computing t-SNE..."):
                                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                                X_reduced = tsne.fit_transform(X_scaled)
                                
                                viz_df = pd.DataFrame(X_reduced, columns=['t-SNE 1', 't-SNE 2'])
                                
                                # Color by different features if available
                                color_by = st.selectbox("Color by", ['None'] + list(data.columns), key="tsne_color")
                                
                                if color_by != 'None':
                                    viz_df[color_by] = data[color_by].values
                                    fig = px.scatter(
                                        viz_df,
                                        x='t-SNE 1',
                                        y='t-SNE 2',
                                        color=color_by,
                                        title=f"t-SNE Visualization (Colored by {color_by})"
                                    )
                                else:
                                    fig = px.scatter(
                                        viz_df,
                                        x='t-SNE 1',
                                        y='t-SNE 2',
                                        title="t-SNE Visualization"
                                    )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                # Pairwise scatter plots
                if len(viz_features) >= 2:
                    st.subheader("Pairwise Feature Relationships")
                    
                    feature_pair = st.selectbox(
                        "Select feature pair",
                        [(f1, f2) for i, f1 in enumerate(viz_features) for f2 in viz_features[i+1:]]
                    )
                    
                    if feature_pair:
                        fig_pair = px.scatter(
                            data,
                            x=feature_pair[0],
                            y=feature_pair[1],
                            title=f"{feature_pair[0]} vs {feature_pair[1]}"
                        )
                        st.plotly_chart(fig_pair, use_container_width=True)
            else:
                st.warning("⚠️ Need at least 2 numeric features for visualization.")
        else:
            st.info("📁 Please load data first.")
    
    with tab4:
        st.header("Make Predictions")
        
        # Model selection
        models = db.get_models_by_type("Clustering")
        if models.empty:
            st.warning("⚠️ No trained clustering models found. Please train a model first.")
            return
        
        model_options = [f"{row['name']} (ID: {row['id']})" for _, row in models.iterrows()]
        selected_model = st.selectbox("Select Model", model_options)
        model_id = int(selected_model.split("ID: ")[1].split(")")[0])
        
        # Get model details
        model_row = models[models['id'] == model_id].iloc[0]
        try:
            model_params = json.loads(model_row['parameters']) if model_row['parameters'] else {}
        except (json.JSONDecodeError, TypeError):
            model_params = {}
        
        st.subheader("Model Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Algorithm", model_params.get('algorithm', 'Unknown'))
        with col2:
            st.metric("Features", len(model_params.get('features', [])))
        with col3:
            if model_row['accuracy']:
                st.metric("Silhouette Score", f"{model_row['accuracy']:.3f}")
        
        # Data input for prediction
        st.subheader("Input Data for Clustering")
        
        input_method = st.radio("Input Method", ["Manual Input", "Upload CSV"])
        
        if input_method == "Manual Input":
            # Manual feature input
            features = model_params.get('features', [])
            input_values = {}
            
            cols = st.columns(min(3, len(features)))
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    input_values[feature] = st.number_input(f"{feature}", key=f"input_{feature}")
            
            if st.button("🔍 Predict Cluster", type="primary"):
                try:
                    # Initialize predictor
                    predictor = ClusteringPredictor(model_row['file_path'])  # file_path
                    
                    # Prepare input data
                    input_array = np.array([[input_values[feature] for feature in features]])
                    
                    # Make prediction
                    cluster_label, cluster_center, distance = predictor.predict(input_array)
                    
                    # Save prediction to database
                    db.save_prediction(
                        model_id=model_id,
                        input_data=input_values,
                        prediction=f"Cluster {cluster_label[0]}",
                        confidence=1.0 / (1.0 + distance[0])  # Convert distance to confidence-like score
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Predicted Cluster:** {cluster_label[0]}")
                    with col2:
                        st.info(f"**Distance to Center:** {distance[0]:.3f}")
                    
                    if cluster_center is not None:
                        st.subheader("Cluster Center")
                        center_df = pd.DataFrame({
                            'Feature': features,
                            'Cluster Center': cluster_center[cluster_label[0]],
                            'Input Value': [input_values[f] for f in features],
                            'Difference': [input_values[f] - cluster_center[cluster_label[0]][i] for i, f in enumerate(features)]
                        })
                        st.dataframe(center_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
        
        elif input_method == "Upload CSV":
            uploaded_pred_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'], key="pred_upload")
            
            if uploaded_pred_file is not None:
                pred_data = pd.read_csv(uploaded_pred_file)
                st.subheader("Data Preview")
                st.dataframe(pred_data.head(), use_container_width=True)
                
                # Check if required features are present
                required_features = model_params.get('features', [])
                missing_features = [f for f in required_features if f not in pred_data.columns]
                
                if missing_features:
                    st.error(f"❌ Missing required features: {missing_features}")
                else:
                    if st.button("🔍 Predict Clusters (Batch)", type="primary"):
                        try:
                            # Initialize predictor
                            predictor = ClusteringPredictor(model_row['file_path'])
                            
                            # Prepare input data
                            input_array = pred_data[required_features].values
                            
                            # Make predictions
                            cluster_labels, cluster_centers, distances = predictor.predict(input_array)
                            
                            # Add results to dataframe
                            pred_data['Predicted_Cluster'] = cluster_labels
                            pred_data['Distance_to_Center'] = distances
                            
                            st.subheader("Prediction Results")
                            st.dataframe(pred_data, use_container_width=True)
                            
                            # Download results
                            csv_buffer = BytesIO()
                            pred_data.to_csv(csv_buffer, index=False)
                            
                            st.download_button(
                                label="📥 Download Results",
                                data=csv_buffer.getvalue(),
                                file_name="clustering_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Cluster distribution
                            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                            fig_dist = px.bar(
                                x=cluster_counts.index,
                                y=cluster_counts.values,
                                title="Predicted Cluster Distribution"
                            )
                            st.plotly_chart(fig_dist, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Batch prediction failed: {str(e)}")
    
    with tab5:
        st.header("Model Management")
        
        # List all clustering models
        models = db.get_models_by_type("Clustering")
        
        if not models.empty:
            st.subheader("Trained Models")
            
            # Create models dataframe for display
            models_display = models.copy()
            for idx, row in models_display.iterrows():
                try:
                    params = json.loads(row['parameters']) if row['parameters'] else {}
                except (json.JSONDecodeError, TypeError):
                    params = {}
                models_display.at[idx, 'Algorithm'] = params.get('algorithm', 'Unknown')
                models_display.at[idx, 'Feature Count'] = len(params.get('features', []))
                models_display.at[idx, 'Silhouette Score'] = f"{row['accuracy']:.3f}" if row['accuracy'] else "N/A"
            
            # Select columns to display
            display_cols = ['id', 'name', 'Algorithm', 'Feature Count', 'Silhouette Score', 'created_at']
            available_cols = [col for col in display_cols if col in models_display.columns]
            st.dataframe(models_display[available_cols], use_container_width=True)
            
            # Model comparison
            if len(models) >= 2:
                st.subheader("Model Comparison")
                
                comparison_models = st.multiselect(
                    "Select models to compare",
                    [f"{row['name']} (ID: {row['id']})" for _, row in models.iterrows()],
                    max_selections=5
                )
                
                if len(comparison_models) >= 2:
                    comparison_data = []
                    for model_str in comparison_models:
                        model_id = int(model_str.split("ID: ")[1].split(")")[0])
                        model_row = models[models['id'] == model_id].iloc[0]
                        try:
                            params = json.loads(model_row['parameters']) if model_row['parameters'] else {}
                        except (json.JSONDecodeError, TypeError):
                            params = {}
                        
                        comparison_data.append({
                            'Model': model_row['name'],
                            'Algorithm': params.get('algorithm', 'Unknown'),
                            'Features': len(params.get('features', [])),
                            'Silhouette Score': model_row['accuracy'] if model_row['accuracy'] else 0,
                            'Created': model_row['created_at']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Performance comparison chart
                    if 'Silhouette Score' in comparison_df.columns:
                        fig_comparison = px.bar(
                            comparison_df,
                            x='Model',
                            y='Silhouette Score',
                            title="Model Performance Comparison"
                        )
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Detailed comparison table
                    st.dataframe(comparison_df, use_container_width=True)
            
            # Model deletion
            st.subheader("Model Management Actions")
            
            delete_model = st.selectbox(
                "Select model to delete",
                ["None"] + [f"{row['name']} (ID: {row['id']})" for _, row in models.iterrows()]
            )
            
            if delete_model != "None":
                if st.button("🗑️ Delete Model", type="secondary"):
                    model_id = int(delete_model.split("ID: ")[1].split(")")[0])
                    try:
                        # Delete from database
                        db.delete_model(model_id)
                        st.success("✅ Model deleted successfully!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to delete model: {str(e)}")
        
        else:
            st.info("📝 No clustering models found. Train your first model in the 'Train Clusters' tab!")

if __name__ == "__main__":
    show_clustering_page()
