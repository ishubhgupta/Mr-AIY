"""
Neural Networks module page for the Streamlit application.
Handles various neural network architectures for tabular and sequence data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import io
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import database and neural network modules
try:
    from database import DatabaseManager
    from code.neural_networks import train_mlp_model, train_lstm_model, prepare_tabular_data, predict_with_nn_model
except ImportError:
    st.error("Failed to import required modules. Please check your installation.")

def show_neural_networks_page():
    """Main Neural Networks page function."""
    st.markdown("# 🧠 Neural Networks")
    
    # Initialize session state for data persistence
    if 'nn_data' not in st.session_state:
        st.session_state.nn_data = None
    if 'nn_target_column' not in st.session_state:
        st.session_state.nn_target_column = None
    if 'nn_feature_columns' not in st.session_state:
        st.session_state.nn_feature_columns = []
    if 'nn_task_type' not in st.session_state:
        st.session_state.nn_task_type = "Classification"
    
    tabs = st.tabs(["📚 Train Model", "🎯 Make Predictions", "📊 Data Analysis", "📈 Model Comparison", "⚙️ Advanced Settings"])
    
    with tabs[0]:
        show_nn_training()
    
    with tabs[1]:
        show_nn_prediction()
    
    with tabs[2]:
        show_data_analysis()
    
    with tabs[3]:
        show_model_comparison()
    
    with tabs[4]:
        show_advanced_settings()

def show_nn_training():
    """Neural network training interface."""
    st.subheader("🚀 Train Neural Network")
    
    # Model type selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Multi-Layer Perceptron (MLP)", "LSTM", "GRU", "CNN-1D"]
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔧 Model Configuration")
        
        model_name = st.text_input(
            "Model Name", 
            value=f"nn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if model_type == "Multi-Layer Perceptron (MLP)":
            config = show_mlp_config()
        elif model_type in ["LSTM", "GRU"]:
            config = show_rnn_config(model_type)
        elif model_type == "CNN-1D":
            config = show_cnn1d_config()
        
        # Common training parameters
        st.markdown("#### 🎛️ Training Parameters")
        epochs = st.slider("Epochs", min_value=10, max_value=500, value=100)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=2)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
        
        optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop", "AdamW"])
        
        # Validation split
        validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    
    with col2:
        st.markdown("### 📁 Data Upload")
        
        data_type = st.selectbox("Data Type", ["Tabular Data", "Time Series"])
        
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=['csv', 'xlsx', 'json']
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    return
                
                # Store data in session state
                st.session_state.nn_data = df
                
                st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Show data preview
                st.markdown("#### 👀 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Data configuration
                show_data_configuration(df, data_type)
                
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
        
        elif st.session_state.nn_data is not None:
            # Show already loaded data
            df = st.session_state.nn_data
            st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            show_data_configuration(df, data_type)
        
        else:
            # Option to use sample data
            if st.button("📝 Use Sample Dataset"):
                df = create_sample_dataset(data_type)
                st.session_state.nn_data = df
                st.success("✅ Sample dataset loaded!")
                st.dataframe(df.head(), use_container_width=True)
                st.experimental_rerun()
    
    # Training button
    if st.session_state.nn_data is not None and model_name:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Training", type="primary", use_container_width=True):
                with st.spinner("🔄 Training neural network... This may take a while."):
                    try:
                        # Get data from session state
                        df = st.session_state.nn_data
                        target_col = st.session_state.nn_target_column
                        
                        if target_col and target_col in df.columns:
                            # Prepare data for training
                            X = df.drop(columns=[target_col])
                            y = df[target_col]
                            
                            # Handle categorical columns
                            for col in X.select_dtypes(include=['object']).columns:
                                X[col] = pd.Categorical(X[col]).codes
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # Encode target if categorical
                            label_encoder = None
                            if y.dtype == 'object':
                                label_encoder = LabelEncoder()
                                y_encoded = label_encoder.fit_transform(y)
                            else:
                                y_encoded = y.values
                            
                            # Split data
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_scaled, y_encoded, test_size=validation_split, random_state=42
                            )
                            
                            # Create models directory
                            models_dir = "models/neural_networks"
                            os.makedirs(models_dir, exist_ok=True)
                            
                            # Train model
                            if model_type == "Multi-Layer Perceptron (MLP)":
                                results = train_mlp_model(
                                    X_train, y_train, X_val, y_val,
                                    hidden_sizes=config.get('layers', [128, 64]),
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    model_save_dir=models_dir
                                )
                            elif model_type == "LSTM":
                                results = train_lstm_model(
                                    X_train, y_train, X_val, y_val,
                                    hidden_size=config.get('hidden_size', 64),
                                    num_layers=config.get('num_layers', 2),
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    model_save_dir=models_dir
                                )
                            
                            # Display results
                            st.success("🎉 Training completed successfully!")
                            
                            # Show metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Best Validation Loss", f"{results['best_val_loss']:.4f}")
                            with col2:
                                st.metric("Training Time", results['training_time'])
                            with col3:
                                if 'accuracy' in results['metrics']:
                                    st.metric("Accuracy", f"{results['metrics']['accuracy']:.4f}")
                            
                            # Show training plot
                            if results.get('plot_path'):
                                st.image(results['plot_path'])
                            

                            # Save model info to database
                            try:
                                db_manager = DatabaseManager()
                                model_type_short = model_type.split(' ')[0]  # Get short type (MLP, LSTM, etc.)
                                
                                # Create model record using the correct method with proper error handling
                                try:
                                    model_id = db_manager.add_model(
                                        name=model_name,
                                        model_type=model_type_short,
                                        file_path=results.get('model_path', ''),
                                        parameters=json.dumps({
                                            'epochs': epochs,
                                            'batch_size': batch_size,
                                            'learning_rate': learning_rate,
                                            'hidden_sizes': config.get('layers', []) if model_type == "Multi-Layer Perceptron (MLP)" else [],
                                            'hidden_size': config.get('hidden_size', 64) if model_type == "LSTM" else None,
                                            'num_layers': config.get('num_layers', 2) if model_type == "LSTM" else None
                                        }),
                                        loss=results.get('best_val_loss'),
                                        description=f"{model_type} trained on {datetime.now().strftime('%Y-%m-%d')}"
                                    )
                                    st.success(f"✅ Model information saved to database (ID: {model_id})")
                                except TypeError as te:
                                    # Try with different parameter names if the method signature is different
                                    try:
                                        model_id = db_manager.add_model(
                                            name=model_name,
                                            model_type=model_type_short,
                                            file_path=results.get('model_path', ''),
                                            parameters=json.dumps({
                                                'epochs': epochs,
                                                'batch_size': batch_size,
                                                'learning_rate': learning_rate,
                                                'model_config': config
                                            }),
                                            accuracy=results['metrics'].get('accuracy', None),
                                            description=f"{model_type} trained on {datetime.now().strftime('%Y-%m-%d')}"
                                        )
                                        st.success(f"✅ Model information saved to database (ID: {model_id})")
                                    except Exception as e2:
                                        st.warning(f"⚠️ Could not save to database: {str(e2)}")
                                        st.info("Model training completed successfully, but database save failed. Model files are still saved locally.")
                            except Exception as e:
                                st.error(f"❌ Failed to save model info to database: {str(e)}")
                                st.info("Model training completed but database save failed. Model files are still saved locally.")
                        
                        else:
                            st.error("❌ Please select a target column")
                            
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        st.exception(e)
    else:
        st.warning("⚠️ Please upload a dataset and provide a model name to start training.")

def show_mlp_config():
    """MLP specific configuration."""
    st.markdown("#### MLP Configuration")
    
    num_layers = st.slider("Number of Hidden Layers", min_value=1, max_value=10, value=3)
    
    layer_sizes = []
    for i in range(num_layers):
        size = st.number_input(f"Layer {i+1} Size", min_value=8, max_value=1024, value=128, key=f"mlp_layer_{i}")
        layer_sizes.append(size)
    
    activation = st.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid", "LeakyReLU", "ELU"])
    dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.2, step=0.1)
    
    return {
        'type': 'MLP',
        'layers': layer_sizes,
        'activation': activation,
        'dropout': dropout_rate
    }

def show_rnn_config(model_type):
    """RNN (LSTM/GRU) specific configuration."""
    st.markdown(f"#### {model_type} Configuration")
    
    hidden_size = st.number_input("Hidden Size", min_value=32, max_value=512, value=128)
    num_layers = st.slider("Number of Layers", min_value=1, max_value=5, value=2)
    sequence_length = st.number_input("Sequence Length", min_value=5, max_value=200, value=50)
    
    bidirectional = st.checkbox("Bidirectional", value=False)
    dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.2, step=0.1)
    
    return {
        'type': model_type,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'sequence_length': sequence_length,
        'bidirectional': bidirectional,
        'dropout': dropout_rate
    }

def show_cnn1d_config():
    """1D CNN specific configuration."""
    st.markdown("#### CNN-1D Configuration")
    
    num_conv_layers = st.slider("Number of Conv Layers", min_value=1, max_value=5, value=3)
    
    filter_sizes = []
    kernel_sizes = []
    for i in range(num_conv_layers):
        filters = st.number_input(f"Conv Layer {i+1} Filters", min_value=16, max_value=512, value=64, key=f"cnn1d_filters_{i}")
        kernel = st.number_input(f"Conv Layer {i+1} Kernel Size", min_value=2, max_value=15, value=3, key=f"cnn1d_kernel_{i}")
        filter_sizes.append(filters)
        kernel_sizes.append(kernel)
    
    pool_size = st.number_input("Pool Size", min_value=2, max_value=10, value=2)
    dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.3, step=0.1)
    
    return {
        'type': 'CNN-1D',
        'filters': filter_sizes,
        'kernels': kernel_sizes,
        'pool_size': pool_size,
        'dropout': dropout_rate
    }

def show_data_configuration(df, data_type):
    """Data configuration interface."""
    st.markdown("#### ⚙️ Data Configuration")
    
    if data_type == "Tabular Data":
        target_column = st.selectbox("🎯 Target Column", df.columns.tolist())
        st.session_state.nn_target_column = target_column
        
        feature_columns = st.multiselect("📊 Feature Columns", 
                                       [col for col in df.columns if col != target_column],
                                       default=[col for col in df.columns if col != target_column][:10])  # Limit default selection
        st.session_state.nn_feature_columns = feature_columns
        
        task_type = st.selectbox("📋 Task Type", ["Classification", "Regression"])
        st.session_state.nn_task_type = task_type
        
        # Data preprocessing options
        st.markdown("#### 🔧 Preprocessing Options")
        normalize_features = st.checkbox("Normalize Features", value=True)
        handle_missing = st.selectbox("Handle Missing Values", ["Drop", "Fill Mean", "Fill Median", "Fill Mode"])
    
    elif data_type == "Time Series":
        target_column = st.selectbox("🎯 Target Column", df.columns.tolist())
        st.session_state.nn_target_column = target_column
        
        time_column = st.selectbox("⏰ Time Column", df.columns.tolist())
        
        forecast_horizon = st.number_input("🔮 Forecast Horizon", min_value=1, max_value=50, value=10)
        lookback_window = st.number_input("👁️ Lookback Window", min_value=5, max_value=200, value=30)

def show_nn_prediction():
    """Neural network prediction interface."""
    st.subheader("🎯 Make Predictions")
    
    # Initialize database
    try:
        db_manager = DatabaseManager()
        nn_models = db_manager.get_models_by_type('MLP')
        if nn_models.empty:
            # Also check for LSTM models
            lstm_models = db_manager.get_models_by_type('LSTM')
            if not lstm_models.empty:
                nn_models = lstm_models
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        nn_models = pd.DataFrame()
    
    if not nn_models.empty:
        st.success(f"✅ Found {len(nn_models)} trained models")
        
        selected_model = st.selectbox(
            "🔍 Select Model",
            nn_models['name'].tolist() if 'name' in nn_models.columns else nn_models.index.tolist()
        )
        
        # Get model details
        if 'name' in nn_models.columns:
            model_info = nn_models[nn_models['name'] == selected_model].iloc[0]
        else:
            model_info = nn_models.iloc[0]
        
        # Display model info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Type", model_info.get('type', 'Neural Network'))
        with col2:
            if 'accuracy' in model_info:
                st.metric("Accuracy", f"{model_info['accuracy']:.4f}")
        with col3:
            st.metric("Created", str(model_info.get('created_at', 'Unknown'))[:10])
        
        # Prediction input method
        input_method = st.selectbox(
            "📝 Input Method",
            ["Manual Input", "Upload File", "Batch Prediction"]
        )
        
        if input_method == "Manual Input":
            show_manual_input_interface(model_info)
        elif input_method == "Upload File":
            show_file_upload_interface(model_info)
        elif input_method == "Batch Prediction":
            show_batch_prediction_interface(model_info)
    
    else:
        st.info("ℹ️ No neural network models found. Please train a model first.")
        
        # Demo prediction option
        if st.button("🎮 Try Demo Model"):
            show_demo_prediction()

def show_manual_input_interface(model_info):
    """Manual input interface for predictions."""
    st.markdown("### ✏️ Manual Input")
    
    # Get expected features from the session state or create sample ones
    if st.session_state.nn_feature_columns:
        features = st.session_state.nn_feature_columns[:10]  # Limit to 10 features for UI
    else:
        features = [f"Feature_{i+1}" for i in range(6)]
    
    # Create input fields dynamically
    input_values = {}
    cols = st.columns(min(3, len(features)))
    
    for i, feature in enumerate(features):
        with cols[i % 3]:
            input_values[feature] = st.number_input(f"{feature}", value=0.0, key=f"manual_input_{feature}")
    
    if st.button("🔮 Predict", type="primary"):
        try:
            # Create input array
            input_array = np.array([[input_values[f] for f in features]])
            
            # Simulate prediction (replace with actual model prediction)
            prediction = simulate_prediction(input_array[0])
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Prediction:** {prediction['class']}")
                st.success(f"**Confidence:** {prediction['confidence']:.2%}")
            
            with col2:
                # Show prediction probabilities
                fig = px.bar(x=prediction['classes'], y=prediction['probabilities'], 
                            title="Class Probabilities")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

def show_file_upload_interface(model_info):
    """File upload interface for predictions."""
    st.markdown("### 📁 File Upload Prediction")
    
    uploaded_file = st.file_uploader("Upload file for prediction", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                pred_data = pd.read_csv(uploaded_file)
            else:
                pred_data = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {pred_data.shape[0]} rows, {pred_data.shape[1]} columns")
            st.dataframe(pred_data.head())
            
            if st.button("🔮 Make Predictions"):
                # Simulate batch predictions
                predictions = []
                for i in range(len(pred_data)):
                    pred = simulate_prediction(pred_data.iloc[i].values[:6])
                    predictions.append(pred['class'])
                
                pred_data['Prediction'] = predictions
                
                st.success("✅ Predictions completed!")
                st.dataframe(pred_data)
                
                # Download button
                csv = pred_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def show_batch_prediction_interface(model_info):
    """Batch prediction interface."""
    st.markdown("### 📊 Batch Prediction")
    st.info("Upload a dataset for batch predictions")
    
    # Similar to file upload but optimized for large files
    show_file_upload_interface(model_info)

def show_data_analysis():
    """Data analysis and visualization interface."""
    st.subheader("📊 Data Analysis")
    
    # Check if data exists in session state
    if st.session_state.nn_data is not None:
        df = st.session_state.nn_data
        st.success(f"✅ Using loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Analysis tabs
        analysis_tabs = st.tabs(["📋 Overview", "📈 Distributions", "🔗 Correlations", "❓ Missing Values", "⚠️ Outliers"])
        
        with analysis_tabs[0]:
            show_data_overview(df)
        
        with analysis_tabs[1]:
            show_data_distributions(df)
        
        with analysis_tabs[2]:
            show_correlations(df)
        
        with analysis_tabs[3]:
            show_missing_values(df)
        
        with analysis_tabs[4]:
            show_outliers(df)
    else:
        st.info("ℹ️ No data loaded. Please upload data in the 'Train Model' tab first.")
        
        # Option to upload data here
        uploaded_file = st.file_uploader("📁 Upload Dataset for Analysis", type=['csv', 'xlsx'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.session_state.nn_data = df
                st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
        
        # Use sample data option
        if st.button("📝 Use Sample Data for Analysis"):
            df = create_sample_dataset("Tabular Data")
            st.session_state.nn_data = df
            st.success("✅ Sample data loaded for analysis!")
            st.experimental_rerun()

def show_data_overview(df):
    """Show data overview."""
    st.markdown("### Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
    
    # Data types
    st.markdown("### Data Types")
    st.dataframe(df.dtypes.to_frame('Data Type'), use_container_width=True)
    
    # Basic statistics
    st.markdown("### Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

def show_data_distributions(df):
    """Show data distributions."""
    st.markdown("### Data Distributions")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_columns:
        selected_column = st.selectbox("Select Column", numeric_columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)

def show_correlations(df):
    """Show correlation analysis."""
    st.markdown("### Correlation Analysis")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr()
        
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        # Strong correlations
        st.markdown("### Strong Correlations (|r| > 0.7)")
        strong_corr = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_corr.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': corr_value
                    })
        
        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
        else:
            st.info("No strong correlations found.")

def show_missing_values(df):
    """Show missing values analysis."""
    st.markdown("### Missing Values Analysis")
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    }).sort_values('Missing Count', ascending=False)
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True)
        
        # Missing values heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, ax=ax)
        ax.set_title("Missing Values Heatmap")
        st.pyplot(fig)
    else:
        st.success("No missing values found in the dataset!")

def show_outliers(df):
    """Show outliers analysis."""
    st.markdown("### Outliers Analysis")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_columns:
        selected_column = st.selectbox("Select Column for Outlier Analysis", numeric_columns)
        
        # Calculate outliers using IQR
        Q1 = df[selected_column].quantile(0.25)
        Q3 = df[selected_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Outliers", len(outliers))
            st.metric("Outlier Percentage", f"{len(outliers)/len(df)*100:.2f}%")
        
        with col2:
            st.metric("Lower Bound", f"{lower_bound:.2f}")
            st.metric("Upper Bound", f"{upper_bound:.2f}")
        
        if len(outliers) > 0:
            st.dataframe(outliers, use_container_width=True)

def show_training_progress():
    """Show training progress simulation."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create placeholder for metrics
    metrics_placeholder = st.empty()
    
    # Simulate training
    epochs = 50
    train_losses = []
    val_losses = []
    accuracies = []
    
    for epoch in range(epochs):
        # Simulate metrics
        train_loss = 2.5 * np.exp(-epoch/20) + np.random.normal(0, 0.1)
        val_loss = 2.7 * np.exp(-epoch/18) + np.random.normal(0, 0.15)
        accuracy = 1 - np.exp(-epoch/15) + np.random.normal(0, 0.02)
        accuracy = max(0, min(1, accuracy))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {accuracy:.4f}')
        
        # Update metrics every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            with metrics_placeholder.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    # Loss plot
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(train_losses, label='Training Loss', color='blue')
                    ax.plot(val_losses, label='Validation Loss', color='red')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.set_title('Training Progress')
                    ax.legend()
                    st.pyplot(fig)
                
                with col2:
                    # Accuracy plot
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(accuracies, label='Accuracy', color='green')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Accuracy')
                    ax.set_title('Model Accuracy')
                    ax.legend()
                    st.pyplot(fig)
    
    st.success("🎉 Training completed successfully!")
    st.balloons()

# Helper functions
def create_sample_dataset(data_type):
    """Create sample dataset for demo."""
    np.random.seed(42)
    
    if data_type == "Tabular Data":
        n_samples = 1000
        n_features = 10
        
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Create target based on some features
        y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    elif data_type == "Time Series":
        dates = pd.date_range('2020-01-01', periods=365)
        values = np.cumsum(np.random.randn(365)) + 100
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'feature1': np.random.randn(365),
            'feature2': np.random.randn(365)
        })
        
        return df

def simulate_prediction(features):
    """Simulate prediction results."""
    # Simple simulation
    score = sum(features) + np.random.randn()
    
    classes = ['Class A', 'Class B', 'Class C']
    probabilities = np.random.dirichlet([1, 1, 1])
    
    predicted_class = classes[np.argmax(probabilities)]
    confidence = np.max(probabilities)
    
    return {
        'class': predicted_class,
        'confidence': confidence,
        'classes': classes,
        'probabilities': probabilities.tolist()
    }

def show_model_comparison():
    """Model comparison interface."""
    st.subheader("📈 Model Comparison")
    
    try:
        db_manager = DatabaseManager()
        
        # Get all neural network models
        mlp_models = db_manager.get_models_by_type('MLP')
        lstm_models = db_manager.get_models_by_type('LSTM')
        
        all_models = pd.concat([mlp_models, lstm_models], ignore_index=True) if not mlp_models.empty or not lstm_models.empty else pd.DataFrame()
        
        if not all_models.empty:
            st.success(f"✅ Found {len(all_models)} models for comparison")
            
            # Model selection for comparison
            selected_models = st.multiselect(
                "🔍 Select Models to Compare",
                all_models['name'].tolist() if 'name' in all_models.columns else all_models.index.tolist(),
                default=all_models['name'].tolist()[:3] if 'name' in all_models.columns else []
            )
            
            if selected_models:
                # Filter selected models
                comparison_data = all_models[all_models['name'].isin(selected_models)] if 'name' in all_models.columns else all_models
                
                # Display comparison table
                st.dataframe(comparison_data, use_container_width=True)
                
                # Performance comparison chart
                if 'accuracy' in comparison_data.columns:
                    fig = px.bar(
                        comparison_data, 
                        x='name', 
                        y='accuracy',
                        title="Model Accuracy Comparison",
                        labels={'accuracy': 'Accuracy', 'name': 'Model Name'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Training time comparison if available
                if 'created_at' in comparison_data.columns:
                    fig2 = px.timeline(
                        comparison_data,
                        x_start='created_at',
                        x_end='created_at',
                        y='name',
                        title="Model Creation Timeline"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("ℹ️ Please select models to compare")
        else:
            st.info("ℹ️ No models found for comparison. Train some models first!")
            
    except Exception as e:
        st.error(f"❌ Error accessing models: {str(e)}")

def show_advanced_settings():
    """Advanced settings interface."""
    st.subheader("⚙️ Advanced Settings")
    
    # Model Management
    st.markdown("### 🗂️ Model Management")
    
    try:
        db_manager = DatabaseManager()
        
        # Get all neural network models
        mlp_models = db_manager.get_models_by_type('MLP')
        lstm_models = db_manager.get_models_by_type('LSTM')
        
        all_models = pd.concat([mlp_models, lstm_models], ignore_index=True) if not mlp_models.empty or not lstm_models.empty else pd.DataFrame()
        
        if not all_models.empty:
            st.dataframe(all_models, use_container_width=True)
            
            # Model deletion
            st.markdown("#### 🗑️ Delete Models")
            model_to_delete = st.selectbox(
                "Select model to delete",
                ["None"] + (all_models['name'].tolist() if 'name' in all_models.columns else [])
            )
            
            if model_to_delete != "None":
                if st.button("🗑️ Delete Model", type="secondary"):
                    try:
                        # Get model info
                        model_info = all_models[all_models['name'] == model_to_delete].iloc[0]
                        model_id = model_info['id'] if 'id' in model_info else None
                        
                        if model_id:
                            db_manager.delete_model(model_id)
                            st.success(f"✅ Model '{model_to_delete}' deleted successfully!")
                            st.experimental_rerun()
                        else:
                            st.error("❌ Could not find model ID")
                    except Exception as e:
                        st.error(f"❌ Failed to delete model: {str(e)}")
        else:
            st.info("ℹ️ No models found")
    
    except Exception as e:
        st.error(f"❌ Error accessing database: {str(e)}")
    
    # System Settings
    st.markdown("### 🔧 System Settings")
    
    # GPU Settings
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        pass
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("GPU Available", "✅ Yes" if gpu_available else "❌ No")
    with col2:
        st.metric("CPU Cores", os.cpu_count() or "Unknown")
    
    # Model Storage Settings
    st.markdown("#### 💾 Storage Settings")
    models_dir = st.text_input("Models Directory", value="models/neural_networks")
    
    if st.button("📁 Create Directory"):
        try:
            os.makedirs(models_dir, exist_ok=True)
            st.success(f"✅ Directory '{models_dir}' created/verified")
        except Exception as e:
            st.error(f"❌ Failed to create directory: {str(e)}")

def show_demo_prediction():
    """Demo prediction interface."""
    st.markdown("### 🎮 Demo Model Prediction")
    st.info("This is a demonstration using a simulated model")
    
    # Create demo input
    col1, col2 = st.columns(2)
    
    with col1:
        demo_input1 = st.slider("Demo Feature 1", -5.0, 5.0, 0.0)
        demo_input2 = st.slider("Demo Feature 2", -5.0, 5.0, 0.0)
        demo_input3 = st.slider("Demo Feature 3", -5.0, 5.0, 0.0)
    
    with col2:
        demo_input4 = st.slider("Demo Feature 4", -5.0, 5.0, 0.0)
        demo_input5 = st.slider("Demo Feature 5", -5.0, 5.0, 0.0)
        demo_input6 = st.slider("Demo Feature 6", -5.0, 5.0, 0.0)
    
    if st.button("🔮 Run Demo Prediction"):
        demo_features = [demo_input1, demo_input2, demo_input3, demo_input4, demo_input5, demo_input6]
        prediction = simulate_prediction(demo_features)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Prediction:** {prediction['class']}")
            st.success(f"**Confidence:** {prediction['confidence']:.2%}")
        
        with col2:
            fig = px.bar(x=prediction['classes'], y=prediction['probabilities'], 
                        title="Demo Prediction Probabilities")
            st.plotly_chart(fig, use_container_width=True)
