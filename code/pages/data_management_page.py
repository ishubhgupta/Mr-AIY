"""
Data Management Page for AI Vision Suite
Handles data upload, preprocessing, visualization, and management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import json
import io
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import zipfile
import pickle

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from database import DatabaseManager

def show_data_management_page():
    """Display the Data Management page with all functionality"""
    
    st.title("📊 Data Management")
    st.markdown("### Comprehensive data handling, preprocessing, and analysis")
    st.markdown("---")
    
    # Initialize session state
    if 'dm_data' not in st.session_state:
        st.session_state.dm_data = None
    if 'dm_processed_data' not in st.session_state:
        st.session_state.dm_processed_data = None
    if 'dm_data_history' not in st.session_state:
        st.session_state.dm_data_history = []
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Data Upload", "🔍 Data Explorer", "🛠️ Preprocessing", 
        "📈 Visualization", "💾 Data Storage", "📋 Data Catalog"
    ])
    
    with tab1:
        show_data_upload()
    
    with tab2:
        show_data_explorer()
    
    with tab3:
        show_data_preprocessing()
    
    with tab4:
        show_data_visualization()
    
    with tab5:
        show_data_storage()
    
    with tab6:
        show_data_catalog()

def show_data_upload():
    """Data upload interface with multiple sources and formats"""
    st.header("📁 Data Upload & Import")
    
    # Data source selection
    data_source = st.selectbox(
        "🔗 Select Data Source",
        ["Upload File", "Sample Datasets", "Database Connection", "URL Import", "Manual Entry"]
    )
    
    if data_source == "Upload File":
        show_file_upload()
    elif data_source == "Sample Datasets":
        show_sample_datasets()
    elif data_source == "Database Connection":
        show_database_connection()
    elif data_source == "URL Import":
        show_url_import()
    elif data_source == "Manual Entry":
        show_manual_entry()

def show_file_upload():
    """File upload interface"""
    st.subheader("📤 Upload Data Files")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: CSV, Excel, JSON, Parquet, TXT"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 {uploaded_file.name}", expanded=True):
                try:
                    # Load data based on file type
                    if uploaded_file.name.endswith('.csv'):
                        # CSV options
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            delimiter = st.selectbox("Delimiter", [',', ';', '\t', '|'], key=f"delim_{uploaded_file.name}")
                        with col2:
                            encoding = st.selectbox("Encoding", ['utf-8', 'latin-1', 'iso-8859-1'], key=f"enc_{uploaded_file.name}")
                        with col3:
                            header_row = st.number_input("Header Row", min_value=0, max_value=10, value=0, key=f"header_{uploaded_file.name}")
                        
                        df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding, header=header_row)
                        
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        # Excel options
                        col1, col2 = st.columns(2)
                        with col1:
                            sheet_name = st.text_input("Sheet Name", value=0, key=f"sheet_{uploaded_file.name}")
                        with col2:
                            header_row = st.number_input("Header Row", min_value=0, max_value=10, value=0, key=f"header_{uploaded_file.name}")
                        
                        try:
                            sheet_name = int(sheet_name)
                        except:
                            pass
                        
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row)
                        
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(uploaded_file)
                        
                    elif uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(uploaded_file)
                        
                    elif uploaded_file.name.endswith('.txt'):
                        # Text file options
                        delimiter = st.selectbox("Delimiter", ['\t', ',', ';', '|', ' '], key=f"txt_delim_{uploaded_file.name}")
                        df = pd.read_csv(uploaded_file, delimiter=delimiter)
                    
                    # Display data info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", df.shape[0])
                    with col2:
                        st.metric("Columns", df.shape[1])
                    with col3:
                        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                    with col4:
                        st.metric("Missing Values", df.isnull().sum().sum())
                    
                    # Data preview
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Load data button
                    if st.button(f"Load {uploaded_file.name}", key=f"load_{uploaded_file.name}"):
                        st.session_state.dm_data = df
                        st.session_state.dm_data_history.append({
                            'name': uploaded_file.name,
                            'timestamp': datetime.now(),
                            'shape': df.shape,
                            'source': 'file_upload'
                        })
                        st.success(f"✅ {uploaded_file.name} loaded successfully!")
                        st.experimental_rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")

def show_sample_datasets():
    """Sample datasets interface"""
    st.subheader("📊 Sample Datasets")
    
    # Available sample datasets
    sample_datasets = {
        "Iris Dataset": "Classic flower classification dataset",
        "Titanic Dataset": "Passenger survival prediction dataset", 
        "Boston Housing": "House price prediction dataset",
        "Wine Quality": "Wine quality classification dataset",
        "Customer Churn": "Telecom customer churn dataset",
        "Stock Market": "Financial time series data",
        "E-commerce Sales": "Sales transaction data",
        "IoT Sensor Data": "Time series sensor readings"
    }
    
    selected_dataset = st.selectbox("Choose Sample Dataset", list(sample_datasets.keys()))
    st.info(sample_datasets[selected_dataset])
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Description:** {sample_datasets[selected_dataset]}")
    with col2:
        if st.button("📥 Load Dataset"):
            df = create_sample_dataset(selected_dataset)
            st.session_state.dm_data = df
            st.session_state.dm_data_history.append({
                'name': selected_dataset,
                'timestamp': datetime.now(),
                'shape': df.shape,
                'source': 'sample_data'
            })
            st.success(f"✅ {selected_dataset} loaded successfully!")
            st.experimental_rerun()

def show_database_connection():
    """Database connection interface"""
    st.subheader("🗄️ Database Connection")
    
    db_type = st.selectbox("Database Type", ["SQLite", "PostgreSQL", "MySQL", "MongoDB"])
    
    if db_type == "SQLite":
        db_file = st.file_uploader("Upload SQLite Database", type=['db', 'sqlite', 'sqlite3'])
        if db_file:
            # Show tables
            st.info("SQLite database connection functionality can be implemented here")
    else:
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host")
            database = st.text_input("Database")
        with col2:
            port = st.number_input("Port", value=5432 if db_type == "PostgreSQL" else 3306)
            username = st.text_input("Username")
        
        password = st.text_input("Password", type="password")
        
        if st.button("🔗 Connect"):
            st.info(f"{db_type} connection functionality can be implemented here")

def show_url_import():
    """URL import interface"""
    st.subheader("🌐 Import from URL")
    
    url = st.text_input("Data URL", placeholder="https://example.com/data.csv")
    
    if url:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**URL:** {url}")
        with col2:
            if st.button("📥 Import"):
                try:
                    if url.endswith('.csv'):
                        df = pd.read_csv(url)
                    elif url.endswith('.json'):
                        df = pd.read_json(url)
                    else:
                        st.error("Unsupported URL format")
                        return
                    
                    st.session_state.dm_data = df
                    st.success("✅ Data imported from URL!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Error importing from URL: {str(e)}")

def show_manual_entry():
    """Manual data entry interface"""
    st.subheader("✏️ Manual Data Entry")
    
    col1, col2 = st.columns(2)
    with col1:
        num_rows = st.number_input("Number of Rows", min_value=1, max_value=100, value=5)
    with col2:
        num_cols = st.number_input("Number of Columns", min_value=1, max_value=20, value=3)
    
    # Column names
    st.markdown("#### Column Names")
    column_names = []
    cols = st.columns(min(num_cols, 4))
    for i in range(num_cols):
        with cols[i % 4]:
            col_name = st.text_input(f"Column {i+1}", value=f"col_{i+1}", key=f"col_name_{i}")
            column_names.append(col_name)
    
    # Data entry
    st.markdown("#### Data Entry")
    if st.button("📝 Create Data Entry Form"):
        # Create empty dataframe
        df = pd.DataFrame(columns=column_names)
        for i in range(num_rows):
            df.loc[i] = [None] * num_cols
        
        # Display editable dataframe
        edited_df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
        
        if st.button("💾 Save Manual Data"):
            st.session_state.dm_data = edited_df
            st.success("✅ Manual data saved!")
            st.experimental_rerun()

def show_data_explorer():
    """Data exploration interface"""
    st.header("🔍 Data Explorer")
    
    if st.session_state.dm_data is None:
        st.warning("⚠️ No data loaded. Please upload data first.")
        return
    
    df = st.session_state.dm_data
    
    # Data overview
    st.subheader("📋 Data Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col5:
        st.metric("Duplicates", df.duplicated().sum())
    
    # Data types and info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Data preview with filtering
    st.subheader("👀 Data Preview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        show_rows = st.selectbox("Rows to show", [10, 25, 50, 100, 500])
    with col2:
        columns_to_show = st.multiselect("Columns to show", df.columns.tolist(), default=df.columns.tolist()[:10])
    with col3:
        sample_data = st.checkbox("Random Sample", value=False)
    
    # Apply filters
    if columns_to_show:
        display_df = df[columns_to_show]
        if sample_data:
            display_df = display_df.sample(min(show_rows, len(display_df)))
        else:
            display_df = display_df.head(show_rows)
        
        st.dataframe(display_df, use_container_width=True)
    
    # Missing values analysis
    st.subheader("❓ Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    }).sort_values('Missing Count', ascending=False)
    
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if not missing_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(missing_df, use_container_width=True)
        with col2:
            fig = px.bar(missing_df, x='Column', y='Missing Percentage', 
                        title="Missing Values by Column")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("🎉 No missing values found!")

def show_data_preprocessing():
    """Data preprocessing interface"""
    st.header("🛠️ Data Preprocessing")
    
    if st.session_state.dm_data is None:
        st.warning("⚠️ No data loaded. Please upload data first.")
        return
    
    df = st.session_state.dm_data.copy()
    
    # Preprocessing options
    preprocessing_type = st.selectbox(
        "🔧 Preprocessing Operation",
        ["Missing Values", "Data Cleaning", "Feature Engineering", "Data Transformation", "Feature Selection"]
    )
    
    if preprocessing_type == "Missing Values":
        show_missing_values_preprocessing(df)
    elif preprocessing_type == "Data Cleaning":
        show_data_cleaning(df)
    elif preprocessing_type == "Feature Engineering":
        show_feature_engineering(df)
    elif preprocessing_type == "Data Transformation":
        show_data_transformation(df)
    elif preprocessing_type == "Feature Selection":
        show_feature_selection(df)

def show_missing_values_preprocessing(df):
    """Missing values preprocessing"""
    st.subheader("❓ Handle Missing Values")
    
    # Show missing values summary
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0].index.tolist()
    
    if not missing_cols:
        st.success("🎉 No missing values found!")
        return
    
    st.warning(f"Found missing values in {len(missing_cols)} columns")
    
    # Select columns to process
    selected_cols = st.multiselect("Select columns to process", missing_cols, default=missing_cols)
    
    if selected_cols:
        strategy = st.selectbox(
            "Missing Value Strategy",
            ["Drop rows", "Drop columns", "Fill with mean", "Fill with median", "Fill with mode", "Forward fill", "Backward fill", "Custom value"]
        )
        
        if strategy == "Custom value":
            custom_value = st.text_input("Custom fill value")
        
        if st.button("🔄 Apply Missing Value Treatment"):
            processed_df = df.copy()
            
            try:
                if strategy == "Drop rows":
                    processed_df = processed_df.dropna(subset=selected_cols)
                elif strategy == "Drop columns":
                    processed_df = processed_df.drop(columns=selected_cols)
                elif strategy == "Fill with mean":
                    for col in selected_cols:
                        if processed_df[col].dtype in ['int64', 'float64']:
                            processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                elif strategy == "Fill with median":
                    for col in selected_cols:
                        if processed_df[col].dtype in ['int64', 'float64']:
                            processed_df[col].fillna(processed_df[col].median(), inplace=True)
                elif strategy == "Fill with mode":
                    for col in selected_cols:
                        processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                elif strategy == "Forward fill":
                    processed_df[selected_cols] = processed_df[selected_cols].fillna(method='ffill')
                elif strategy == "Backward fill":
                    processed_df[selected_cols] = processed_df[selected_cols].fillna(method='bfill')
                elif strategy == "Custom value":
                    processed_df[selected_cols] = processed_df[selected_cols].fillna(custom_value)
                
                st.session_state.dm_processed_data = processed_df
                st.success(f"✅ Missing values handled using '{strategy}' strategy")
                
                # Show before/after comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Missing Values", df.isnull().sum().sum())
                with col2:
                    st.metric("Processed Missing Values", processed_df.isnull().sum().sum())
                
            except Exception as e:
                st.error(f"❌ Error processing missing values: {str(e)}")

def show_data_cleaning(df):
    """Data cleaning operations"""
    st.subheader("🧹 Data Cleaning")
    
    cleaning_operation = st.selectbox(
        "Cleaning Operation",
        ["Remove Duplicates", "Outlier Detection", "Data Type Conversion", "Text Cleaning", "Date/Time Processing"]
    )
    
    if cleaning_operation == "Remove Duplicates":
        st.write(f"**Found {df.duplicated().sum()} duplicate rows**")
        
        if df.duplicated().sum() > 0:
            subset_cols = st.multiselect("Consider duplicates based on columns", df.columns.tolist(), default=df.columns.tolist())
            keep_option = st.selectbox("Keep which duplicate", ["first", "last", "none"])
            
            if st.button("🗑️ Remove Duplicates"):
                processed_df = df.drop_duplicates(subset=subset_cols, keep=keep_option if keep_option != "none" else False)
                st.session_state.dm_processed_data = processed_df
                st.success(f"✅ Removed {df.shape[0] - processed_df.shape[0]} duplicate rows")
    
    elif cleaning_operation == "Outlier Detection":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select column for outlier detection", numeric_cols)
            method = st.selectbox("Outlier detection method", ["IQR", "Z-Score", "Modified Z-Score"])
            
            if method == "IQR":
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                
                st.write(f"**Found {len(outliers)} outliers using IQR method**")
                st.write(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
                
                if len(outliers) > 0:
                    st.dataframe(outliers[[selected_col]], use_container_width=True)
                    
                    if st.button("🗑️ Remove Outliers"):
                        processed_df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
                        st.session_state.dm_processed_data = processed_df
                        st.success(f"✅ Removed {len(outliers)} outliers")

def show_feature_engineering(df):
    """Feature engineering operations"""
    st.subheader("⚙️ Feature Engineering")
    
    operation = st.selectbox(
        "Feature Engineering Operation",
        ["Create New Features", "Binning/Discretization", "Encoding Categorical Variables", "Date/Time Features"]
    )
    
    if operation == "Create New Features":
        st.markdown("#### Create New Features from Existing Columns")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                feature1 = st.selectbox("First Feature", numeric_cols)
            with col2:
                operation_type = st.selectbox("Operation", ["+", "-", "*", "/", "**"])
            with col3:
                feature2 = st.selectbox("Second Feature", numeric_cols)
            
            new_feature_name = st.text_input("New Feature Name", value=f"{feature1}_{operation_type}_{feature2}")
            
            if st.button("➕ Create Feature"):
                processed_df = df.copy()
                try:
                    if operation_type == "+":
                        processed_df[new_feature_name] = processed_df[feature1] + processed_df[feature2]
                    elif operation_type == "-":
                        processed_df[new_feature_name] = processed_df[feature1] - processed_df[feature2]
                    elif operation_type == "*":
                        processed_df[new_feature_name] = processed_df[feature1] * processed_df[feature2]
                    elif operation_type == "/":
                        processed_df[new_feature_name] = processed_df[feature1] / processed_df[feature2]
                    elif operation_type == "**":
                        processed_df[new_feature_name] = processed_df[feature1] ** processed_df[feature2]
                    
                    st.session_state.dm_processed_data = processed_df
                    st.success(f"✅ Created new feature: {new_feature_name}")
                    
                except Exception as e:
                    st.error(f"❌ Error creating feature: {str(e)}")
    
    elif operation == "Encoding Categorical Variables":
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            selected_col = st.selectbox("Select categorical column", categorical_cols)
            encoding_method = st.selectbox("Encoding method", ["Label Encoding", "One-Hot Encoding", "Target Encoding"])
            
            if st.button("🔄 Apply Encoding"):
                processed_df = df.copy()
                
                try:
                    if encoding_method == "Label Encoding":
                        le = LabelEncoder()
                        processed_df[f"{selected_col}_encoded"] = le.fit_transform(processed_df[selected_col].astype(str))
                    
                    elif encoding_method == "One-Hot Encoding":
                        one_hot = pd.get_dummies(processed_df[selected_col], prefix=selected_col)
                        processed_df = pd.concat([processed_df, one_hot], axis=1)
                    
                    st.session_state.dm_processed_data = processed_df
                    st.success(f"✅ Applied {encoding_method} to {selected_col}")
                    
                except Exception as e:
                    st.error(f"❌ Error encoding: {str(e)}")

def show_data_transformation(df):
    """Data transformation operations"""
    st.subheader("🔄 Data Transformation")
    
    transformation_type = st.selectbox(
        "Transformation Type",
        ["Scaling", "Normalization", "Log Transform", "Square Root Transform", "Box-Cox Transform"]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found for transformation")
        return
    
    selected_cols = st.multiselect("Select columns to transform", numeric_cols, default=numeric_cols[:3])
    
    if selected_cols and st.button("🔄 Apply Transformation"):
        processed_df = df.copy()
        
        try:
            if transformation_type == "Scaling":
                scaler = StandardScaler()
                processed_df[selected_cols] = scaler.fit_transform(processed_df[selected_cols])
                
            elif transformation_type == "Normalization":
                scaler = MinMaxScaler()
                processed_df[selected_cols] = scaler.fit_transform(processed_df[selected_cols])
                
            elif transformation_type == "Log Transform":
                for col in selected_cols:
                    processed_df[f"{col}_log"] = np.log1p(processed_df[col])
                    
            elif transformation_type == "Square Root Transform":
                for col in selected_cols:
                    processed_df[f"{col}_sqrt"] = np.sqrt(np.abs(processed_df[col]))
            
            st.session_state.dm_processed_data = processed_df
            st.success(f"✅ Applied {transformation_type} transformation")
            
        except Exception as e:
            st.error(f"❌ Error applying transformation: {str(e)}")

def show_feature_selection(df):
    """Feature selection operations"""
    st.subheader("🎯 Feature Selection")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for feature selection")
        return
    
    target_col = st.selectbox("Select target column", numeric_cols)
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    if feature_cols:
        method = st.selectbox("Feature selection method", ["SelectKBest", "Correlation Threshold"])
        
        if method == "SelectKBest":
            k_features = st.slider("Number of features to select", 1, len(feature_cols), min(5, len(feature_cols)))
            
            if st.button("🎯 Select Features"):
                X = df[feature_cols].fillna(0)
                y = df[target_col].fillna(0)
                
                selector = SelectKBest(f_classif, k=k_features)
                X_selected = selector.fit_transform(X, y)
                
                selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                
                st.success(f"✅ Selected {len(selected_features)} features:")
                st.write(selected_features)
                
                # Create processed dataframe with selected features
                processed_df = df[[target_col] + selected_features].copy()
                st.session_state.dm_processed_data = processed_df

def show_data_visualization():
    """Data visualization interface"""
    st.header("📈 Data Visualization")
    
    if st.session_state.dm_data is None:
        st.warning("⚠️ No data loaded. Please upload data first.")
        return
    
    df = st.session_state.dm_processed_data if st.session_state.dm_processed_data is not None else st.session_state.dm_data
    
    # Visualization type selection
    viz_type = st.selectbox(
        "📊 Visualization Type",
        ["Distribution Plots", "Correlation Analysis", "Comparison Plots", "Time Series", "Advanced Plots"]
    )
    
    if viz_type == "Distribution Plots":
        show_distribution_plots(df)
    elif viz_type == "Correlation Analysis":
        show_correlation_analysis(df)
    elif viz_type == "Comparison Plots":
        show_comparison_plots(df)
    elif viz_type == "Time Series":
        show_time_series_plots(df)
    elif viz_type == "Advanced Plots":
        show_advanced_plots(df)

def show_distribution_plots(df):
    """Distribution visualization"""
    st.subheader("📊 Distribution Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols:
        selected_col = st.selectbox("Select column", numeric_cols + categorical_cols)
        
        if selected_col in numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Categorical distribution
            value_counts = df[selected_col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(df):
    """Correlation analysis visualization"""
    st.subheader("🔗 Correlation Analysis")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        # Correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Correlation Heatmap",
                       color_continuous_scale="RdBu",
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Strong correlations
        st.subheader("🎯 Strong Correlations")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_value
                    })
        
        if strong_corr:
            st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
        else:
            st.info("No strong correlations (>0.7) found")

def show_comparison_plots(df):
    """Comparison plots"""
    st.subheader("📊 Comparison Analysis")
    
    plot_type = st.selectbox("Plot Type", ["Scatter Plot", "Bar Chart", "Group Comparison"])
    
    if plot_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols)
            with col2:
                y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col])
            with col3:
                color_col = st.selectbox("Color by", ["None"] + df.columns.tolist())
            
            if color_col == "None":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
            
            st.plotly_chart(fig, use_container_width=True)

def show_time_series_plots(df):
    """Time series visualization"""
    st.subheader("📈 Time Series Analysis")
    
    # Detect datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to parse potential date columns
    potential_date_cols = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            potential_date_cols.append(col)
    
    if not datetime_cols and potential_date_cols:
        st.info("No datetime columns detected. Try converting text columns to datetime.")
        
        date_col = st.selectbox("Select date column to convert", potential_date_cols)
        date_format = st.text_input("Date format (optional)", placeholder="e.g., %Y-%m-%d")
        
        if st.button("Convert to DateTime"):
            try:
                if date_format:
                    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
                else:
                    df[date_col] = pd.to_datetime(df[date_col])
                st.success(f"✅ Converted {date_col} to datetime")
                datetime_cols = [date_col]
            except Exception as e:
                st.error(f"❌ Error converting to datetime: {str(e)}")
    
    if datetime_cols:
        date_col = st.selectbox("Select date column", datetime_cols)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            value_col = st.selectbox("Select value column", numeric_cols)
            
            # Sort by date
            df_sorted = df.sort_values(date_col)
            
            fig = px.line(df_sorted, x=date_col, y=value_col, 
                         title=f"Time Series: {value_col}")
            st.plotly_chart(fig, use_container_width=True)

def show_advanced_plots(df):
    """Advanced visualization options"""
    st.subheader("🎨 Advanced Plots")
    
    plot_type = st.selectbox(
        "Advanced Plot Type", 
        ["Pair Plot", "3D Scatter", "Parallel Coordinates", "Radar Chart"]
    )
    
    if plot_type == "3D Scatter":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 3:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="3d_x")
            with col2:
                y_col = st.selectbox("Y-axis", numeric_cols, key="3d_y")
            with col3:
                z_col = st.selectbox("Z-axis", numeric_cols, key="3d_z")
            with col4:
                color_col = st.selectbox("Color by", ["None"] + df.columns.tolist(), key="3d_color")
            
            if color_col == "None":
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col)
            else:
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col)
            
            st.plotly_chart(fig, use_container_width=True)

def show_data_storage():
    """Data storage and export interface"""
    st.header("💾 Data Storage & Export")
    
    if st.session_state.dm_processed_data is not None:
        df = st.session_state.dm_processed_data
        st.success("Using processed data")
    elif st.session_state.dm_data is not None:
        df = st.session_state.dm_data
        st.info("Using original data")
    else:
        st.warning("⚠️ No data available for storage")
        return
    
    # Data export options
    st.subheader("📤 Export Data")
    
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON", "Parquet", "Pickle"])
    
    col1, col2 = st.columns(2)
    with col1:
        filename = st.text_input("Filename", value=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    with col2:
        include_index = st.checkbox("Include Index", value=False)
    
    # Generate download
    if st.button("📥 Generate Download"):
        try:
            if export_format == "CSV":
                output = df.to_csv(index=include_index)
                mime_type = "text/csv"
                file_ext = ".csv"
            elif export_format == "Excel":
                output = io.BytesIO()
                df.to_excel(output, index=include_index, engine='openpyxl')
                output.seek(0)
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                file_ext = ".xlsx"
            elif export_format == "JSON":
                output = df.to_json(orient='records', indent=2)
                mime_type = "application/json"
                file_ext = ".json"
            elif export_format == "Parquet":
                output = io.BytesIO()
                df.to_parquet(output, index=include_index)
                output.seek(0)
                mime_type = "application/octet-stream"
                file_ext = ".parquet"
            elif export_format == "Pickle":
                output = io.BytesIO()
                pickle.dump(df, output)
                output.seek(0)
                mime_type = "application/octet-stream"
                file_ext = ".pkl"
            
            st.download_button(
                label=f"📥 Download {export_format}",
                data=output,
                file_name=f"{filename}{file_ext}",
                mime=mime_type
            )
            
        except Exception as e:
            st.error(f"❌ Error generating download: {str(e)}")
    
    # Save to local storage
    st.subheader("💾 Save to Local Storage")
    
    local_dir = st.text_input("Storage Directory", value="data/processed")
    
    if st.button("💾 Save Locally"):
        try:
            os.makedirs(local_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(local_dir, f"data_{timestamp}.csv")
            df.to_csv(filepath, index=False)
            st.success(f"✅ Data saved to {filepath}")
        except Exception as e:
            st.error(f"❌ Error saving locally: {str(e)}")

def show_data_catalog():
    """Data catalog and history interface"""
    st.header("📋 Data Catalog")
    
    # Data history
    st.subheader("📚 Data History")
    
    if st.session_state.dm_data_history:
        history_df = pd.DataFrame(st.session_state.dm_data_history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.dm_data_history = []
            st.success("✅ History cleared")
            st.experimental_rerun()
    else:
        st.info("No data history available")
    
    # Current data status
    st.subheader("📊 Current Data Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Data:**")
        if st.session_state.dm_data is not None:
            df = st.session_state.dm_data
            st.write(f"- Shape: {df.shape}")
            st.write(f"- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.write(f"- Missing values: {df.isnull().sum().sum()}")
        else:
            st.write("No original data loaded")
    
    with col2:
        st.markdown("**Processed Data:**")
        if st.session_state.dm_processed_data is not None:
            df = st.session_state.dm_processed_data
            st.write(f"- Shape: {df.shape}")
            st.write(f"- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.write(f"- Missing values: {df.isnull().sum().sum()}")
        else:
            st.write("No processed data available")
    
    # Data quality report
    st.subheader("📋 Data Quality Report")
    
    if st.session_state.dm_data is not None:
        df = st.session_state.dm_data
        
        quality_report = {
            'Metric': [
                'Total Rows',
                'Total Columns', 
                'Missing Values',
                'Duplicate Rows',
                'Numeric Columns',
                'Categorical Columns',
                'Memory Usage (MB)'
            ],
            'Value': [
                df.shape[0],
                df.shape[1],
                df.isnull().sum().sum(),
                df.duplicated().sum(),
                len(df.select_dtypes(include=[np.number]).columns),
                len(df.select_dtypes(include=['object']).columns),
                f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(quality_report), use_container_width=True)

def create_sample_dataset(dataset_name):
    """Create sample datasets for demonstration"""
    np.random.seed(42)
    
    if dataset_name == "Iris Dataset":
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        return df
    
    elif dataset_name == "Titanic Dataset":
        n_samples = 891
        df = pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Age': np.random.normal(29.7, 14.5, n_samples),
            'SibSp': np.random.poisson(0.52, n_samples),
            'Parch': np.random.poisson(0.38, n_samples),
            'Fare': np.random.exponential(32.2, n_samples),
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
        })
        return df
    
    elif dataset_name == "Customer Churn":
        n_samples = 1000
        df = pd.DataFrame({
            'CustomerID': range(1, n_samples + 1),
            'Age': np.random.randint(18, 80, n_samples),
            'Tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(100, 8000, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        })
        return df
    
    # Add more sample datasets as needed
    else:
        # Default dataset
        n_samples = 1000
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = np.random.choice([0, 1], n_samples)
        return df

if __name__ == "__main__":
    show_data_management_page()
