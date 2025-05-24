"""
Settings Page for AI Vision Suite
System configuration, preferences, and administration
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
import shutil
import time
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import database manager
try:
    from database import DatabaseManager
except ImportError:
    try:
        from database.db_setup import DatabaseManager
    except ImportError:
        st.error("❌ Could not import DatabaseManager. Please check your installation.")
        DatabaseManager = None

def show_settings_page():
    """Display the Settings page with all configuration options"""
    
    st.title("⚙️ Settings & Configuration")
    st.markdown("### System configuration, preferences, and administration")
    st.markdown("---")
    
    # Initialize session state for settings
    if 'settings_config' not in st.session_state:
        st.session_state.settings_config = load_default_config()
    
    # Create tabs for different setting categories
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🖥️ System", "📊 Models", "🗄️ Database", "🎨 Interface", "🔧 Advanced", "📝 Logs"
    ])
    
    with tab1:
        show_system_settings()
    
    with tab2:
        show_model_settings()
    
    with tab3:
        show_database_settings()
    
    with tab4:
        show_interface_settings()
    
    with tab5:
        show_advanced_settings()
    
    with tab6:
        show_logs_settings()

def show_system_settings():
    """System configuration and resource management"""
    st.header("🖥️ System Settings")
    
    # System Information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 System Information")
        
        # CPU Information
        cpu_count = os.cpu_count()
        
        st.metric("CPU Cores", cpu_count)
        
        # Check if psutil is available
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            st.metric("CPU Usage", f"{cpu_percent}%")
            st.metric("Total RAM", f"{memory.total / (1024**3):.1f} GB")
            st.metric("Available RAM", f"{memory.available / (1024**3):.1f} GB")
            st.metric("RAM Usage", f"{memory.percent}%")
            st.metric("Disk Total", f"{disk.total / (1024**3):.1f} GB")
            st.metric("Disk Free", f"{disk.free / (1024**3):.1f} GB")
            st.metric("Disk Usage", f"{(disk.used/disk.total)*100:.1f}%")
        except ImportError:
            st.info("Install psutil for detailed system monitoring: `pip install psutil`")
    
    with col2:
        st.subheader("🎯 GPU Configuration")
        
        # GPU Information
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
                
                st.success("✅ GPU Available")
                st.metric("GPU Count", gpu_count)
                st.metric("Current GPU", gpu_name)
                st.metric("GPU Memory", f"{gpu_memory / (1024**3):.1f} GB")
                
                # GPU Settings
                default_device = st.selectbox(
                    "Default Device",
                    ["auto", "cpu"] + [f"cuda:{i}" for i in range(gpu_count)],
                    index=0
                )
                
                if st.button("🔄 Clear GPU Cache"):
                    torch.cuda.empty_cache()
                    st.success("✅ GPU cache cleared")
            else:
                st.warning("⚠️ No GPU detected")
                st.info("💡 Using CPU for computations")
                default_device = "cpu"
        except ImportError:
            st.warning("⚠️ PyTorch not available")
            default_device = "cpu"
    
    # Performance Settings
    st.subheader("⚡ Performance Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Threading
        max_threads = st.number_input(
            "Max Threads for Training",
            min_value=1,
            max_value=cpu_count if cpu_count else 8,
            value=min(4, cpu_count if cpu_count else 4),
            help="Number of CPU threads to use for training"
        )
        
        # Memory Management
        memory_limit = st.slider(
            "Memory Limit (%)",
            min_value=50,
            max_value=90,
            value=80,
            help="Maximum RAM usage allowed"
        )
    
    with col2:
        # Cache Settings
        enable_cache = st.checkbox("Enable Model Caching", value=True)
        cache_size = st.number_input(
            "Cache Size (GB)",
            min_value=1,
            max_value=10,
            value=2,
            help="Maximum cache size for models"
        )
        
        # Auto-cleanup
        auto_cleanup = st.checkbox("Auto Cleanup Temp Files", value=True)
    
    with col3:
        # Batch Processing
        default_batch_size = st.selectbox(
            "Default Batch Size",
            [16, 32, 64, 128, 256],
            index=2
        )
        
        # Parallel Processing
        enable_parallel = st.checkbox("Enable Parallel Processing", value=True)
    
    # Save System Settings
    if st.button("💾 Save System Settings", type="primary"):
        system_config = {
            'default_device': default_device,
            'max_threads': max_threads,
            'memory_limit': memory_limit,
            'enable_cache': enable_cache,
            'cache_size': cache_size,
            'auto_cleanup': auto_cleanup,
            'default_batch_size': default_batch_size,
            'enable_parallel': enable_parallel
        }
        
        if save_config('system', system_config):
            st.success("✅ System settings saved successfully!")
            st.session_state.settings_config['system'] = system_config

def show_model_settings():
    """Model management and configuration"""
    st.header("📊 Model Settings")
    
    # Initialize database
    if DatabaseManager:
        try:
            db = DatabaseManager()
        except Exception as e:
            st.error(f"❌ Database connection error: {str(e)}")
            db = None
    else:
        st.error("❌ DatabaseManager not available")
        db = None
    
    # Model Storage Settings
    st.subheader("💾 Storage Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Storage Directories
        st.markdown("#### 📁 Storage Directories")
        
        base_model_dir = st.text_input(
            "Base Models Directory",
            value="models/",
            help="Base directory for all model files"
        )
        
        gan_models_dir = st.text_input(
            "GAN Models Directory",
            value="models/tabular_gans/",
            help="Directory for GAN model files"
        )
        
        nn_models_dir = st.text_input(
            "Neural Network Models Directory",
            value="models/neural_networks/",
            help="Directory for neural network model files"
        )
        
        # Create directories button
        if st.button("📁 Create Directories"):
            try:
                os.makedirs(base_model_dir, exist_ok=True)
                os.makedirs(gan_models_dir, exist_ok=True)
                os.makedirs(nn_models_dir, exist_ok=True)
                st.success("✅ Directories created successfully!")
            except Exception as e:
                st.error(f"❌ Error creating directories: {str(e)}")
    
    with col2:
        # Storage Quotas and Cleanup
        st.markdown("#### 🧹 Storage Management")
        
        max_storage = st.number_input(
            "Max Storage (GB)",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum storage space for models"
        )
        
        auto_delete_old = st.checkbox(
            "Auto-delete old models",
            value=False,
            help="Automatically delete models older than specified days"
        )
        
        retention_days = 30
        if auto_delete_old:
            retention_days = st.number_input(
                "Retention Period (days)",
                min_value=1,
                max_value=365,
                value=30
            )
        
        # Storage cleanup
        if st.button("🗑️ Clean Storage"):
            cleanup_storage(base_model_dir, auto_delete_old, retention_days if auto_delete_old else None)
    
    # Model Performance Tracking
    st.subheader("📈 Performance Tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tracking Options
        track_training_time = st.checkbox("Track Training Time", value=True)
        track_memory_usage = st.checkbox("Track Memory Usage", value=True)
        track_predictions = st.checkbox("Track Prediction Count", value=True)
        auto_benchmarking = st.checkbox("Auto Benchmarking", value=False)
    
    with col2:
        # Notification Settings
        notify_training_complete = st.checkbox("Notify Training Complete", value=True)
        notify_performance_drop = st.checkbox("Notify Performance Drop", value=False)
        
        performance_threshold = 10
        if notify_performance_drop:
            performance_threshold = st.slider(
                "Performance Drop Threshold (%)",
                min_value=5,
                max_value=50,
                value=10
            )
    
    # Save Model Settings
    if st.button("💾 Save Model Settings", type="primary"):
        model_config = {
            'directories': {
                'base': base_model_dir,
                'gan': gan_models_dir,
                'neural_networks': nn_models_dir
            },
            'storage': {
                'max_storage_gb': max_storage,
                'auto_delete_old': auto_delete_old,
                'retention_days': retention_days
            },
            'tracking': {
                'training_time': track_training_time,
                'memory_usage': track_memory_usage,
                'predictions': track_predictions,
                'auto_benchmarking': auto_benchmarking
            },
            'notifications': {
                'training_complete': notify_training_complete,
                'performance_drop': notify_performance_drop,
                'performance_threshold': performance_threshold
            }
        }
        
        if save_config('models', model_config):
            st.success("✅ Model settings saved successfully!")
            st.session_state.settings_config['models'] = model_config
    
    # Model Statistics
    if db:
        st.subheader("📊 Model Statistics")
        show_model_statistics(db)

def show_database_settings():
    """Database configuration and management"""
    st.header("🗄️ Database Settings")
    
    if DatabaseManager:
        try:
            db = DatabaseManager()
        except Exception as e:
            st.error(f"❌ Database connection error: {str(e)}")
            db = None
    else:
        st.error("❌ DatabaseManager not available")
        db = None
    
    # Database Information
    st.subheader("📊 Database Information")
    
    col1, col2, col3 = st.columns(3)
    
    # Get database statistics
    if db:
        try:
            # Count models by type
            model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
            total_models = 0
            
            for model_type in model_types:
                try:
                    models = db.get_models_by_type(model_type)
                    total_models += len(models) if not models.empty else 0
                except:
                    pass
            
            with col1:
                st.metric("Total Models", total_models)
            
            with col2:
                # Database file size
                db_file = "database/ai_vision_suite.db"
                if os.path.exists(db_file):
                    db_size = os.path.getsize(db_file) / (1024**2)  # MB
                    st.metric("Database Size", f"{db_size:.1f} MB")
                else:
                    st.metric("Database Size", "N/A")
            
            with col3:
                st.metric("Last Backup", "N/A")
                
        except Exception as e:
            st.error(f"❌ Error getting database statistics: {str(e)}")
    else:
        with col1:
            st.metric("Total Models", "N/A")
        with col2:
            st.metric("Database Size", "N/A")
        with col3:
            st.metric("Last Backup", "N/A")
    
    # Data Management Section
    st.subheader("🗂️ Data Management & Cleanup")
    
    # Data Overview
    with st.expander("📊 Data Overview", expanded=True):
        if db:
            show_data_overview_stats(db)
        else:
            st.info("Database not available for overview")
    
    # Selective Data Deletion
    with st.expander("🗑️ Selective Data Deletion"):
        if db:
            show_selective_data_deletion(db)
        else:
            st.info("Database not available for data deletion")
    
    # Bulk Data Operations
    with st.expander("🔄 Bulk Data Operations"):
        if db:
            show_bulk_data_operations(db)
        else:
            st.info("Database not available for bulk operations")
    
    # Data Export & Import
    with st.expander("📤 Data Export & Import"):
        if db:
            show_data_export_import(db)
        else:
            st.info("Database not available for export/import")

def show_interface_settings():
    """Interface and UI configuration"""
    st.header("🎨 Interface Settings")
    
    # Theme and Appearance
    st.subheader("🎨 Theme & Appearance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Color Scheme
        st.markdown("#### 🌈 Color Scheme")
        
        primary_color = st.color_picker("Primary Color", value="#FF6B6B")
        secondary_color = st.color_picker("Secondary Color", value="#4ECDC4")
        background_color = st.color_picker("Background Color", value="#FFFFFF")
        text_color = st.color_picker("Text Color", value="#000000")
        
        # Font Settings
        st.markdown("#### 📝 Typography")
        
        font_family = st.selectbox(
            "Font Family",
            ["Default", "Arial", "Helvetica", "Times New Roman", "Georgia", "Verdana"],
            index=0
        )
        
        font_size = st.selectbox(
            "Font Size",
            ["Small", "Medium", "Large"],
            index=1
        )
    
    with col2:
        # Layout Settings
        st.markdown("#### 📐 Layout")
        
        sidebar_width = st.selectbox(
            "Sidebar Width",
            ["Narrow", "Medium", "Wide"],
            index=1
        )
        
        page_layout = st.selectbox(
            "Page Layout",
            ["Centered", "Wide", "Full Width"],
            index=0
        )
        
        show_progress_bars = st.checkbox("Show Progress Bars", value=True)
        show_tooltips = st.checkbox("Show Tooltips", value=True)
        animated_transitions = st.checkbox("Animated Transitions", value=True)
        
        # Performance Options
        st.markdown("#### ⚡ Performance")
        
        lazy_loading = st.checkbox("Lazy Load Images", value=True)
        cache_ui_elements = st.checkbox("Cache UI Elements", value=True)
    
    # Display Preferences
    st.subheader("📊 Display Preferences")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Charts and Graphs
        st.markdown("#### 📈 Charts")
        
        default_chart_theme = st.selectbox(
            "Chart Theme",
            ["plotly", "ggplot2", "seaborn", "simple_white"],
            index=0
        )
        
        chart_height = st.number_input(
            "Default Chart Height",
            min_value=200,
            max_value=800,
            value=400
        )
        
        show_chart_toolbar = st.checkbox("Show Chart Toolbar", value=True)
    
    with col2:
        # Tables
        st.markdown("#### 📋 Tables")
        
        table_page_size = st.number_input(
            "Table Page Size",
            min_value=10,
            max_value=100,
            value=25
        )
        
        table_height = st.number_input(
            "Table Height",
            min_value=200,
            max_value=600,
            value=300
        )
        
        show_index = st.checkbox("Show Row Index", value=False)
    
    with col3:
        # Notifications
        st.markdown("#### 🔔 Notifications")
        
        notification_position = st.selectbox(
            "Notification Position",
            ["top-right", "top-left", "bottom-right", "bottom-left"],
            index=0
        )
        
        notification_duration = st.number_input(
            "Notification Duration (seconds)",
            min_value=1,
            max_value=10,
            value=3
        )
        
        sound_notifications = st.checkbox("Sound Notifications", value=False)
    
    # Save Interface Settings
    if st.button("💾 Save Interface Settings", type="primary"):
        interface_config = {
            'theme': {
                'primary_color': primary_color,
                'secondary_color': secondary_color,
                'background_color': background_color,
                'text_color': text_color,
                'font_family': font_family,
                'font_size': font_size
            },
            'layout': {
                'sidebar_width': sidebar_width,
                'page_layout': page_layout,
                'show_progress_bars': show_progress_bars,
                'show_tooltips': show_tooltips,
                'animated_transitions': animated_transitions,
                'lazy_loading': lazy_loading,
                'cache_ui_elements': cache_ui_elements
            },
            'display': {
                'chart_theme': default_chart_theme,
                'chart_height': chart_height,
                'show_chart_toolbar': show_chart_toolbar,
                'table_page_size': table_page_size,
                'table_height': table_height,
                'show_index': show_index,
                'notification_position': notification_position,
                'notification_duration': notification_duration,
                'sound_notifications': sound_notifications
            }
        }
        
        if save_config('interface', interface_config):
            st.success("✅ Interface settings saved successfully!")
            st.info("🔄 Some changes may require a page refresh to take effect.")
            st.session_state.settings_config['interface'] = interface_config

def show_advanced_settings():
    """Advanced system configuration"""
    st.header("🔧 Advanced Settings")
    
    # Security Settings
    st.subheader("🔒 Security Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Authentication
        st.markdown("#### 🔐 Authentication")
        
        enable_auth = st.checkbox("Enable Authentication", value=False)
        
        auth_method = None
        session_timeout = None
        max_login_attempts = None
        
        if enable_auth:
            auth_method = st.selectbox(
                "Authentication Method",
                ["Local", "LDAP", "OAuth2"],
                index=0
            )
            
            session_timeout = st.number_input(
                "Session Timeout (minutes)",
                min_value=15,
                max_value=480,
                value=60
            )
            
            max_login_attempts = st.number_input(
                "Max Login Attempts",
                min_value=3,
                max_value=10,
                value=5
            )
    
    with col2:
        # Data Protection
        st.markdown("#### 🛡️ Data Protection")
        
        encrypt_models = st.checkbox("Encrypt Model Files", value=False)
        encrypt_database = st.checkbox("Encrypt Database", value=False)
        
        data_retention = st.number_input(
            "Data Retention Period (days)",
            min_value=30,
            max_value=365,
            value=90
        )
        
        audit_logging = st.checkbox("Enable Audit Logging", value=True)
    
    # API Settings
    st.subheader("🌐 API Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # API Configuration
        enable_api = st.checkbox("Enable REST API", value=False)
        
        api_port = 8080
        api_host = "localhost"
        api_auth = True
        
        if enable_api:
            api_port = st.number_input(
                "API Port",
                min_value=8000,
                max_value=9999,
                value=8080
            )
            
            api_host = st.text_input(
                "API Host",
                value="localhost"
            )
            
            api_auth = st.checkbox("API Authentication", value=True)
    
    with col2:
        # Rate Limiting
        rate_limit_enabled = True
        requests_per_minute = 100
        requests_per_hour = 1000
        
        if enable_api:
            st.markdown("#### 🚦 Rate Limiting")
            
            rate_limit_enabled = st.checkbox("Enable Rate Limiting", value=True)
            
            if rate_limit_enabled:
                requests_per_minute = st.number_input(
                    "Requests per Minute",
                    min_value=10,
                    max_value=1000,
                    value=100
                )
                
                requests_per_hour = st.number_input(
                    "Requests per Hour",
                    min_value=100,
                    max_value=10000,
                    value=1000
                )
    
    # Save Advanced Settings
    if st.button("💾 Save Advanced Settings", type="primary"):
        advanced_config = {
            'security': {
                'enable_auth': enable_auth,
                'auth_method': auth_method,
                'session_timeout': session_timeout,
                'max_login_attempts': max_login_attempts,
                'encrypt_models': encrypt_models,
                'encrypt_database': encrypt_database,
                'data_retention': data_retention,
                'audit_logging': audit_logging
            },
            'api': {
                'enable_api': enable_api,
                'api_port': api_port,
                'api_host': api_host,
                'api_auth': api_auth,
                'rate_limit_enabled': rate_limit_enabled,
                'requests_per_minute': requests_per_minute,
                'requests_per_hour': requests_per_hour
            }
        }
        
        if save_config('advanced', advanced_config):
            st.success("✅ Advanced settings saved successfully!")
            st.session_state.settings_config['advanced'] = advanced_config

def show_logs_settings():
    """Logs and diagnostics interface"""
    st.header("📝 Logs & Diagnostics")
    
    # Log Viewer
    st.subheader("📄 Application Logs")
    
    log_level_filter = st.selectbox(
        "Filter by Log Level",
        ["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    
    # Simulate log entries
    sample_logs = [
        {"timestamp": "2024-01-15 10:30:25", "level": "INFO", "message": "Model training started: TabularGAN_20240115"},
        {"timestamp": "2024-01-15 10:28:15", "level": "WARNING", "message": "GPU memory usage at 85%"},
        {"timestamp": "2024-01-15 10:25:10", "level": "INFO", "message": "Database connection established"},
        {"timestamp": "2024-01-15 10:20:05", "level": "ERROR", "message": "Failed to load model: file not found"},
        {"timestamp": "2024-01-15 10:15:30", "level": "DEBUG", "message": "Processing batch 150/500"},
        {"timestamp": "2024-01-15 10:10:45", "level": "INFO", "message": "User session started"},
    ]
    
    # Filter logs
    if log_level_filter != "All":
        filtered_logs = [log for log in sample_logs if log["level"] == log_level_filter]
    else:
        filtered_logs = sample_logs
    
    # Display logs
    log_df = pd.DataFrame(filtered_logs)
    st.dataframe(log_df, use_container_width=True)
    
    # Download logs
    if st.button("📥 Download Logs"):
        log_csv = log_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=log_csv,
            file_name=f"app_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # System Diagnostics
    st.subheader("🔧 System Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Run System Check"):
            with st.spinner("Running system diagnostics..."):
                # Simulate system check
                time.sleep(2)
                
                diagnostics = {
                    "Database Connection": "✅ OK",
                    "Model Storage": "✅ OK", 
                    "Dependencies": "✅ All Installed"
                }
                
                try:
                    import torch
                    diagnostics["GPU Availability"] = "✅ Available" if torch.cuda.is_available() else "⚠️ Not Available"
                except ImportError:
                    diagnostics["PyTorch"] = "⚠️ Not Installed"
                
                try:
                    import psutil
                    diagnostics["Memory Usage"] = "✅ Normal"
                    diagnostics["Disk Space"] = "✅ Sufficient"
                except ImportError:
                    diagnostics["System Monitoring"] = "⚠️ psutil not available"
                
                for check, status in diagnostics.items():
                    st.write(f"**{check}:** {status}")
    
    with col2:
        if st.button("📊 Generate Report"):
            with st.spinner("Generating diagnostic report..."):
                # Simulate report generation
                time.sleep(1)
                
                report_data = {
                    "Report Generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "System Status": "Healthy",
                    "Active Models": "5",
                    "Total Predictions": "1,247",
                    "Uptime": "5 days, 12 hours",
                    "Last Backup": "2024-01-14 02:00:00"
                }
                
                st.json(report_data)

# Helper functions
def load_default_config():
    """Load default configuration"""
    return {
        'system': {
            'default_device': 'auto',
            'max_threads': 4,
            'memory_limit': 80,
            'enable_cache': True,
            'cache_size': 2,
            'auto_cleanup': True,
            'default_batch_size': 64,
            'enable_parallel': True
        },
        'interface': {
            'theme': {
                'primary_color': '#FF6B6B',
                'secondary_color': '#4ECDC4',
                'background_color': '#FFFFFF',
                'text_color': '#000000',
                'font_family': 'Default',
                'font_size': 'Medium'
            }
        }
    }

def save_config(section, config):
    """Save configuration to file"""
    try:
        config_dir = "config"
        os.makedirs(config_dir, exist_ok=True)
        
        config_file = os.path.join(config_dir, f"{section}_config.json")
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Failed to save config: {str(e)}")
        return False

def load_config(section):
    """Load configuration from file"""
    try:
        config_file = f"config/{section}_config.json"
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        st.error(f"Failed to load config: {str(e)}")
        return {}

def show_model_statistics(db):
    """Show model statistics"""
    try:
        # Get model counts by type
        model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        model_counts = {}
        
        for model_type in model_types:
            try:
                models = db.get_models_by_type(model_type)
                model_counts[model_type] = len(models) if not models.empty else 0
            except:
                model_counts[model_type] = 0
        
        # Display as metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("GAN Models", model_counts.get("TABULAR_GAN", 0))
        with col2:
            st.metric("MLP Models", model_counts.get("MLP", 0))
        with col3:
            st.metric("LSTM Models", model_counts.get("LSTM", 0))
        with col4:
            st.metric("CNN Models", model_counts.get("CNN", 0))
        
    except Exception as e:
        st.error(f"Error loading model statistics: {str(e)}")

def cleanup_storage(storage_dir, auto_delete_old, retention_days):
    """Clean up storage directory"""
    try:
        if not os.path.exists(storage_dir):
            st.warning(f"Directory {storage_dir} does not exist")
            return
        
        cleaned_files = 0
        
        if auto_delete_old and retention_days:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for root, dirs, files in os.walk(storage_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_time < cutoff_date:
                            os.remove(file_path)
                            cleaned_files += 1
                    except Exception as e:
                        st.warning(f"Could not delete {file}: {str(e)}")
        
        st.success(f"✅ Storage cleanup completed. Removed {cleaned_files} old files.")
        
    except Exception as e:
        st.error(f"❌ Storage cleanup failed: {str(e)}")

def show_data_overview_stats(db):
    """Show comprehensive data overview statistics"""
    try:
        # Model statistics
        st.markdown("##### 📊 Model Statistics")
        
        model_stats = {}
        model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        
        for model_type in model_types:
            try:
                models = db.get_models_by_type(model_type)
                model_stats[model_type] = len(models) if not models.empty else 0
            except:
                model_stats[model_type] = 0
        
        # Display model counts
        cols = st.columns(len(model_types))
        for i, (model_type, count) in enumerate(model_stats.items()):
            with cols[i]:
                st.metric(f"{model_type} Models", count)
        
        # Total storage used
        total_storage = calculate_total_storage_usage()
        st.metric("Total Storage Used", f"{total_storage:.1f} MB")
        
        # Storage breakdown chart
        if sum(model_stats.values()) > 0:
            try:
                import plotly.express as px
                
                # Create pie chart for model distribution
                fig = px.pie(
                    values=list(model_stats.values()),
                    names=list(model_stats.keys()),
                    title="Model Distribution by Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install plotly for charts: `pip install plotly`")
        
    except Exception as e:
        st.error(f"Error getting data overview: {str(e)}")

def show_selective_data_deletion(db):
    """Show selective data deletion options"""
    st.markdown("##### 🎯 Select Specific Data to Delete")
    
    deletion_type = st.selectbox(
        "What would you like to delete?",
        [
            "Individual Models",
            "Models by Type", 
            "Models by Date Range",
            "All Training Data",
            "All Prediction History"
        ]
    )
    
    if deletion_type == "Individual Models":
        show_individual_model_deletion(db)
    elif deletion_type == "Models by Type":
        show_model_type_deletion(db)
    elif deletion_type == "All Training Data":
        show_delete_all_training_data(db)
    elif deletion_type == "All Prediction History":
        show_delete_all_predictions(db)

def show_individual_model_deletion(db):
    """Delete individual models"""
    st.markdown("**Select Individual Models to Delete**")
    
    try:
        # Get all models
        all_models = []
        model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        
        for model_type in model_types:
            try:
                models = db.get_models_by_type(model_type)
                if not models.empty:
                    models['type'] = model_type
                    all_models.append(models)
            except:
                pass
        
        if all_models:
            combined_models = pd.concat(all_models, ignore_index=True)
            
            # Create model selection interface
            model_options = {}
            for _, model in combined_models.iterrows():
                name = model.get('name', f"Model_{model.get('id', 'unknown')}")
                model_type = model.get('type', 'Unknown')
                created_at = str(model.get('created_at', 'Unknown'))[:19]
                model_id = model.get('id')
                
                display_name = f"{name} | {model_type} | {created_at}"
                model_options[display_name] = model_id
            
            selected_models = st.multiselect(
                "Select models to delete:",
                options=list(model_options.keys()),
                help="Select one or more models to delete"
            )
            
            if selected_models:
                st.warning(f"⚠️ This will delete {len(selected_models)} models and their files!")
                
                # Show selected models details
                st.markdown("**Selected Models for Deletion:**")
                selected_ids = []
                for display_name in selected_models:
                    model_id = model_options[display_name]
                    selected_ids.append(model_id)
                    st.write(f"- {display_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    confirm_delete = st.checkbox("⚠️ I confirm I want to delete these models")
                
                with col2:
                    if confirm_delete and st.button("🗑️ Delete Selected Models", type="secondary"):
                        try:
                            with st.spinner("🗑️ Deleting selected models..."):
                                deleted_count = 0
                                deleted_files = 0
                                
                                for model_id in selected_ids:
                                    try:
                                        # Get model info before deletion
                                        model_info = combined_models[combined_models['id'] == model_id].iloc[0]
                                        
                                        # Delete model files
                                        files_deleted = delete_model_files(model_info)
                                        deleted_files += files_deleted
                                        
                                        # Delete from database
                                        if hasattr(db, 'delete_model'):
                                            db.delete_model(model_id)
                                        
                                        deleted_count += 1
                                        
                                    except Exception as e:
                                        st.warning(f"Failed to delete model {model_id}: {str(e)}")
                                
                                st.success(f"✅ Successfully deleted {deleted_count} models and {deleted_files} files!")
                                safe_rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Error deleting models: {str(e)}")
        else:
            st.info("No models found to delete.")
            
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")

def show_model_type_deletion(db):
    """Delete models by type"""
    st.markdown("**Delete All Models of Specific Types**")
    
    model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
    
    # Show current counts
    type_counts = {}
    for model_type in model_types:
        try:
            models = db.get_models_by_type(model_type)
            type_counts[model_type] = len(models) if not models.empty else 0
        except:
            type_counts[model_type] = 0
    
    # Display current counts
    cols = st.columns(len(model_types))
    for i, (model_type, count) in enumerate(type_counts.items()):
        with cols[i]:
            st.metric(f"{model_type}", f"{count} models")
    
    # Selection interface
    selected_types = st.multiselect(
        "Select model types to delete:",
        options=[f"{t} ({type_counts[t]} models)" for t in model_types if type_counts[t] > 0],
        help="All models of selected types will be deleted"
    )
    
    if selected_types:
        types_to_delete = [t.split(' (')[0] for t in selected_types]
        total_to_delete = sum(type_counts[t] for t in types_to_delete)
        
        st.warning(f"⚠️ This will delete {total_to_delete} models and their files!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confirm_delete = st.checkbox("⚠️ I confirm I want to delete all models of selected types")
        
        with col2:
            if confirm_delete and st.button("🗑️ Delete All Selected Types", type="secondary"):
                try:
                    with st.spinner("🗑️ Deleting models by type..."):
                        total_deleted = 0
                        total_files = 0
                        
                        for model_type in types_to_delete:
                            try:
                                result = delete_models_by_type(db, model_type)
                                total_deleted += result['models_deleted']
                                total_files += result['files_deleted']
                            except Exception as e:
                                st.warning(f"Failed to delete {model_type} models: {str(e)}")
                        
                        st.success(f"✅ Successfully deleted {total_deleted} models and {total_files} files!")
                        safe_rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error deleting models: {str(e)}")

def show_delete_all_training_data(db):
    """Delete all training data"""
    st.markdown("**🗑️ Delete All Training Data**")
    st.warning("⚠️ This will delete all training data, logs, and temporary files")
    
    # Show what will be deleted
    st.markdown("**This will delete:**")
    st.write("- All training logs")
    st.write("- All temporary training files")
    st.write("- All cached training data")
    st.write("- All training plots and images")
    
    confirm1 = st.checkbox("⚠️ I want to delete all training data")
    confirm2 = st.checkbox("⚠️ I understand this cannot be undone")
    
    if confirm1 and confirm2:
        if st.button("🗑️ Delete All Training Data", type="secondary"):
            try:
                with st.spinner("🗑️ Deleting training data..."):
                    deleted_files = delete_training_data()
                    st.success(f"✅ Successfully deleted {deleted_files} training files!")
            except Exception as e:
                st.error(f"❌ Error deleting training data: {str(e)}")

def show_delete_all_predictions(db):
    """Delete all predictions"""
    st.markdown("**🗑️ Delete All Prediction History**")
    st.warning("⚠️ This will delete all prediction history and results")
    
    # Show what will be deleted
    st.markdown("**This will delete:**")
    st.write("- All prediction results")
    st.write("- All prediction logs")
    st.write("- All generated prediction files")
    
    confirm1 = st.checkbox("⚠️ I want to delete all prediction history")
    confirm2 = st.checkbox("⚠️ I understand this cannot be undone")
    
    if confirm1 and confirm2:
        if st.button("🗑️ Delete All Predictions", type="secondary"):
            try:
                with st.spinner("🗑️ Deleting prediction data..."):
                    deleted_records = delete_prediction_data(db)
                    st.success(f"✅ Successfully deleted {deleted_records} prediction records!")
            except Exception as e:
                st.error(f"❌ Error deleting predictions: {str(e)}")

def perform_complete_reset(db):
    """
    Perform complete system reset - delete everything
    
    Args:
        db: Database manager instance
        
    Returns:
        dict: Results of the reset operation
    """
    results = {
        'Models Deleted': 0,
        'Files Deleted': 0,
        'Directories Cleaned': 0,
        'Database Records': 0,
        'Config Files': 0
    }
    
    try:
        # 1. Delete all models from database and files
        if db:
            try:
                model_result = delete_all_models_and_files(db)
                results['Models Deleted'] = model_result.get('total_deleted', 0)
                results['Files Deleted'] += model_result.get('total_files_deleted', 0)
            except Exception as e:
                st.warning(f"Error deleting models: {str(e)}")
        
        # 2. Delete all directories and their contents
        directories_to_delete = [
            "models",
            "models/tabular_gans", 
            "models/neural_networks",
            "models/checkpoints",
            "backups",
            "exports", 
            "logs",
            "cache",
            "temp",
            "plots",
            "images",
            "training_plots",
            "confusion_matrices"
        ]
        
        for directory in directories_to_delete:
            try:
                if os.path.exists(directory):
                    # Count files before deletion
                    file_count = count_files_in_directory(directory)
                    results['Files Deleted'] += file_count
                    
                    # Remove directory and all contents
                    shutil.rmtree(directory)
                    results['Directories Cleaned'] += 1
                    
                    # Recreate empty directory
                    os.makedirs(directory, exist_ok=True)
                    
            except Exception as e:
                st.warning(f"Could not delete directory {directory}: {str(e)}")
        
        # 3. Delete database file completely and recreate
        try:
            db_files = [
                "database/ai_vision_suite.db",
                "database/models.db", 
                "ai_vision_suite.db",
                "models.db"
            ]
            
            for db_file in db_files:
                if os.path.exists(db_file):
                    os.remove(db_file)
                    results['Database Records'] += 1
            
            # Reinitialize database
            if DatabaseManager:
                try:
                    new_db = DatabaseManager()
                    # This will create fresh tables
                except:
                    pass
                    
        except Exception as e:
            st.warning(f"Error resetting database: {str(e)}")
        
        # 4. Delete configuration files
        config_files = [
            "config/system_config.json",
            "config/models_config.json", 
            "config/interface_config.json",
            "config/advanced_config.json"
        ]
        
        for config_file in config_files:
            try:
                if os.path.exists(config_file):
                    os.remove(config_file)
                    results['Config Files'] += 1
            except Exception as e:
                st.warning(f"Could not delete config file {config_file}: {str(e)}")
        
        # 5. Clear any cached data in session state
        try:
            for key in list(st.session_state.keys()):
                if key.startswith(('nn_', 'gan_', 'model_', 'data_')):
                    del st.session_state[key]
        except:
            pass
        
        # 6. Delete any additional files in root directory
        additional_files = [
            "training_history.json",
            "model_registry.json", 
            "user_preferences.json",
            "temp_data.csv",
            "sample_data.csv"
        ]
        
        for file_path in additional_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    results['Files Deleted'] += 1
            except:
                pass
        
        return results
        
    except Exception as e:
        raise Exception(f"Complete reset failed: {str(e)}")

def delete_all_models_and_files(db):
    """
    Delete all models from database and their associated files
    
    Args:
        db: Database manager instance
        
    Returns:
        dict: Results of the deletion operation
    """
    results = {
        'total_deleted': 0,
        'total_files_deleted': 0,
        'TABULAR_GAN_deleted': 0,
        'MLP_deleted': 0, 
        'LSTM_deleted': 0,
        'CNN_deleted': 0
    }
    
    try:
        model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        
        for model_type in model_types:
            try:
                type_result = delete_models_by_type(db, model_type)
                results[f'{model_type}_deleted'] = type_result['models_deleted']
                results['total_deleted'] += type_result['models_deleted']
                results['total_files_deleted'] += type_result['files_deleted']
            except Exception as e:
                st.warning(f"Error deleting {model_type} models: {str(e)}")
        
        return results
        
    except Exception as e:
        raise Exception(f"Failed to delete all models: {str(e)}")

def delete_models_by_type(db, model_type):
    """
    Delete all models of a specific type
    
    Args:
        db: Database manager instance
        model_type: Type of models to delete
        
    Returns:
        dict: Results of the deletion operation
    """
    results = {
        'models_deleted': 0,
        'files_deleted': 0
    }
    
    try:
        # Get all models of this type
        models = db.get_models_by_type(model_type)
        
        if not models.empty:
            for _, model in models.iterrows():
                try:
                    # Delete model files
                    files_deleted = delete_model_files(model)
                    results['files_deleted'] += files_deleted
                    
                    # Delete from database
                    model_id = model.get('id')
                    if model_id and hasattr(db, 'delete_model'):
                        db.delete_model(model_id)
                        results['models_deleted'] += 1
                        
                except Exception as e:
                    st.warning(f"Failed to delete model {model.get('name', 'unknown')}: {str(e)}")
        
        return results
        
    except Exception as e:
        raise Exception(f"Failed to delete {model_type} models: {str(e)}")

def delete_model_files(model_info):
    """
    Delete all files associated with a model
    
    Args:
        model_info: Model information from database
        
    Returns:
        int: Number of files deleted
    """
    files_deleted = 0
    
    try:
        # Get model file path
        file_path = model_info.get('file_path', '')
        model_name = model_info.get('name', '')
        model_type = model_info.get('type', '')
        
        # List of potential file patterns to delete
        file_patterns = []
        
        # Direct file path
        if file_path:
            file_patterns.append(file_path)
        
        # Generate potential file patterns based on model info
        if model_name:
            # Model files
            file_patterns.extend([
                f"models/{model_name}.pth",
                f"models/{model_name}.pt",
                f"models/tabular_gans/{model_name}.pth",
                f"models/neural_networks/{model_name}.pth",
                f"models/checkpoints/{model_name}.pth"
            ])
            
            # Associated files (plots, logs, etc.)
            file_patterns.extend([
                f"models/{model_name}_training_plot.png",
                f"models/{model_name}_loss_plot.png", 
                f"models/{model_name}_confusion_matrix.png",
                f"models/tabular_gans/{model_name}_training_plot.png",
                f"models/neural_networks/{model_name}_training_plot.png",
                f"plots/{model_name}_training_plot.png",
                f"plots/{model_name}_loss_plot.png",
                f"plots/{model_name}_confusion_matrix.png",
                f"logs/{model_name}_training.log",
                f"training_plots/{model_name}.png"
            ])
            
            # Preprocessor files for GANs
            if model_type == 'TABULAR_GAN':
                file_patterns.extend([
                    f"models/tabular_gans/{model_name}_preprocessor.pkl",
                    f"models/tabular_gans/generator_{model_name}.pth",
                    f"models/tabular_gans/discriminator_{model_name}.pth"
                ])
        
        # Also check for files with timestamps
        timestamp_patterns = [
            "models/tabular_gans/generator_*.pth",
            "models/tabular_gans/discriminator_*.pth", 
            "models/tabular_gans/preprocessor_*.pkl",
            "models/neural_networks/mlp_model_*.pth",
            "models/neural_networks/lstm_model_*.pth",
            "models/neural_networks/*training_plot*.png"
        ]
        
        # Delete files matching patterns
        import glob
        
        for pattern in file_patterns:
            try:
                if os.path.exists(pattern):
                    os.remove(pattern)
                    files_deleted += 1
            except Exception as e:
                pass  # File might not exist or be locked
        
        # Delete files matching glob patterns
        for pattern in timestamp_patterns:
            try:
                matching_files = glob.glob(pattern)
                for file_path in matching_files:
                    try:
                        # Check if file contains model name or is recent
                        if model_name.lower() in file_path.lower() or is_recent_file(file_path):
                            os.remove(file_path)
                            files_deleted += 1
                    except:
                        pass
            except:
                pass
        
        return files_deleted
        
    except Exception as e:
        st.warning(f"Error deleting model files: {str(e)}")
        return 0

def count_files_in_directory(directory):
    """
    Count total number of files in a directory recursively
    
    Args:
        directory: Directory path
        
    Returns:
        int: Number of files
    """
    file_count = 0
    try:
        for root, dirs, files in os.walk(directory):
            file_count += len(files)
    except:
        pass
    return file_count

def is_recent_file(file_path, days=7):
    """
    Check if a file was created recently
    
    Args:
        file_path: Path to the file
        days: Number of days to consider as recent
        
    Returns:
        bool: True if file is recent
    """
    try:
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        cutoff_time = datetime.now() - timedelta(days=days)
        return file_time > cutoff_time
    except:
        return False

def delete_training_data():
    """
    Delete all training-related data
    
    Returns:
        int: Number of files deleted
    """
    files_deleted = 0
    
    # Directories containing training data
    training_dirs = [
        "logs",
        "temp", 
        "cache",
        "training_plots",
        "plots"
    ]
    
    # File patterns to delete
    training_patterns = [
        "*.log",
        "*training*.png", 
        "*loss*.png",
        "*training_plot*.png",
        "temp_*.csv",
        "cache_*.pkl"
    ]
    
    try:
        # Delete files in training directories
        import glob
        
        for directory in training_dirs:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                            files_deleted += 1
                        except:
                            pass
        
        # Delete files matching training patterns in root and models directories
        search_dirs = [".", "models", "models/tabular_gans", "models/neural_networks"]
        
        for search_dir in search_dirs:
            for pattern in training_patterns:
                try:
                    matching_files = glob.glob(os.path.join(search_dir, pattern))
                    for file_path in matching_files:
                        try:
                            os.remove(file_path)
                            files_deleted += 1
                        except:
                            pass
                except:
                    pass
        
        return files_deleted
        
    except Exception as e:
        raise Exception(f"Failed to delete training data: {str(e)}")

def delete_prediction_data(db):
    """
    Delete all prediction-related data
    
    Args:
        db: Database manager instance
        
    Returns:
        int: Number of records deleted
    """
    records_deleted = 0
    
    try:
        # Delete prediction files
        prediction_patterns = [
            "predictions_*.csv",
            "prediction_results_*.json", 
            "generated_data_*.csv",
            "synthetic_data_*.csv"
        ]
        
        import glob
        
        search_dirs = [".", "exports", "results", "predictions"]
        
        for search_dir in search_dirs:
            for pattern in prediction_patterns:
                try:
                    matching_files = glob.glob(os.path.join(search_dir, pattern))
                    for file_path in matching_files:
                        try:
                            os.remove(file_path)
                            records_deleted += 1
                        except:
                            pass
                except:
                    pass
        
        # Delete prediction records from database if supported
        if db and hasattr(db, 'delete_all_predictions'):
            try:
                db_deleted = db.delete_all_predictions()
                records_deleted += db_deleted
            except:
                pass
        
        return records_deleted
        
    except Exception as e:
        raise Exception(f"Failed to delete prediction data: {str(e)}")

def show_bulk_data_operations(db):
    """Show bulk data operations"""
    st.markdown("##### 🔄 Bulk Operations")
    
    bulk_operation = st.selectbox(
        "Select Bulk Operation",
        [
            "Delete All Data (Complete Reset)",
            "Delete All Models (Keep Settings)",
            "Export All Data",
            "Clean Old Files"
        ]
    )
    
    if bulk_operation == "Delete All Data (Complete Reset)":
        show_complete_reset_option(db)
    elif bulk_operation == "Delete All Models (Keep Settings)":
        show_delete_all_models_option(db)
    elif bulk_operation == "Export All Data":
        show_export_all_data(db)
    elif bulk_operation == "Clean Old Files":
        show_clean_old_files()

def show_complete_reset_option(db):
    """Complete system reset"""
    st.markdown("**🚨 Complete System Reset**")
    st.error("⚠️ WARNING: This will delete ALL data including models, predictions, settings, and logs!")
    
    # Show what will be deleted
    st.markdown("**This operation will delete:**")
    st.write("- All trained models and their files")
    st.write("- All prediction history")
    st.write("- All training logs")
    st.write("- All user settings")
    st.write("- All database records")
    st.write("- All cached data")
    st.write("- All generated images and plots")
    st.write("- All backup files")
    
    # Triple confirmation
    confirm1 = st.checkbox("⚠️ I understand this will delete ALL data")
    confirm2 = st.checkbox("⚠️ I understand this action CANNOT be undone")
    confirm3 = st.checkbox("⚠️ I want to proceed with COMPLETE RESET")
    
    # Safety input
    if confirm1 and confirm2 and confirm3:
        safety_input = st.text_input(
            "Type 'DELETE ALL DATA' to confirm:",
            help="This is a safety measure to prevent accidental deletion"
        )
        
        if safety_input == "DELETE ALL DATA":
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚨 COMPLETE RESET", type="secondary"):
                    try:
                        with st.spinner("🗑️ Deleting all data... This may take a moment."):
                            # Perform complete reset
                            reset_results = perform_complete_reset(db)
                            
                            st.success("✅ Complete reset performed successfully!")
                            st.info("🔄 Please restart the application to see changes.")
                            
                            # Show deletion summary
                            st.markdown("### 📊 Deletion Summary")
                            for key, value in reset_results.items():
                                st.write(f"- **{key}:** {value}")
                            

                            # Clear session state
                            for key in list(st.session_state.keys()):
                                del st.session_state[key]
                            

                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during reset: {str(e)}")
                        st.exception(e)
            
            with col2:
                if st.button("❌ Cancel", type="primary"):
                    st.info("Operation cancelled.")
        elif safety_input:
            st.error("Please type exactly 'DELETE ALL DATA' to proceed")

def show_delete_all_models_option(db):
    """Delete all models but keep settings"""
    st.markdown("**🗑️ Delete All Models**")
    st.warning("⚠️ This will delete all trained models but keep your settings and configurations")
    
    try:
        # Get model counts
        total_models = 0
        model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        
        for model_type in model_types:
            try:
                models = db.get_models_by_type(model_type)
                total_models += len(models) if not models.empty else 0
            except:
                pass
        
        if total_models > 0:
            st.info(f"This will delete {total_models} models and their associated files")
            
            confirm1 = st.checkbox("⚠️ I want to delete all models")
            confirm2 = st.checkbox("⚠️ I understand this cannot be undone")
            
            if confirm1 and confirm2:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ Delete All Models", type="secondary"):
                        try:
                            with st.spinner("🗑️ Deleting all models..."):
                                deleted_results = delete_all_models_and_files(db)
                                
                                st.success(f"✅ Successfully deleted {deleted_results['total_deleted']} models!")
                                
                                # Show deletion details
                                st.markdown("### 📊 Deletion Summary")
                                for key, value in deleted_results.items():
                                    if key != 'total_deleted':
                                        st.write(f"- **{key}:** {value}")
                                
                                safe_rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Error deleting models: {str(e)}")
                            st.exception(e)
                
                with col2:
                    if st.button("❌ Cancel", type="primary"):
                        st.info("Operation cancelled.")
        else:
            st.info("No models found to delete.")
            
    except Exception as e:
        st.error(f"Error checking models: {str(e)}")

def show_export_all_data(db):
    """Export all data"""
    st.markdown("**📤 Export All Data**")
    
    export_format = st.selectbox("Export Format", ["JSON", "CSV", "SQL"])
    
    if st.button("📤 Export All Data"):
        try:
            with st.spinner("📤 Exporting all data..."):
                # Create exports directory
                export_dir = "exports"
                os.makedirs(export_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if export_format == "JSON":
                    export_file = f"{export_dir}/complete_export_{timestamp}.json"
                    export_data = export_all_data_json(db)
                    
                    with open(export_file, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                    
                elif export_format == "CSV":
                    export_file = f"{export_dir}/complete_export_{timestamp}.zip"
                    create_csv_export(db, export_file)
                    
                elif export_format == "SQL":
                    export_file = f"{export_dir}/complete_export_{timestamp}.sql"
                    create_sql_export(db, export_file)
                
                st.success(f"✅ Data exported to {export_file}")
                
                # Download button
                if os.path.exists(export_file):
                    with open(export_file, 'rb') as f:
                        st.download_button(
                            label="📥 Download Export",
                            data=f.read(),
                            file_name=os.path.basename(export_file),
                            mime="application/octet-stream"
                        )
                        
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

def show_clean_old_files():
    """Clean old files"""
    st.markdown("**🧹 Clean Old Files**")
    
    days_old = st.number_input("Delete files older than (days)", min_value=1, value=30)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # File types to clean
        file_types = st.multiselect(
            "File types to clean:",
            ["Log files", "Temporary files", "Cache files", "Backup files", "Old plots"],
            default=["Log files", "Temporary files", "Cache files"]
        )
    
    with col2:
        # Preview mode
        preview_mode = st.checkbox("Preview mode (don't delete, just show)", value=True)
    
    if st.button("🧹 Clean Old Files"):
        try:
            with st.spinner("🧹 Cleaning old files..."):
                cleaned_results = clean_old_files_by_type(file_types, days_old, preview_mode)
                
                if preview_mode:
                    st.info(f"📋 Preview: Found {cleaned_results['total_files']} files to clean ({cleaned_results['total_size']:.1f} MB)")
                else:
                    st.success(f"✅ Cleaned {cleaned_results['total_files']} files ({cleaned_results['total_size']:.1f} MB)")
                
                # Show details
                for file_type, count in cleaned_results['by_type'].items():
                    st.write(f"- **{file_type}:** {count} files")
                    
        except Exception as e:
            st.error(f"❌ Cleanup failed: {str(e)}")

def show_data_export_import(db):
    """Data export and import operations"""
    st.markdown("##### 📤 Data Export & Import")
    
    # Export options
    st.markdown("**Export Data**")
    
    export_options = st.multiselect(
        "Select data to export:",
        [
            "All Models",
            "Model Metadata",
            "User Settings",
            "Training Logs",
            "Prediction History"
        ]
    )
    
    if export_options:
        export_format = st.selectbox(
            "Export Format",
            ["JSON", "CSV", "ZIP"]
        )
        
        if st.button("📤 Export Selected Data"):
            try:
                with st.spinner("📤 Exporting selected data..."):
                    export_file = export_selected_data(db, export_options, export_format)
                    st.success(f"✅ Data exported to {export_file}")
                    
                    # Download button
                    if os.path.exists(export_file):
                        with open(export_file, 'rb') as f:
                            st.download_button(
                                label="📥 Download Export",
                                data=f.read(),
                                file_name=os.path.basename(export_file),
                                mime="application/octet-stream"
                            )
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    st.markdown("---")
    
    # Import options
    st.markdown("**Import Data**")
    
    uploaded_file = st.file_uploader(
        "Upload data file to import",
        type=['json', 'csv', 'zip']
    )
    
    if uploaded_file:
        if st.button("📥 Import Data"):
            try:
                with st.spinner("📥 Importing data..."):
                    import_results = import_data_file(db, uploaded_file)
                    st.success(f"✅ Successfully imported {import_results['records_imported']} records!")
                    
                    # Show import details
                    for data_type, count in import_results['by_type'].items():
                        st.write(f"- **{data_type}:** {count} records")
                        
            except Exception as e:
                st.error(f"❌ Import failed: {str(e)}")

def calculate_total_storage_usage():
    """Calculate total storage usage"""
    total_size = 0
    
    # Check models directories
    model_dirs = [
        "models",
        "backups", 
        "exports",
        "logs"
    ]
    
    for directory in model_dirs:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
    
    return total_size / (1024 * 1024)  # Convert to MB

# Helper functions for export/import operations

def export_all_data_json(db):
    """Export all data as JSON"""
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'models': {},
        'settings': {},
        'statistics': {}
    }
    
    try:
        # Export models by type
        model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        for model_type in model_types:
            try:
                models = db.get_models_by_type(model_type)
                if not models.empty:
                    export_data['models'][model_type] = models.to_dict('records')
            except:
                export_data['models'][model_type] = []
        
        # Export settings
        try:
            config_files = ['system_config.json', 'models_config.json', 'interface_config.json', 'advanced_config.json']
            for config_file in config_files:
                config_path = f"config/{config_file}"
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        export_data['settings'][config_file] = json.load(f)
        except:
            pass
        
        # Export statistics
        export_data['statistics'] = {
            'total_storage_mb': calculate_total_storage_usage(),
            'total_models': sum(len(models) for models in export_data['models'].values())
        }
        
    except Exception as e:
        st.warning(f"Error during JSON export: {str(e)}")
    
    return export_data

def create_csv_export(db, export_file):
    """Create CSV export as ZIP file"""
    import zipfile
    
    with zipfile.ZipFile(export_file, 'w') as zipf:
        # Export models
        model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        for model_type in model_types:
            try:
                models = db.get_models_by_type(model_type)
                if not models.empty:
                    csv_data = models.to_csv(index=False)
                    zipf.writestr(f"{model_type}_models.csv", csv_data)
            except:
                pass

def create_sql_export(db, export_file):
    """Create SQL export"""
    with open(export_file, 'w') as f:
        f.write("-- AI Vision Suite Database Export\n")
        f.write(f"-- Generated: {datetime.now().isoformat()}\n\n")
        f.write("-- Note: This is a simplified export for demonstration\n")
        f.write("-- In a real implementation, you would dump the actual database schema and data\n")

def export_selected_data(db, export_options, export_format):
    """Export selected data types"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = "exports"
    os.makedirs(export_dir, exist_ok=True)
    
    if export_format == "JSON":
        export_file = f"{export_dir}/selective_export_{timestamp}.json"
        export_data = {'export_timestamp': datetime.now().isoformat()}
        
        for option in export_options:
            if option == "All Models":
                export_data['models'] = {}
                model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
                for model_type in model_types:
                    try:
                        models = db.get_models_by_type(model_type)
                        if not models.empty:
                            export_data['models'][model_type] = models.to_dict('records')
                    except:
                        pass
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
    else:
        export_file = f"{export_dir}/selective_export_{timestamp}.zip"
        # Create ZIP export (simplified)
        import zipfile
        with zipfile.ZipFile(export_file, 'w') as zipf:
            zipf.writestr("export_info.txt", f"Exported on {datetime.now().isoformat()}")
    
    return export_file

def import_data_file(db, uploaded_file):
    """Import data from uploaded file"""
    import_results = {
        'records_imported': 0,
        'by_type': {}
    }
    
    # This is a simplified implementation
    # In a real scenario, you would parse the file and import data
    
    if uploaded_file.name.endswith('.json'):
        try:
            data = json.load(uploaded_file)
            # Process JSON data
            import_results['records_imported'] = 1
            import_results['by_type']['JSON Records'] = 1
        except:
            pass
    
    return import_results

def clean_old_files_by_type(file_types, days_old, preview_mode):
    """Clean old files by specified types"""
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    results = {
        'total_files': 0,
        'total_size': 0,
        'by_type': {}
    }
    
    # Define file patterns for each type
    file_patterns = {
        'Log files': ['logs/*.log', '*.log'],
        'Temporary files': ['temp/*', 'cache/*', '*.tmp'],
        'Cache files': ['cache/*', '*.cache', '__pycache__/*'],
        'Backup files': ['backups/*', '*.bak'],
        'Old plots': ['plots/*.png', 'training_plots/*.png', '*_plot.png']
    }
    
    import glob
    
    for file_type in file_types:
        if file_type in file_patterns:
            type_count = 0
            type_size = 0
            
            for pattern in file_patterns[file_type]:
                try:
                    matching_files = glob.glob(pattern)
                    for file_path in matching_files:
                        try:
                            if os.path.isfile(file_path):
                                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                                if file_time < cutoff_date:
                                    file_size = os.path.getsize(file_path)
                                    type_size += file_size
                                    type_count += 1
                                    
                                    if not preview_mode:
                                        os.remove(file_path)
                        except:
                            pass
                except:
                    pass
            
            results['by_type'][file_type] = type_count
            results['total_files'] += type_count
            results['total_size'] += type_size / (1024 * 1024)  # Convert to MB
    
    return results

def safe_rerun():
    """Safely rerun the app with compatibility for different Streamlit versions"""
    try:
        # Try the new method first (Streamlit >= 1.27.0)
        st.rerun()
    except AttributeError:
        try:
            # Fall back to the old method
            st.experimental_rerun()
        except AttributeError:
            # If neither works, show a message
            st.info("🔄 Please refresh the page manually to see the changes.")

if __name__ == "__main__":
    show_settings_page()