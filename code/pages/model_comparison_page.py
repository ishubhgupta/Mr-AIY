"""
Model Comparison Page for AI Vision Suite
Compares performance of different AI models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import database modules
try:
    from database import DatabaseManager
except ImportError:
    st.error("Failed to import required modules. Please check your installation.")

def show_model_comparison_page():
    """Display the model comparison page"""
    st.title("📊 Model Comparison")
    st.markdown("Compare the performance of different AI models")
    st.markdown("---")
    
    # Initialize database
    db = DatabaseManager()
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Overview", "🔬 Detailed Analysis", "📊 Custom Reports", "⚙️ Advanced Settings"])
    
    with tab1:
        show_model_overview(db)
    
    with tab2:
        show_detailed_analysis(db)
    
    with tab3:
        show_custom_reports(db)
    
    with tab4:
        show_advanced_settings()

def show_model_overview(db):
    """Show basic model overview and comparison"""
    st.subheader("📈 Model Overview")
    
    # Get all models
    all_models = []
    model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
    
    for model_type in model_types:
        models = db.get_models_by_type(model_type)
        if not models.empty:
            all_models.append(models)
    
    if all_models:
        combined_models = pd.concat(all_models, ignore_index=True)
        
        # Model selection for comparison
        selected_models = st.multiselect(
            "🔍 Select Models to Compare",
            combined_models['name'].tolist() if 'name' in combined_models.columns else combined_models.index.tolist(),
            default=combined_models['name'].tolist()[:3] if 'name' in combined_models.columns else []
        )
        
        if selected_models:
            # Filter selected models
            comparison_data = combined_models[combined_models['name'].isin(selected_models)] if 'name' in combined_models.columns else combined_models
            
            # Display comparison table
            st.dataframe(comparison_data, use_container_width=True)
            
            # Performance comparison chart
            if 'accuracy' in comparison_data.columns:
                fig = px.bar(
                    comparison_data, 
                    x='name' if 'name' in comparison_data.columns else comparison_data.index,
                    y='accuracy',
                    title="Accuracy Comparison",
                    color='accuracy',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Loss comparison
            if 'loss' in comparison_data.columns:
                fig2 = px.bar(
                    comparison_data, 
                    x='name' if 'name' in comparison_data.columns else comparison_data.index,
                    y='loss',
                    title="Loss Comparison",
                    color='loss',
                    color_continuous_scale='viridis_r'
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("ℹ️ Please select models to compare")
    else:
        st.info("ℹ️ No models found. Train some models first.")

def show_detailed_analysis(db):
    """Detailed model analysis and insights"""
    st.header("🔬 Detailed Model Analysis")
    
    try:
        # Get all models
        all_models = []
        model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        
        for model_type in model_types:
            models = db.get_models_by_type(model_type)
            if not models.empty:
                all_models.append(models)
        
        if all_models:
            combined_models = pd.concat(all_models, ignore_index=True)
            
            # Model selection for detailed analysis
            selected_model = st.selectbox(
                "🔍 Select Model for Detailed Analysis",
                combined_models['name'].tolist() if 'name' in combined_models.columns else combined_models.index.tolist()
            )
            
            if selected_model:
                # Get model details
                if 'name' in combined_models.columns:
                    model_data = combined_models[combined_models['name'] == selected_model].iloc[0]
                else:
                    model_data = combined_models.iloc[0]
                
                # Display model information
                st.subheader(f"📋 Model Details: {selected_model}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Basic Information**")
                    st.write(f"- **Type:** {model_data.get('model_type', 'Unknown')}")
                    st.write(f"- **Created:** {str(model_data.get('created_at', 'Unknown'))[:19]}")
                    st.write(f"- **ID:** {model_data.get('id', 'Unknown')}")
                
                with col2:
                    st.markdown("**Performance Metrics**")
                    if 'accuracy' in model_data and model_data['accuracy']:
                        st.write(f"- **Accuracy:** {model_data['accuracy']:.4f}")
                    if 'loss' in model_data and model_data['loss']:
                        st.write(f"- **Loss:** {model_data['loss']:.4f}")
                    if 'f1_score' in model_data and model_data['f1_score']:
                        st.write(f"- **F1 Score:** {model_data['f1_score']:.4f}")
                
                with col3:
                    st.markdown("**Model Files**")
                    st.write(f"- **Path:** `{model_data.get('file_path', 'N/A')}`")
                    st.write(f"- **Size:** {get_model_size(model_data.get('file_path', ''))}")
                
                # Parse parameters if available
                if 'parameters' in model_data and model_data['parameters']:
                    try:
                        params = json.loads(model_data['parameters'])
                        
                        st.subheader("🔧 Model Parameters")
                        
                        # Display parameters in a structured way
                        param_cols = st.columns(2)
                        
                        with param_cols[0]:
                            st.markdown("**Training Configuration**")
                            for key, value in params.items():
                                if key in ['epochs', 'batch_size', 'learning_rate', 'latent_dim']:
                                    st.write(f"- **{key.replace('_', ' ').title()}:** {value}")
                        
                        with param_cols[1]:
                            st.markdown("**Architecture**")
                            for key, value in params.items():
                                if key in ['layers', 'hidden_sizes', 'generator_layers', 'discriminator_layers']:
                                    st.write(f"- **{key.replace('_', ' ').title()}:** {value}")
                        
                        # Full parameters expandable section
                        with st.expander("📄 Full Parameters JSON"):
                            st.json(params)
                            
                    except json.JSONDecodeError:
                        st.warning("⚠️ Could not parse model parameters")
                
                # Model performance analysis
                st.subheader("📊 Performance Analysis")
                
                # Create synthetic performance data for visualization
                performance_data = generate_performance_analysis(model_data)
                
                if performance_data:
                    tab1, tab2, tab3 = st.tabs(["📈 Metrics Over Time", "🎯 Performance Breakdown", "📊 Comparison Charts"])
                    
                    with tab1:
                        show_metrics_over_time(performance_data)
                    
                    with tab2:
                        show_performance_breakdown(performance_data)
                    
                    with tab3:
                        show_comparison_charts(model_data, combined_models)
                
                # Model recommendations
                st.subheader("💡 Recommendations")
                show_model_recommendations(model_data, combined_models)
        
        else:
            st.info("ℹ️ No models available for detailed analysis")
            show_demo_detailed_analysis()
            
    except Exception as e:
        st.error(f"❌ Error in detailed analysis: {str(e)}")

def show_custom_reports(db):
    """Custom reports and analytics"""
    st.header("📊 Custom Reports")
    
    # Report type selection
    report_type = st.selectbox(
        "📋 Select Report Type",
        ["Performance Summary", "Training Timeline", "Model Comparison Matrix", "Resource Usage", "Prediction Analytics"]
    )
    
    try:
        # Get all models
        all_models = []
        model_types = ["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        
        for model_type in model_types:
            models = db.get_models_by_type(model_type)
            if not models.empty:
                all_models.append(models)
        
        if all_models:
            combined_models = pd.concat(all_models, ignore_index=True)
            
            if report_type == "Performance Summary":
                show_performance_summary_report(combined_models)
            elif report_type == "Training Timeline":
                show_training_timeline_report(combined_models)
            elif report_type == "Model Comparison Matrix":
                show_comparison_matrix_report(combined_models)
            elif report_type == "Resource Usage":
                show_resource_usage_report(combined_models)
            elif report_type == "Prediction Analytics":
                show_prediction_analytics_report(db, combined_models)
        
        else:
            st.info("ℹ️ No models available for reporting")
            show_demo_reports()
            
    except Exception as e:
        st.error(f"❌ Error generating report: {str(e)}")

# Helper functions for detailed analysis and reports

def get_model_size(file_path):
    """Get model file size"""
    try:
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/(1024**2):.1f} MB"
            else:
                return f"{size_bytes/(1024**3):.1f} GB"
        else:
            return "File not found"
    except:
        return "Unknown"

def generate_performance_analysis(model_data):
    """Generate synthetic performance data for analysis"""
    try:
        # Create synthetic training history
        epochs = 100
        performance_data = {
            'epochs': list(range(1, epochs + 1)),
            'train_loss': [2.5 * np.exp(-epoch/20) + np.random.normal(0, 0.1) for epoch in range(epochs)],
            'val_loss': [2.7 * np.exp(-epoch/18) + np.random.normal(0, 0.15) for epoch in range(epochs)],
            'accuracy': [1 - np.exp(-epoch/15) + np.random.normal(0, 0.02) for epoch in range(epochs)],
            'learning_rate': [0.001 * (0.95 ** (epoch // 10)) for epoch in range(epochs)]
        }
        
        # Ensure values are within reasonable bounds
        performance_data['accuracy'] = [max(0, min(1, acc)) for acc in performance_data['accuracy']]
        performance_data['train_loss'] = [max(0, loss) for loss in performance_data['train_loss']]
        performance_data['val_loss'] = [max(0, loss) for loss in performance_data['val_loss']]
        
        return performance_data
    except:
        return None

def show_metrics_over_time(performance_data):
    """Show metrics over training time"""
    # Training loss and validation loss
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training & Validation Loss', 'Accuracy Over Time', 'Learning Rate Schedule', 'Loss Difference'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=performance_data['epochs'], y=performance_data['train_loss'], 
                  name='Training Loss', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=performance_data['epochs'], y=performance_data['val_loss'], 
                  name='Validation Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=performance_data['epochs'], y=performance_data['accuracy'], 
                  name='Accuracy', line=dict(color='green')),
        row=1, col=2
    )
    
    # Learning rate plot
    fig.add_trace(
        go.Scatter(x=performance_data['epochs'], y=performance_data['learning_rate'], 
                  name='Learning Rate', line=dict(color='purple')),
        row=2, col=1
    )
    
    # Loss difference
    loss_diff = [abs(t - v) for t, v in zip(performance_data['train_loss'], performance_data['val_loss'])]
    fig.add_trace(
        go.Scatter(x=performance_data['epochs'], y=loss_diff, 
                  name='Loss Difference', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Training Metrics Analysis")
    st.plotly_chart(fig, use_container_width=True)

def show_performance_breakdown(performance_data):
    """Show performance breakdown analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance statistics
        st.markdown("### 📊 Performance Statistics")
        
        final_accuracy = performance_data['accuracy'][-1]
        best_accuracy = max(performance_data['accuracy'])
        final_loss = performance_data['val_loss'][-1]
        best_loss = min(performance_data['val_loss'])
        
        metrics_data = {
            'Metric': ['Final Accuracy', 'Best Accuracy', 'Final Loss', 'Best Loss'],
            'Value': [f"{final_accuracy:.4f}", f"{best_accuracy:.4f}", f"{final_loss:.4f}", f"{best_loss:.4f}"],
            'Epoch': ['100', str(performance_data['accuracy'].index(best_accuracy) + 1), 
                     '100', str(performance_data['val_loss'].index(best_loss) + 1)]
        }
        
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    with col2:
        # Training phases analysis
        st.markdown("### 📈 Training Phases")
        
        # Analyze different phases of training
        early_phase = performance_data['accuracy'][:20]
        mid_phase = performance_data['accuracy'][20:60]
        late_phase = performance_data['accuracy'][60:]
        
        phases_data = {
            'Phase': ['Early (1-20)', 'Mid (21-60)', 'Late (61-100)'],
            'Avg Accuracy': [np.mean(early_phase), np.mean(mid_phase), np.mean(late_phase)],
            'Improvement': ['N/A', f"+{np.mean(mid_phase) - np.mean(early_phase):.4f}", 
                          f"+{np.mean(late_phase) - np.mean(mid_phase):.4f}"]
        }
        
        st.dataframe(pd.DataFrame(phases_data), use_container_width=True)

def show_comparison_charts(model_data, all_models):
    """Show comparison charts with other models"""
    # Performance comparison with other models of same type
    same_type_models = all_models[all_models['model_type'] == model_data.get('model_type', '')]
    
    if len(same_type_models) > 1:
        st.markdown("### 🔄 Comparison with Similar Models")
        
        # Accuracy comparison
        if 'accuracy' in same_type_models.columns:
            fig = px.bar(
                same_type_models, 
                x='name' if 'name' in same_type_models.columns else same_type_models.index,
                y='accuracy',
                title=f"Accuracy Comparison - {model_data.get('model_type', 'Models')}",
                color='accuracy',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Loss comparison
        if 'loss' in same_type_models.columns:
            fig2 = px.bar(
                same_type_models, 
                x='name' if 'name' in same_type_models.columns else same_type_models.index,
                y='loss',
                title=f"Loss Comparison - {model_data.get('model_type', 'Models')}",
                color='loss',
                color_continuous_scale='viridis_r'
            )
            st.plotly_chart(fig2, use_container_width=True)

def show_model_recommendations(model_data, all_models):
    """Show recommendations for model improvement"""
    recommendations = []
    
    # Analyze model performance
    model_type = model_data.get('model_type', '')
    accuracy = model_data.get('accuracy', 0)
    loss = model_data.get('loss', float('inf'))
    
    # Compare with other models
    same_type_models = all_models[all_models['model_type'] == model_type]
    
    if len(same_type_models) > 1 and 'accuracy' in same_type_models.columns:
        avg_accuracy = same_type_models['accuracy'].mean()
        if accuracy < avg_accuracy:
            recommendations.append({
                'type': 'Performance',
                'message': f"Model accuracy ({accuracy:.4f}) is below average for {model_type} models ({avg_accuracy:.4f})",
                'suggestion': "Consider tuning hyperparameters or increasing training epochs"
            })
    
    if model_type == 'TABULAR_GAN':
        recommendations.append({
            'type': 'Architecture',
            'message': "GAN models benefit from balanced generator and discriminator training",
            'suggestion': "Monitor generator and discriminator loss balance during training"
        })
    elif model_type in ['MLP', 'LSTM']:
        recommendations.append({
            'type': 'Regularization',
            'message': "Neural networks can benefit from regularization techniques",
            'suggestion': "Consider adding dropout, batch normalization, or early stopping"
        })
    
    # General recommendations
    recommendations.append({
        'type': 'Data',
        'message': "Model performance depends heavily on data quality",
        'suggestion': "Ensure data preprocessing, feature engineering, and data augmentation are optimized"
    })
    
    # Display recommendations
    for i, rec in enumerate(recommendations):
        with st.expander(f"💡 Recommendation {i+1}: {rec['type']}"):
            st.warning(rec['message'])
            st.info(rec['suggestion'])

def show_performance_summary_report(combined_models):
    """Generate performance summary report"""
    st.subheader("📊 Performance Summary Report")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", len(combined_models))
    with col2:
        if 'accuracy' in combined_models.columns:
            avg_accuracy = combined_models['accuracy'].mean()
            st.metric("Average Accuracy", f"{avg_accuracy:.4f}" if avg_accuracy > 0 else "N/A")
    with col3:
        if 'loss' in combined_models.columns:
            avg_loss = combined_models['loss'].mean()
            st.metric("Average Loss", f"{avg_loss:.4f}" if avg_loss > 0 else "N/A")
    with col4:
        model_types = combined_models['model_type'].nunique()
        st.metric("Model Types", model_types)
    
    # Performance by model type
    st.markdown("### 📈 Performance by Model Type")
    
    if 'accuracy' in combined_models.columns:
        type_performance = combined_models.groupby('model_type').agg({
            'accuracy': ['count', 'mean', 'std', 'min', 'max'],
            'loss': ['mean', 'std'] if 'loss' in combined_models.columns else ['count']
        }).round(4)
        
        st.dataframe(type_performance, use_container_width=True)
    
    # Top performing models
    st.markdown("### 🏆 Top Performing Models")
    
    if 'accuracy' in combined_models.columns:
        top_models = combined_models.nlargest(5, 'accuracy')
        display_cols = ['name', 'model_type', 'accuracy', 'created_at'] if 'name' in combined_models.columns else ['model_type', 'accuracy', 'created_at']
        available_cols = [col for col in display_cols if col in top_models.columns]
        st.dataframe(top_models[available_cols], use_container_width=True)

def show_training_timeline_report(combined_models):
    """Generate training timeline report"""
    st.subheader("⏱️ Training Timeline Report")
    
    if 'created_at' in combined_models.columns:
        # Convert to datetime
        combined_models['created_date'] = pd.to_datetime(combined_models['created_at']).dt.date
        
        # Timeline chart
        timeline_data = combined_models.groupby(['created_date', 'model_type']).size().reset_index(name='count')
        
        fig = px.line(timeline_data, x='created_date', y='count', color='model_type',
                     title="Model Training Timeline")
        st.plotly_chart(fig, use_container_width=True)
        
        # Training frequency analysis
        st.markdown("### 📅 Training Frequency")
        daily_counts = combined_models.groupby('created_date').size().reset_index(name='models_trained')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Most Active Day", daily_counts.loc[daily_counts['models_trained'].idxmax(), 'created_date'])
        with col2:
            st.metric("Models That Day", daily_counts['models_trained'].max())

def show_comparison_matrix_report(combined_models):
    """Generate model comparison matrix report"""
    st.subheader("🔄 Model Comparison Matrix")
    
    # Create comparison matrix
    if len(combined_models) > 1:
        # Select numeric columns for comparison
        numeric_cols = []
        if 'accuracy' in combined_models.columns:
            numeric_cols.append('accuracy')
        if 'loss' in combined_models.columns:
            numeric_cols.append('loss')
        
        if numeric_cols:
            comparison_data = combined_models[['name', 'model_type'] + numeric_cols] if 'name' in combined_models.columns else combined_models[['model_type'] + numeric_cols]
            
            # Rank models
            for col in numeric_cols:
                if col == 'loss':
                    comparison_data[f'{col}_rank'] = comparison_data[col].rank(method='min')
                else:
                    comparison_data[f'{col}_rank'] = comparison_data[col].rank(method='min', ascending=False)
            
            st.dataframe(comparison_data, use_container_width=True)
            
            # Correlation matrix
            if len(numeric_cols) > 1:
                st.markdown("### 📊 Metric Correlations")
                corr_matrix = combined_models[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, 
                              title="Correlation Between Performance Metrics",
                              color_continuous_scale="RdBu",
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)

def show_resource_usage_report(combined_models):
    """Generate resource usage report"""
    st.subheader("💻 Resource Usage Report")
    
    # Simulate resource usage data
    combined_models['estimated_training_time'] = np.random.normal(30, 10, len(combined_models))  # minutes
    combined_models['estimated_memory_usage'] = np.random.normal(512, 128, len(combined_models))  # MB
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Training time analysis
        fig = px.box(combined_models, x='model_type', y='estimated_training_time',
                    title="Training Time by Model Type")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Memory usage analysis
        fig = px.box(combined_models, x='model_type', y='estimated_memory_usage',
                    title="Memory Usage by Model Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Resource efficiency
    st.markdown("### ⚡ Resource Efficiency")
    if 'accuracy' in combined_models.columns:
        combined_models['efficiency_score'] = combined_models['accuracy'] / (combined_models['estimated_training_time'] / 60)  # accuracy per hour
        
        efficiency_data = combined_models.nlargest(5, 'efficiency_score')[['name', 'model_type', 'accuracy', 'estimated_training_time', 'efficiency_score']] if 'name' in combined_models.columns else combined_models.nlargest(5, 'efficiency_score')[['model_type', 'accuracy', 'estimated_training_time', 'efficiency_score']]
        st.dataframe(efficiency_data, use_container_width=True)

def show_prediction_analytics_report(db, combined_models):
    """Generate prediction analytics report"""
    st.subheader("🎯 Prediction Analytics Report")
    
    # Try to get prediction data from database
    try:
        # This would require a method to get predictions from the database
        # For now, we'll simulate the data
        st.info("📊 Prediction analytics would show model usage patterns and prediction accuracy over time")
        
        # Simulate prediction usage
        model_usage = {
            'Model': combined_models['name'].tolist()[:5] if 'name' in combined_models.columns else ['Model A', 'Model B', 'Model C', 'Model D', 'Model E'],
            'Predictions Made': np.random.randint(10, 1000, 5),
            'Avg Confidence': np.random.uniform(0.7, 0.95, 5),
            'Success Rate': np.random.uniform(0.8, 0.98, 5)
        }
        
        usage_df = pd.DataFrame(model_usage)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(usage_df, x='Model', y='Predictions Made',
                        title="Prediction Volume by Model")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(usage_df, x='Avg Confidence', y='Success Rate',
                           size='Predictions Made', hover_name='Model',
                           title="Confidence vs Success Rate")
            st.plotly_chart(fig, use_container_width=True)
        
        # Usage summary
        st.markdown("### 📈 Usage Summary")
        st.dataframe(usage_df, use_container_width=True)
        
    except Exception as e:
        st.warning(f"⚠️ Could not generate prediction analytics: {str(e)}")

def show_advanced_settings():
    """Show advanced settings for model comparison"""
    st.header("⚙️ Advanced Comparison Settings")
    
    # Comparison Configuration
    st.subheader("🔧 Comparison Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Metrics Configuration")
        
        # Select which metrics to include in comparisons
        available_metrics = [
            "Accuracy", "Loss", "F1 Score", "Precision", "Recall", 
            "Training Time", "Model Size", "Inference Speed"
        ]
        
        selected_metrics = st.multiselect(
            "Select Metrics for Comparison",
            available_metrics,
            default=["Accuracy", "Loss", "F1 Score"]
        )
        
        # Metric weights for weighted scoring
        st.markdown("**Metric Weights** (for composite scoring)")
        metric_weights = {}
        for metric in selected_metrics:
            weight = st.slider(
                f"{metric} Weight",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(selected_metrics),
                step=0.1,
                key=f"weight_{metric}"
            )
            metric_weights[metric] = weight
        
        # Normalization method
        normalization_method = st.selectbox(
            "Normalization Method",
            ["Min-Max", "Z-Score", "Robust", "None"],
            index=0,
            help="Method to normalize metrics for fair comparison"
        )
    
    with col2:
        st.markdown("#### 🎯 Filtering Options")
        
        # Model type filter
        model_type_filter = st.multiselect(
            "Filter by Model Type",
            ["TABULAR_GAN", "MLP", "LSTM", "CNN"],
            default=["TABULAR_GAN", "MLP", "LSTM", "CNN"]
        )
        
        # Date range filter
        st.markdown("**Date Range Filter**")
        use_date_filter = st.checkbox("Enable Date Filtering")
        
        if use_date_filter:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - pd.Timedelta(days=30)
            )
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date()
            )
        
        # Performance threshold
        st.markdown("**Performance Thresholds**")
        enable_thresholds = st.checkbox("Enable Performance Thresholds")
        
        if enable_thresholds:
            min_accuracy = st.slider(
                "Minimum Accuracy",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
            
            max_loss = st.slider(
                "Maximum Loss",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.1
            )
    
    # Visualization Settings
    st.subheader("📈 Visualization Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎨 Chart Appearance")
        
        # Chart style
        chart_style = st.selectbox(
            "Chart Style",
            ["Default", "Dark", "Minimal", "Colorful"],
            index=0
        )
        
        # Color scheme
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Viridis", "Plasma", "Turbo", "Rainbow", "Set1"],
            index=0
        )
        
        # Chart height
        chart_height = st.number_input(
            "Default Chart Height (px)",
            min_value=300,
            max_value=800,
            value=500,
            step=50
        )
        
        # Show grid
        show_grid = st.checkbox("Show Grid Lines", value=True)
        
        # Show legend
        show_legend = st.checkbox("Show Legend", value=True)
    
    with col2:
        st.markdown("#### 📊 Data Display")
        
        # Number of models to show
        max_models_display = st.number_input(
            "Maximum Models to Display",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        # Decimal places
        decimal_places = st.number_input(
            "Decimal Places for Metrics",
            min_value=2,
            max_value=6,
            value=4,
            step=1
        )
        
        # Sort order
        sort_by = st.selectbox(
            "Default Sort By",
            ["Accuracy (Desc)", "Loss (Asc)", "Date (Desc)", "Name (Asc)"],
            index=0
        )
        
        # Group by model type
        group_by_type = st.checkbox("Group by Model Type", value=False)
        
        # Show statistics
        show_statistics = st.checkbox("Show Summary Statistics", value=True)
    
    # Export Settings
    st.subheader("📤 Export Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 Export Options")
        
        # Export format
        export_format = st.selectbox(
            "Default Export Format",
            ["CSV", "JSON", "Excel", "PDF"],
            index=0
        )
        
        # Include charts in export
        include_charts = st.checkbox("Include Charts in Export", value=True)
        
        # Export filename pattern
        filename_pattern = st.text_input(
            "Filename Pattern",
            value="model_comparison_{timestamp}",
            help="Use {timestamp}, {date}, {metric} as placeholders"
        )
    
    with col2:
        st.markdown("#### 📊 Report Settings")
        
        # Auto-generate reports
        auto_reports = st.checkbox("Auto-generate Reports", value=False)
        
        if auto_reports:
            report_frequency = st.selectbox(
                "Report Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=1
            )
            
            report_recipients = st.text_area(
                "Email Recipients",
                placeholder="email1@example.com, email2@example.com",
                help="Comma-separated email addresses"
            )
        
        # Include raw data
        include_raw_data = st.checkbox("Include Raw Data in Reports", value=False)
    
    # Performance Settings
    st.subheader("⚡ Performance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🚀 Optimization")
        
        # Cache settings
        enable_caching = st.checkbox("Enable Data Caching", value=True)
        
        if enable_caching:
            cache_duration = st.number_input(
                "Cache Duration (minutes)",
                min_value=5,
                max_value=60,
                value=15,
                step=5
            )
        
        # Parallel processing
        enable_parallel = st.checkbox("Enable Parallel Processing", value=True)
        
        if enable_parallel:
            max_workers = st.number_input(
                "Max Worker Threads",
                min_value=1,
                max_value=8,
                value=4,
                step=1
            )
    
    with col2:
        st.markdown("#### 📊 Data Limits")
        
        # Memory management
        max_memory_usage = st.slider(
            "Max Memory Usage (%)",
            min_value=50,
            max_value=90,
            value=70,
            step=5
        )
        
        # Batch size for large datasets
        batch_size = st.number_input(
            "Batch Size for Large Datasets",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        # Timeout settings
        query_timeout = st.number_input(
            "Query Timeout (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            step=10
        )
    
    # Save Settings
    if st.button("💾 Save Advanced Settings", type="primary"):
        try:
            # Collect all settings
            advanced_settings = {
                'metrics': {
                    'selected_metrics': selected_metrics,
                    'metric_weights': metric_weights,
                    'normalization_method': normalization_method
                },
                'filtering': {
                    'model_type_filter': model_type_filter,
                    'use_date_filter': use_date_filter,
                    'enable_thresholds': enable_thresholds
                },
                'visualization': {
                    'chart_style': chart_style,
                    'color_scheme': color_scheme,
                    'chart_height': chart_height,
                    'show_grid': show_grid,
                    'show_legend': show_legend,
                    'max_models_display': max_models_display,
                    'decimal_places': decimal_places,
                    'sort_by': sort_by,
                    'group_by_type': group_by_type,
                    'show_statistics': show_statistics
                },
                'export': {
                    'export_format': export_format,
                    'include_charts': include_charts,
                    'filename_pattern': filename_pattern,
                    'auto_reports': auto_reports,
                    'include_raw_data': include_raw_data
                },
                'performance': {
                    'enable_caching': enable_caching,
                    'enable_parallel': enable_parallel,
                    'max_memory_usage': max_memory_usage,
                    'batch_size': batch_size,
                    'query_timeout': query_timeout
                }
            }
            
            # Add conditional settings
            if use_date_filter:
                advanced_settings['filtering'].update({
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat()
                })
            
            if enable_thresholds:
                advanced_settings['filtering'].update({
                    'min_accuracy': min_accuracy,
                    'max_loss': max_loss
                })
            
            if enable_caching:
                advanced_settings['performance']['cache_duration'] = cache_duration
            
            if enable_parallel:
                advanced_settings['performance']['max_workers'] = max_workers
            
            if auto_reports:
                advanced_settings['export'].update({
                    'report_frequency': report_frequency,
                    'report_recipients': report_recipients
                })
            
            # Save to session state
            st.session_state['comparison_advanced_settings'] = advanced_settings
            
            # Try to save to file
            try:
                import os
                import json
                
                config_dir = "config"
                os.makedirs(config_dir, exist_ok=True)
                
                config_file = os.path.join(config_dir, "comparison_advanced_settings.json")
                with open(config_file, 'w') as f:
                    json.dump(advanced_settings, f, indent=2)
                
                st.success("✅ Advanced settings saved successfully!")
                st.info("🔄 Settings will be applied to future comparisons.")
                
            except Exception as e:
                st.warning(f"⚠️ Settings saved to session but could not save to file: {str(e)}")
                st.success("✅ Advanced settings saved to current session!")
                
        except Exception as e:
            st.error(f"❌ Error saving advanced settings: {str(e)}")
    
    # Reset to Defaults
    if st.button("🔄 Reset to Defaults", type="secondary"):
        # Clear session state
        if 'comparison_advanced_settings' in st.session_state:
            del st.session_state['comparison_advanced_settings']
        
        try:
            # Remove config file
            config_file = "config/comparison_advanced_settings.json"
            if os.path.exists(config_file):
                os.remove(config_file)
        except:
            pass
        
        st.success("✅ Settings reset to defaults!")
        st.rerun()

# Demo functions for when no real data is available

def show_demo_comparison():
    """Show demo comparison when no models are available"""
    st.info("🎮 Showing demo model comparison")
    
    # Create demo data
    demo_models = pd.DataFrame({
        'name': ['DemoGAN_v1', 'DemoMLP_v1', 'DemoLSTM_v1', 'DemoGAN_v2'],
        'model_type': ['TABULAR_GAN', 'MLP', 'LSTM', 'TABULAR_GAN'],
        'accuracy': [0.85, 0.92, 0.88, 0.87],
        'loss': [0.15, 0.08, 0.12, 0.13],
        'created_at': pd.date_range('2024-01-01', periods=4)
    })
    
    # Display demo comparison
    st.dataframe(demo_models, use_container_width=True)
    
    # Demo charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(demo_models, x='name', y='accuracy', 
                    title="Demo Model Accuracy Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(demo_models, names='model_type', 
                    title="Demo Model Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

def show_demo_performance_comparison():
    """Show demo performance comparison"""
    st.info("🎮 Showing demo performance comparison")
    show_demo_comparison()

def show_demo_benchmarking():
    """Show demo benchmarking"""
    st.info("🎮 Showing demo benchmarking analysis")
    
    # Demo benchmark data
    demo_data = {
        'Model': ['DemoGAN_v1', 'DemoMLP_v1', 'DemoLSTM_v1'],
        'Accuracy': [0.85, 0.92, 0.88],
        'Benchmark Score': [85, 92, 88],
        'Rank': [3, 1, 2]
    }
    
    df = pd.DataFrame(demo_data)
    
    fig = px.bar(df, x='Model', y='Benchmark Score',
                title="Demo Benchmark Scores")
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df, use_container_width=True)

def show_demo_detailed_analysis():
    """Show demo detailed analysis"""
    st.info("🎮 Showing demo detailed analysis")
    
    # Create demo performance data
    demo_performance = generate_performance_analysis({'model_type': 'Demo'})
    if demo_performance:
        show_metrics_over_time(demo_performance)

def show_demo_reports():
    """Show demo custom reports"""
    st.info("🎮 Showing demo custom reports")
    show_demo_comparison()

def calculate_benchmark_scores(models, metric, benchmark_type, threshold):
    """Calculate benchmark scores for models"""
    if models.empty:
        return None
    
    benchmark_results = models.copy()
    
    if metric == "Accuracy" and 'accuracy' in models.columns:
        scores = models['accuracy'].fillna(0)
    elif metric == "Loss" and 'loss' in models.columns:
        scores = 1 / (1 + models['loss'].fillna(1))  # Convert loss to score
    else:
        # Generate random scores for demo
        scores = pd.Series(np.random.uniform(0.5, 0.95, len(models)))
    
    if benchmark_type == "Best Performance":
        max_score = scores.max()
        benchmark_results['benchmark_score'] = scores / max_score
    elif benchmark_type == "Industry Standard":
        # Use 0.8 as industry standard
        benchmark_results['benchmark_score'] = scores / 0.8
    elif benchmark_type == "Custom Threshold":
        benchmark_results['benchmark_score'] = scores / threshold
    
    return benchmark_results

def create_benchmark_chart(benchmark_results, metric):
    """Create benchmark visualization chart"""
    fig = px.bar(
        benchmark_results,
        x='name' if 'name' in benchmark_results.columns else benchmark_results.index,
        y='benchmark_score',
        color='model_type',
        title=f"Benchmark Scores - {metric}",
        labels={'benchmark_score': 'Benchmark Score', 'name': 'Model Name'}
    )
    
    # Add benchmark line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                  annotation_text="Benchmark Target")
    
    return fig

def create_performance_radar(comparison_data):
    """Create radar chart for performance comparison"""
    # This would create a radar chart for multiple metrics
    # For now, show a simple message
    st.info("📊 Radar chart for multi-metric comparison would be displayed here")

def prepare_comparison_metrics(comparison_data):
    """Prepare comparison metrics table"""
    # Select relevant columns for comparison
    base_cols = ['name', 'model_type', 'created_at'] if 'name' in comparison_data.columns else ['model_type', 'created_at']
    metric_cols = []
    
    for col in ['accuracy', 'loss', 'f1_score', 'precision', 'recall']:
        if col in comparison_data.columns:
            metric_cols.append(col)
    
    available_cols = [col for col in base_cols + metric_cols if col in comparison_data.columns]
    
    if available_cols:
        result_df = comparison_data[available_cols].copy()
        
        # Format datetime if present
        if 'created_at' in result_df.columns:
            result_df['created_at'] = pd.to_datetime(result_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        return result_df
    else:
        return pd.DataFrame({'Message': ['No comparison data available']})

if __name__ == "__main__":
    show_model_comparison_page()