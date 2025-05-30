"""
Main application file for AI Vision Suite
"""

import streamlit as st
import logging.config
from config import LOGGING, PRODUCTION_SETTINGS, ENVIRONMENT

# Configure logging
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)

# Configure Streamlit settings for production
for key, value in PRODUCTION_SETTINGS.items():
    st.set_option(key, value)

# Import pages after configuration
from code.pages.cnn_page import show_cnn_page
from code.pages.rcnn_page import show_rcnn_page
from code.pages.neural_networks_page import show_neural_networks_page
from code.pages.gan_page import show_gan_page
from code.pages.clustering_page import show_clustering_page
from code.pages.data_management_page import show_data_management_page
from code.pages.model_comparison_page import show_model_comparison_page
from code.pages.settings_page import show_settings_page

try:
    # Initialize database
    from database import DatabaseManager
    db = DatabaseManager()
    db.create_tables()
except Exception as e:
    logger.error(f"Database initialization failed: {e}")
    st.error("Failed to initialize database. Please check the logs.")

def main():
    try:
        # Page configuration
        st.set_page_config(
            page_title="AI Vision Suite",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS (moved to external file)
        with open('static/styles.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        # Sidebar navigation
        st.sidebar.title("🤖 AI Vision Suite")
        
        # Navigation
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
        
        page = st.sidebar.selectbox("Choose a module:", list(pages.keys()))

        # Display selected page
        try:
            if pages[page] == "home":
                show_home()
            elif pages[page] == "dashboard":
                show_dashboard()
            elif pages[page] == "cnn":
                show_cnn_page()
            elif pages[page] == "rcnn":
                show_rcnn_page()
            elif pages[page] == "neural_networks":
                show_neural_networks_page()
            elif pages[page] == "gan":
                show_gan_page()
            elif pages[page] == "clustering":
                show_clustering_page()
            elif pages[page] == "data_management":
                show_data_management_page()
            elif pages[page] == "model_comparison":
                show_model_comparison_page()
            elif pages[page] == "settings":
                show_settings_page()
        except Exception as e:
            logger.error(f"Error displaying page {page}: {e}")
            st.error("An error occurred while displaying this page. Please try again or contact support.")

    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()