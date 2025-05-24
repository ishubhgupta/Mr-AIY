"""
AI Vision Suite - Pages Module
This package contains all the Streamlit pages for the AI Vision Suite application.
"""

__version__ = "1.0.0"

# Import all page modules
from . import cnn_page
from . import rcnn_page
from . import neural_networks_page
from . import gan_page
from . import clustering_page

# Import all page functions for easy access
from .gan_page import show_gan_page
from .neural_networks_page import show_neural_networks_page

try:
    from .data_management_page import show_data_management_page
    from .model_comparison_page import show_model_comparison_page  
    from .settings_page import show_settings_page
    
    __all__ = [
        'cnn_page',
        'rcnn_page',
        'neural_networks_page',
        'gan_page',
        'clustering_page',
        'show_gan_page',
        'show_neural_networks_page', 
        'show_data_management_page',
        'show_model_comparison_page',
        'show_settings_page'
    ]
except ImportError:
    __all__ = [
        'cnn_page',
        'rcnn_page',
        'neural_networks_page',
        'gan_page',
        'clustering_page',
        'show_gan_page',
        'show_neural_networks_page'
    ]
