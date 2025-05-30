"""
Configuration settings for AI Vision Suite
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Database
DB_PATH = os.getenv('DB_PATH', os.path.join(BASE_DIR, 'database', 'ai_vision_suite.db'))

# Application settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Security
SECRET_KEY = os.getenv('SECRET_KEY', 'generate-a-secure-key-in-production')

# Storage paths
DATA_DIR = os.getenv('DATA_DIR', os.path.join(BASE_DIR, 'data'))
MODELS_DIR = os.getenv('MODELS_DIR', os.path.join(BASE_DIR, 'models'))
LOGS_DIR = os.getenv('LOGS_DIR', os.path.join(BASE_DIR, 'logs'))

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': LOG_LEVEL,
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOGS_DIR, 'app.log'),
            'formatter': 'standard',
        },
        'console': {
            'level': LOG_LEVEL,
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': LOG_LEVEL,
            'propagate': True,
        },
    },
}

# Production settings
PRODUCTION_SETTINGS = {
    'server.address': '0.0.0.0',
    'server.port': int(os.getenv('PORT', 8501)),
    'server.maxUploadSize': 200,  # MB
    'server.maxMessageSize': 200,  # MB
    'server.enableCORS': False,
    'server.enableXsrfProtection': True,
    'browser.serverAddress': os.getenv('SERVER_ADDRESS', 'localhost'),
    'browser.gatherUsageStats': False,
}

# Cache settings
CACHE_TTL = 3600  # 1 hour
CACHE_TYPE = "filesystem"
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)