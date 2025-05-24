#!/usr/bin/env python3
"""
Startup script for AI Vision Suite
This script handles initialization and launching of the Streamlit application.
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit', 'torch', 'torchvision', 'sklearn', 'numpy', 
        'pandas', 'matplotlib', 'seaborn', 'plotly', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies."""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def initialize_database():
    """Initialize the database."""
    print("Initializing database...")
    try:
        # Add the current directory to Python path
        sys.path.append(os.getcwd())
        
        from database.db_setup import DatabaseManager
        
        db_manager = DatabaseManager()
        db_manager.create_tables()
        print("✅ Database initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/samples",
        "models/saved",
        "models/checkpoints",
        "database",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created!")

def run_streamlit():
    """Launch the Streamlit application."""
    print("🚀 Starting AI Vision Suite...")
    print("📱 The application will open in your default web browser")
    print("🔗 Access URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 AI Vision Suite stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def main():
    """Main startup function."""
    print("🤖 AI Vision Suite - Startup Script")
    print("=" * 50)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        response = input("Would you like to install them automatically? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually:")
                print("pip install -r requirements.txt")
                return
        else:
            print("❌ Cannot proceed without required dependencies.")
            return
    else:
        print("✅ All dependencies are installed!")
    
    # Initialize database
    if not initialize_database():
        print("❌ Failed to initialize database.")
        return
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()
