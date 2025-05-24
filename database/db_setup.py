"""
Database setup script for AI Vision Suite application.
This script sets up the necessary database structure for storing models, datasets, 
training records, and prediction results.
"""

import os
import sqlite3
from datetime import datetime
import json
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path="database/ai_vision_suite.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """
        Connect to the database.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def close(self):
        """
        Close the database connection.
        """
        if self.conn:
            self.conn.close()
            
    def initialize_database(self):
        """
        Initialize the database by creating all necessary tables.
        """
        self.connect()
        
        # Create Models table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            parameters TEXT,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accuracy REAL,
            loss REAL,
            description TEXT
        )
        ''')
        
        # Create Datasets table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            path TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
        ''')
        
        # Create TrainingRecords table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            dataset_id INTEGER,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            epochs INTEGER,
            batch_size INTEGER,
            learning_rate REAL,
            training_accuracy REAL,
            validation_accuracy REAL,
            loss REAL,
            parameters TEXT,
            FOREIGN KEY (model_id) REFERENCES models (id),
            FOREIGN KEY (dataset_id) REFERENCES datasets (id)
        )
        ''')
        
        # Create Predictions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER,
            input_data TEXT,
            output_result TEXT,
            confidence REAL,
            prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models (id)
        )
        ''')
        
        # Create User Settings table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            setting_name TEXT UNIQUE,
            setting_value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.conn.commit()
        self.close()
        
    def add_model(self, name, model_type, file_path, parameters=None, accuracy=None, loss=None, description=None):
        """
        Add a new model to the database.
        
        Args:
            name (str): Model name
            model_type (str): Type of model (CNN, RCNN, GAN, etc.)
            file_path (str): Path to the saved model file
            parameters (dict, optional): Model parameters
            accuracy (float, optional): Model accuracy
            loss (float, optional): Model loss
            description (str, optional): Model description
        
        Returns:
            int: ID of the newly added model
        """
        self.connect()
        
        if parameters is not None and isinstance(parameters, dict):
            parameters = json.dumps(parameters)
            
        self.cursor.execute('''
        INSERT INTO models (name, type, file_path, parameters, accuracy, loss, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, model_type, file_path, parameters, accuracy, loss, description))
        
        model_id = self.cursor.lastrowid
        self.conn.commit()
        self.close()
        
        return model_id
    
    def add_dataset(self, name, dataset_type, path, description=None, metadata=None):
        """
        Add a new dataset to the database.
        
        Args:
            name (str): Dataset name
            dataset_type (str): Type of dataset (image, tabular, etc.)
            path (str): Path to the dataset
            description (str, optional): Dataset description
            metadata (dict, optional): Dataset metadata
            
        Returns:
            int: ID of the newly added dataset
        """
        self.connect()
        
        if metadata is not None and isinstance(metadata, dict):
            metadata = json.dumps(metadata)
            
        self.cursor.execute('''
        INSERT INTO datasets (name, type, path, description, metadata)
        VALUES (?, ?, ?, ?, ?)
        ''', (name, dataset_type, path, description, metadata))
        
        dataset_id = self.cursor.lastrowid
        self.conn.commit()
        self.close()
        
        return dataset_id
    
    def add_training_record(self, model_id, dataset_id, start_time, end_time, epochs, 
                           batch_size, learning_rate, training_accuracy=None, 
                           validation_accuracy=None, loss=None, parameters=None):
        """
        Add a new training record to the database.
        
        Args:
            model_id (int): ID of the model
            dataset_id (int): ID of the dataset
            start_time (datetime): Training start time
            end_time (datetime): Training end time
            epochs (int): Number of epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            training_accuracy (float, optional): Training accuracy
            validation_accuracy (float, optional): Validation accuracy
            loss (float, optional): Final loss
            parameters (dict, optional): Additional training parameters
            
        Returns:
            int: ID of the newly added training record
        """
        self.connect()
        
        if parameters is not None and isinstance(parameters, dict):
            parameters = json.dumps(parameters)
            
        self.cursor.execute('''
        INSERT INTO training_records (model_id, dataset_id, start_time, end_time, 
                                     epochs, batch_size, learning_rate, 
                                     training_accuracy, validation_accuracy, 
                                     loss, parameters)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_id, dataset_id, start_time, end_time, epochs, batch_size, 
             learning_rate, training_accuracy, validation_accuracy, loss, parameters))
        
        record_id = self.cursor.lastrowid
        self.conn.commit()
        self.close()
        
        return record_id
    
    def add_prediction(self, model_id, input_data, output_result, confidence=None):
        """
        Add a new prediction record to the database.
        
        Args:
            model_id (int): ID of the model
            input_data (str): Input data or path to input data
            output_result (str): Prediction result
            confidence (float, optional): Prediction confidence
            
        Returns:
            int: ID of the newly added prediction record
        """
        self.connect()
        
        self.cursor.execute('''
        INSERT INTO predictions (model_id, input_data, output_result, confidence)
        VALUES (?, ?, ?, ?)
        ''', (model_id, input_data, output_result, confidence))
        
        prediction_id = self.cursor.lastrowid
        self.conn.commit()
        self.close()
        
        return prediction_id
    
    def get_models(self, model_type=None):
        """
        Get all models or models of a specific type.
        
        Args:
            model_type (str, optional): Type of models to retrieve
            
        Returns:
            list: List of model records
        """
        self.connect()
        
        if model_type:
            self.cursor.execute('SELECT * FROM models WHERE type = ?', (model_type,))
        else:
            self.cursor.execute('SELECT * FROM models')
            
        models = self.cursor.fetchall()
        self.close()
        
        return models
    
    def get_datasets(self, dataset_type=None):
        """
        Get all datasets or datasets of a specific type.
        
        Args:
            dataset_type (str, optional): Type of datasets to retrieve
            
        Returns:
            list: List of dataset records
        """
        self.connect()
        
        if dataset_type:
            self.cursor.execute('SELECT * FROM datasets WHERE type = ?', (dataset_type,))
        else:
            self.cursor.execute('SELECT * FROM datasets')
            
        datasets = self.cursor.fetchall()
        self.close()
        
        return datasets
    
    def update_setting(self, name, value):
        """
        Update a user setting.
        
        Args:
            name (str): Setting name
            value (str): Setting value
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.connect()
        
        try:
            self.cursor.execute('''
            INSERT OR REPLACE INTO user_settings (setting_name, setting_value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (name, value))
            
            self.conn.commit()
            self.close()
            return True
        except:
            self.close()
            return False
    
    def get_setting(self, name):
        """
        Get a user setting.
        
        Args:
            name (str): Setting name
            
        Returns:
            str: Setting value or None if not found
        """
        self.connect()
        
        self.cursor.execute('SELECT setting_value FROM user_settings WHERE setting_name = ?', (name,))
        result = self.cursor.fetchone()
        
        self.close()
        
        if result:
            return result[0]
        return None

    # Additional methods for dashboard functionality
    def get_total_models(self):
        """Get total number of models."""
        self.connect()
        self.cursor.execute('SELECT COUNT(*) FROM models')
        count = self.cursor.fetchone()[0]
        self.close()
        return count
    
    def get_total_predictions(self):
        """Get total number of predictions."""
        self.connect()
        self.cursor.execute('SELECT COUNT(*) FROM predictions')
        count = self.cursor.fetchone()[0]
        self.close()
        return count
    
    def get_total_datasets(self):
        """Get total number of datasets."""
        self.connect()
        self.cursor.execute('SELECT COUNT(*) FROM datasets')
        count = self.cursor.fetchone()[0]
        self.close()
        return count
    
    def get_active_models(self):
        """Get number of active models (assuming active means created in last 30 days)."""
        self.connect()
        self.cursor.execute('''
            SELECT COUNT(*) FROM models 
            WHERE created_at > datetime('now', '-30 days')
        ''')
        count = self.cursor.fetchone()[0]
        self.close()
        return count
    
    def get_models_by_type(self, model_type):
        """Get all models of a specific type"""
        query = """
        SELECT * FROM models 
        WHERE type = ?
        ORDER BY created_at DESC
        """
        self.connect()
        try:
            df = pd.read_sql_query(query, self.conn, params=(model_type,))
            return df
        except Exception as e:
            print(f"Error getting models by type: {e}")
            return pd.DataFrame()
        finally:
            self.close()

    def get_models_summary(self):
        """Get summary of all models"""
        query = """
        SELECT type as model_type, COUNT(*) as count, AVG(accuracy) as avg_accuracy
        FROM models 
        GROUP BY type
        """
        self.connect()
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"Error getting models summary: {e}")
            return pd.DataFrame()
        finally:
            self.close()

    def get_training_history(self):
        """Get training history data"""
        query = """
        SELECT type as model_type, accuracy, created_at
        FROM models 
        WHERE accuracy IS NOT NULL
        ORDER BY created_at DESC
        """
        self.connect()
        try:
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            print(f"Error getting training history: {e}")
            return pd.DataFrame()
        finally:
            self.close()

    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        query = """
        SELECT * FROM predictions 
        ORDER BY prediction_time DESC 
        LIMIT ?
        """
        self.connect()
        try:
            df = pd.read_sql_query(query, self.conn, params=(limit,))
            return df
        except Exception as e:
            print(f"Error getting recent predictions: {e}")
            return pd.DataFrame()
        finally:
            self.close()
    
    def get_model_details(self, model_id):
        """Get detailed information about a specific model."""
        self.connect()
        
        self.cursor.execute('''
        SELECT * FROM models WHERE id = ?
        ''', (model_id,))
        
        model = self.cursor.fetchone()
        
        if model:
            # Get column names
            column_names = [description[0] for description in self.cursor.description]
            model_dict = dict(zip(column_names, model))
            
            # Get training records for this model
            self.cursor.execute('''
            SELECT * FROM training_records WHERE model_id = ?
            ORDER BY created_at DESC
            ''', (model_id,))
            
            training_records = self.cursor.fetchall()
            if training_records:
                training_column_names = [description[0] for description in self.cursor.description]
                model_dict['training_history'] = [
                    dict(zip(training_column_names, record)) 
                    for record in training_records
                ]
            
            self.close()
            return model_dict
        
        self.close()
        return None
    
    def delete_model(self, model_id):
        """Delete a model and all associated records."""
        self.connect()
        
        try:
            # Delete training records
            self.cursor.execute('DELETE FROM training_records WHERE model_id = ?', (model_id,))
            
            # Delete predictions
            self.cursor.execute('DELETE FROM predictions WHERE model_id = ?', (model_id,))
            
            # Delete the model
            self.cursor.execute('DELETE FROM models WHERE id = ?', (model_id,))
            
            self.conn.commit()
            self.close()
            return True
        except Exception as e:
            self.close()
            return False

    def create_tables(self):
        """Create all necessary tables (alias for initialize_database)."""
        self.initialize_database()
        

if __name__ == "__main__":
    # Initialize the database
    db_manager = DatabaseManager()
    db_manager.initialize_database()
    
    print("Database initialized successfully!")
