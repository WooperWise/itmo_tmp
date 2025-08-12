#!/usr/bin/env python3
"""
Fraud Transaction Statistical Analysis - Data Exploration Framework

This module provides a comprehensive Python environment setup and data exploration 
framework for fraud transaction statistical analysis.

Author: Statistical Analysis Framework
Date: 2025-01-11
Version: 1.0.0
"""

# Core data manipulation and analysis libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis libraries
import scipy.stats as stats
from scipy.stats import normaltest, shapiro, kstest, jarque_bera, chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors
import pingouin as pg
from diptest import diptest

# Machine learning libraries
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Rectangle

# Date and time handling
from datetime import datetime, timedelta
import pytz

# System and utility libraries
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json

class FraudDataExplorer:
    """
    Comprehensive fraud data exploration and preprocessing framework
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the fraud data explorer
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Set display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        print("✅ FraudDataExplorer initialized successfully!")
        self._print_library_versions()
    
    def _print_library_versions(self):
        """Print versions of key libraries"""
        print(f"📊 Pandas version: {pd.__version__}")
        print(f"🔢 NumPy version: {np.__version__}")
        print(f"📈 SciPy version: {stats.__version__ if hasattr(stats, '__version__') else 'Available'}")
        print(f"🎨 Matplotlib version: {plt.matplotlib.__version__}")
        print(f"🌊 Seaborn version: {sns.__version__}")
    
    def create_sample_fraud_data(self, n_samples: int = 10000, fraud_rate: float = 0.05) -> pd.DataFrame:
        """
        Create a sample fraud transaction dataset matching the schema described in README.md
        
        Parameters:
        -----------
        n_samples : int
            Number of transactions to generate
        fraud_rate : float
            Proportion of fraudulent transactions (0-1)
        
        Returns:
        --------
        pd.DataFrame
            Generated fraud transaction dataset
        """
        print(f"🔧 Creating sample fraud dataset with {n_samples} transactions...")
        
        # Generate base data
        data = {
            'transaction_id': [f'TXN_{i:08d}' for i in range(n_samples)],
            'customer_id': [f'CUST_{np.random.randint(1, n_samples//5):06d}' for _ in range(n_samples)],
            'card_number': np.random.randint(1000000000000000, 9999999999999999, n_samples),
            'timestamp': pd.date_range('2024-09-30', '2024-10-30', periods=n_samples),
            'vendor_category': np.random.choice(['Retail', 'Travel', 'Entertainment', 'Healthcare', 'Education', 'Fuel', 'Restaurant'], n_samples),
            'vendor_type': np.random.choice(['online', 'offline', 'premium', 'fastfood'], n_samples),
            'vendor': [f'Vendor_{np.random.randint(1, 1000):03d}' for _ in range(n_samples)],
            'amount': np.random.lognormal(3, 1.5, n_samples),  # Log-normal distribution for realistic amounts
            'currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD'], n_samples, p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1]),
            'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Japan', 'Canada', 'Australia'], n_samples),
            'city': [f'City_{np.random.randint(1, 500):03d}' for _ in range(n_samples)],
            'city_size': np.random.choice(['small', 'medium', 'large'], n_samples, p=[0.3, 0.4, 0.3]),
            'card_type': np.random.choice(['Basic Credit', 'Gold Credit', 'Platinum Credit', 'Debit'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            'is_card_present': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
            'device': np.random.choice(['Chrome', 'Safari', 'iOS App', 'Android App', 'Firefox'], n_samples),
            'channel': np.random.choice(['web', 'mobile', 'pos'], n_samples, p=[0.4, 0.3, 0.3]),
            'device_fingerprint': [f'FP_{np.random.randint(1000000, 9999999):07d}' for _ in range(n_samples)],
            'ip_address': [f'{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}' for _ in range(n_samples)],
            'is_outside_home_country': np.random.choice([True, False], n_samples, p=[0.15, 0.85]),
            'is_high_risk_vendor': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
            'is_weekend': np.random.choice([True, False], n_samples, p=[0.3, 0.7])
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add last_hour_activity as nested structure
        df['last_hour_activity'] = [
            {
                'num_transactions': np.random.randint(1, 10),
                'total_amount': np.random.uniform(10, 1000),
                'unique_merchants': np.random.randint(1, 5),
                'unique_countries': np.random.randint(1, 3),
                'max_single_amount': np.random.uniform(50, 500)
            } for _ in range(n_samples)
        ]
        
        # Generate fraud labels with realistic patterns
        fraud_indices = np.random.choice(n_samples, int(n_samples * fraud_rate), replace=False)
        df['is_fraud'] = False
        df.loc[fraud_indices, 'is_fraud'] = True
        
        # Make fraudulent transactions more realistic
        # Higher amounts for fraud
        df.loc[df['is_fraud'], 'amount'] *= np.random.uniform(2, 5, sum(df['is_fraud']))
        # More likely to be outside home country
        df.loc[df['is_fraud'], 'is_outside_home_country'] = np.random.choice([True, False], sum(df['is_fraud']), p=[0.7, 0.3])
        # More likely to be high risk vendor
        df.loc[df['is_fraud'], 'is_high_risk_vendor'] = np.random.choice([True, False], sum(df['is_fraud']), p=[0.8, 0.2])
        
        print(f"✅ Sample dataset created successfully! Fraud rate: {df['is_fraud'].mean():.4f}")
        return df
    
    def load_fraud_data(self, file_path: str = 'transaction_fraud_data.parquet') -> pd.DataFrame:
        """
        Load fraud transaction data from parquet file or create sample data if file doesn't exist
        
        Parameters:
        -----------
        file_path : str
            Path to the parquet file
        
        Returns:
        --------
        pd.DataFrame
            Fraud transaction dataset
        """
        if os.path.exists(file_path):
            print(f"📁 Loading data from {file_path}...")
            df = pd.read_parquet(file_path)
            print(f"✅ Data loaded successfully! Shape: {df.shape}")
        else:
            print(f"⚠️  File {file_path} not found. Creating sample data...")
            df = self.create_sample_fraud_data()
            # Save the sample data
            df.to_parquet(file_path, index=False)
            print(f"✅ Sample data created and saved to {file_path}! Shape: {df.shape}")
        
        return df
    
    def basic_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Generate basic information about the dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        
        Returns:
        --------
        Dict
            Dictionary containing basic dataset information
        """
        # Handle duplicate detection carefully for columns with unhashable types
        try:
            # Exclude columns with complex data types from duplicate detection
            simple_cols = []
            for col in df.columns:
                if df[col].dtype in ['object']:
                    # Check if column contains dictionaries or other unhashable types
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if not isinstance(sample_val, (dict, list)):
                        simple_cols.append(col)
                else:
                    simple_cols.append(col)
            
            duplicate_count = df[simple_cols].duplicated().sum() if simple_cols else 0
        except Exception as e:
            print(f"⚠️  Could not calculate duplicates: {e}")
            duplicate_count = 0
        
        info = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': duplicate_count,
            'fraud_rate': df['is_fraud'].mean() if 'is_fraud' in df.columns else None,
            'date_range': (df['timestamp'].min(), df['timestamp'].max()) if 'timestamp' in df.columns else None
        }
        
        return info
    
    def explore_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive missing value analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        
        Returns:
        --------
        pd.DataFrame
            Missing value summary
        """
        # Calculate unique values safely for each column
        unique_counts = []
        for col in df.columns:
            try:
                if df[col].dtype == 'object':
                    # Check if column contains dictionaries or other unhashable types
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if isinstance(sample_val, (dict, list)):
                        unique_counts.append(-1)  # Use -1 to indicate complex data type
                    else:
                        unique_counts.append(df[col].nunique())
                else:
                    unique_counts.append(df[col].nunique())
            except Exception:
                unique_counts.append(-1)  # Use -1 for any problematic columns
        
        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
            'Data_Type': df.dtypes,
            'Non_Null_Count': df.count(),
            'Unique_Values': unique_counts
        })
        
        missing_summary = missing_summary.sort_values('Missing_Percentage', ascending=False)
        missing_summary = missing_summary.reset_index(drop=True)
        
        return missing_summary
    
    def analyze_categorical_variables(self, df: pd.DataFrame) -> Dict:
        """
        Analyze categorical variables in the dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        
        Returns:
        --------
        Dict
            Analysis results for categorical variables
        """
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        analysis = {}
        
        for col in categorical_cols:
            # Skip ID columns and columns with complex data types
            if col not in ['transaction_id', 'customer_id', 'device_fingerprint', 'ip_address', 'last_hour_activity']:
                try:
                    # Check if column contains dictionaries or other unhashable types
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if isinstance(sample_val, (dict, list)):
                        continue  # Skip columns with complex data types
                    
                    analysis[col] = {
                        'unique_count': df[col].nunique(),
                        'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                        'most_frequent_count': df[col].value_counts().iloc[0] if not df[col].empty else 0,
                        'value_counts': df[col].value_counts().head(10).to_dict(),
                        'fraud_rate_by_category': df.groupby(col)['is_fraud'].mean().to_dict() if 'is_fraud' in df.columns else None
                    }
                except Exception as e:
                    print(f"⚠️  Skipping column {col} due to error: {e}")
                    continue
        
        return analysis
    
    def analyze_numerical_variables(self, df: pd.DataFrame) -> Dict:
        """
        Analyze numerical variables in the dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        
        Returns:
        --------
        Dict
            Analysis results for numerical variables
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        analysis = {}
        
        for col in numerical_cols:
            if col not in ['card_number']:  # Skip card numbers
                col_data = df[col].dropna()
                
                analysis[col] = {
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': col_data.quantile(0.25),
                    'q75': col_data.quantile(0.75),
                    'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'outliers_iqr': len(col_data[(col_data < col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))) | 
                                                (col_data > col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)))]),
                    'fraud_correlation': df[col].corr(df['is_fraud']) if 'is_fraud' in df.columns else None
                }
        
        return analysis
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp column
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset with timestamp column
        
        Returns:
        --------
        pd.DataFrame
            Dataset with additional temporal features
        """
        df_temp = df.copy()
        
        if 'timestamp' in df_temp.columns:
            # Ensure timestamp is datetime
            df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
            
            # Extract temporal features
            df_temp['hour'] = df_temp['timestamp'].dt.hour
            df_temp['day_of_week'] = df_temp['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
            df_temp['day_of_month'] = df_temp['timestamp'].dt.day
            df_temp['month'] = df_temp['timestamp'].dt.month
            df_temp['is_weekend_extracted'] = df_temp['day_of_week'].isin([5, 6])  # Saturday, Sunday
            df_temp['is_business_hours'] = df_temp['hour'].between(9, 17)  # 9 AM to 5 PM
            df_temp['is_night_time'] = df_temp['hour'].between(22, 6)  # 10 PM to 6 AM
            
            # Time-based bins
            df_temp['hour_bin'] = pd.cut(df_temp['hour'], bins=[0, 6, 12, 18, 24], 
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening'], 
                                       include_lowest=True)
            
            print("✅ Temporal features extracted successfully!")
        else:
            print("⚠️  No timestamp column found!")
        
        return df_temp
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict:
        """
        Detect outliers in numerical columns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        method : str
            Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
        
        Returns:
        --------
        Dict
            Outlier detection results
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_results = {}
        
        for col in numerical_cols:
            if col not in ['card_number']:
                col_data = df[col].dropna()
                
                if method == 'iqr':
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(col_data))
                    outliers = col_data[z_scores > 3]
                    
                elif method == 'isolation_forest':
                    iso_forest = IsolationForest(contamination=0.1, random_state=self.random_seed)
                    outlier_labels = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                    outliers = col_data[outlier_labels == -1]
                
                outlier_results[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(col_data)) * 100,
                    'outlier_indices': outliers.index.tolist()
                }
        
        return outlier_results
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data quality report
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        
        Returns:
        --------
        Dict
            Comprehensive data quality report
        """
        print("📊 Generating comprehensive data quality report...")
        
        report = {
            'basic_info': self.basic_data_info(df),
            'missing_values': self.explore_missing_values(df),
            'categorical_analysis': self.analyze_categorical_variables(df),
            'numerical_analysis': self.analyze_numerical_variables(df),
            'outlier_analysis': self.detect_outliers(df),
            'recommendations': []
        }
        
        # Generate recommendations
        if report['basic_info']['missing_values'] > 0:
            report['recommendations'].append("Consider implementing missing value imputation strategies")
        
        if report['basic_info']['duplicate_rows'] > 0:
            report['recommendations'].append("Remove duplicate rows to ensure data integrity")
        
        # Check for high cardinality categorical variables
        for col, analysis in report['categorical_analysis'].items():
            if analysis['unique_count'] > len(df) * 0.8:
                report['recommendations'].append(f"High cardinality in {col} - consider feature engineering")
        
        # Check for highly skewed numerical variables
        for col, analysis in report['numerical_analysis'].items():
            if abs(analysis['skewness']) > 2:
                report['recommendations'].append(f"High skewness in {col} - consider transformation")
        
        print("✅ Data quality report generated successfully!")
        return report
    
    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> Dict:
        """
        Create a robust data preprocessing pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        
        Returns:
        --------
        Dict
            Preprocessing pipeline components
        """
        print("🔧 Creating preprocessing pipeline...")
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Remove ID columns from preprocessing
        id_cols = ['transaction_id', 'customer_id', 'device_fingerprint', 'ip_address', 'card_number']
        numerical_cols = [col for col in numerical_cols if col not in id_cols]
        categorical_cols = [col for col in categorical_cols if col not in id_cols]
        
        pipeline = {
            'numerical_preprocessing': {
                'columns': numerical_cols,
                'imputer': SimpleImputer(strategy='median'),
                'scaler': RobustScaler(),  # Robust to outliers
                'outlier_detector': IsolationForest(contamination=0.1, random_state=self.random_seed)
            },
            'categorical_preprocessing': {
                'columns': categorical_cols,
                'imputer': SimpleImputer(strategy='most_frequent'),
                'encoder': 'one_hot'  # Can be changed to label encoding if needed
            },
            'feature_engineering': {
                'temporal_features': True,
                'interaction_features': False,  # Can be enabled for advanced analysis
                'polynomial_features': False   # Can be enabled for advanced analysis
            }
        }
        
        print("✅ Preprocessing pipeline created successfully!")
        return pipeline


def main():
    """
    Main function to demonstrate the fraud data exploration framework
    """
    print("🚀 Starting Fraud Transaction Statistical Analysis Framework")
    print("=" * 60)
    
    # Initialize the explorer
    explorer = FraudDataExplorer(random_seed=42)
    
    # Load or create fraud data
    df_fraud = explorer.load_fraud_data()
    
    # Basic data information
    basic_info = explorer.basic_data_info(df_fraud)
    print("\n📊 Dataset Basic Information:")
    print(f"Shape: {basic_info['shape']}")
    print(f"Memory Usage: {basic_info['memory_usage_mb']:.2f} MB")
    print(f"Missing Values: {basic_info['missing_values']}")
    print(f"Duplicate Rows: {basic_info['duplicate_rows']}")
    print(f"Fraud Rate: {basic_info['fraud_rate']:.4f} ({basic_info['fraud_rate']*100:.2f}%)")
    print(f"Date Range: {basic_info['date_range'][0]} to {basic_info['date_range'][1]}")
    
    # Extract temporal features
    df_fraud_enhanced = explorer.extract_temporal_features(df_fraud)
    
    # Generate comprehensive data quality report
    quality_report = explorer.generate_data_quality_report(df_fraud_enhanced)
    
    # Display key findings
    print("\n🔍 Key Data Quality Findings:")
    print(f"Missing values found in {len(quality_report['missing_values'][quality_report['missing_values']['Missing_Count'] > 0])} columns")
    print(f"Analyzed {len(quality_report['categorical_analysis'])} categorical variables")
    print(f"Analyzed {len(quality_report['numerical_analysis'])} numerical variables")
    
    # Display recommendations
    print("\n💡 Data Quality Recommendations:")
    for i, rec in enumerate(quality_report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Create preprocessing pipeline
    pipeline = explorer.create_preprocessing_pipeline(df_fraud_enhanced)
    print(f"\n🔧 Preprocessing Pipeline Created:")
    print(f"Numerical columns: {len(pipeline['numerical_preprocessing']['columns'])}")
    print(f"Categorical columns: {len(pipeline['categorical_preprocessing']['columns'])}")
    
    print("\n✅ Fraud data exploration framework setup completed successfully!")
    print("📝 Ready for statistical hypothesis testing implementation.")
    
    return df_fraud_enhanced, quality_report, pipeline


if __name__ == "__main__":
    df, report, pipeline = main()