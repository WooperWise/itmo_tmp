#!/usr/bin/env python3
"""
Complete Fraud Transaction Statistical Analysis Demo

This script demonstrates the complete fraud data exploration framework including:
- Data loading and exploration
- Statistical analysis
- Data quality assessment
- Comprehensive visualizations
- Preprocessing pipeline setup

Author: Statistical Analysis Framework
Date: 2025-01-11
Version: 1.0.0
"""

import sys
import os
from pathlib import Path

# Import our custom modules
from fraud_analysis_framework import FraudDataExplorer
from fraud_visualization import FraudVisualizationSuite

def main():
    """
    Complete demonstration of the fraud analysis framework
    """
    print("🚀 Complete Fraud Transaction Statistical Analysis Demo")
    print("=" * 70)
    
    # Initialize components
    explorer = FraudDataExplorer(random_seed=42)
    visualizer = FraudVisualizationSuite(figsize=(15, 10))
    
    print("\n📊 PHASE 1: Data Loading and Basic Exploration")
    print("-" * 50)
    
    # Load fraud data
    df_fraud = explorer.load_fraud_data('transaction_fraud_data.parquet')
    
    # Basic data information
    basic_info = explorer.basic_data_info(df_fraud)
    print(f"\n📋 Dataset Overview:")
    print(f"  • Shape: {basic_info['shape'][0]:,} rows × {basic_info['shape'][1]} columns")
    print(f"  • Memory Usage: {basic_info['memory_usage_mb']:.2f} MB")
    print(f"  • Missing Values: {basic_info['missing_values']:,}")
    print(f"  • Duplicate Rows: {basic_info['duplicate_rows']:,}")
    print(f"  • Fraud Rate: {basic_info['fraud_rate']:.4f} ({basic_info['fraud_rate']*100:.2f}%)")
    print(f"  • Date Range: {basic_info['date_range'][0]} to {basic_info['date_range'][1]}")
    
    print(f"\n📈 Data Types Distribution:")
    for dtype, count in basic_info['dtypes'].items():
        print(f"  • {dtype}: {count} columns")
    
    print("\n📊 PHASE 2: Feature Engineering and Enhancement")
    print("-" * 50)
    
    # Extract temporal features
    df_enhanced = explorer.extract_temporal_features(df_fraud)
    print(f"✅ Enhanced dataset shape: {df_enhanced.shape}")
    
    # Show new temporal features
    temporal_features = ['hour', 'day_of_week', 'day_of_month', 'month', 
                        'is_weekend_extracted', 'is_business_hours', 'is_night_time', 'hour_bin']
    existing_temporal = [col for col in temporal_features if col in df_enhanced.columns]
    print(f"📅 Temporal features added: {existing_temporal}")
    
    print("\n📊 PHASE 3: Comprehensive Data Quality Analysis")
    print("-" * 50)
    
    # Generate comprehensive data quality report
    quality_report = explorer.generate_data_quality_report(df_enhanced)
    
    # Display missing values analysis
    missing_analysis = quality_report['missing_values']
    missing_cols = missing_analysis[missing_analysis['Missing_Count'] > 0]
    if len(missing_cols) > 0:
        print(f"\n⚠️  Missing Values Found in {len(missing_cols)} columns:")
        for _, row in missing_cols.head().iterrows():
            print(f"  • {row['Column']}: {row['Missing_Count']:,} ({row['Missing_Percentage']:.2f}%)")
    else:
        print("✅ No missing values found in the dataset!")
    
    # Display categorical analysis summary
    cat_analysis = quality_report['categorical_analysis']
    print(f"\n📊 Categorical Variables Analysis ({len(cat_analysis)} variables):")
    for col, analysis in list(cat_analysis.items())[:5]:  # Show first 5
        print(f"  • {col}:")
        print(f"    - Unique values: {analysis['unique_count']:,}")
        print(f"    - Most frequent: {analysis['most_frequent']} ({analysis['most_frequent_count']:,} times)")
        if analysis['fraud_rate_by_category']:
            max_fraud_cat = max(analysis['fraud_rate_by_category'].items(), key=lambda x: x[1])
            print(f"    - Highest fraud rate: {max_fraud_cat[0]} ({max_fraud_cat[1]:.4f})")
    
    # Display numerical analysis summary
    num_analysis = quality_report['numerical_analysis']
    print(f"\n🔢 Numerical Variables Analysis ({len(num_analysis)} variables):")
    for col, analysis in num_analysis.items():
        print(f"  • {col}:")
        print(f"    - Mean: {analysis['mean']:.2f}, Median: {analysis['median']:.2f}")
        print(f"    - Std: {analysis['std']:.2f}, Skewness: {analysis['skewness']:.2f}")
        print(f"    - Outliers (IQR): {analysis['outliers_iqr']:,} ({(analysis['outliers_iqr']/analysis['count']*100):.2f}%)")
        if analysis['fraud_correlation'] is not None:
            print(f"    - Fraud correlation: {analysis['fraud_correlation']:.4f}")
    
    # Display outlier analysis
    outlier_analysis = quality_report['outlier_analysis']
    print(f"\n🎯 Outlier Analysis Summary:")
    total_outliers = sum(analysis['outlier_count'] for analysis in outlier_analysis.values())
    print(f"  • Total outliers detected: {total_outliers:,}")
    for col, analysis in outlier_analysis.items():
        if analysis['outlier_count'] > 0:
            print(f"  • {col}: {analysis['outlier_count']:,} outliers ({analysis['outlier_percentage']:.2f}%)")
    
    # Display recommendations
    recommendations = quality_report['recommendations']
    print(f"\n💡 Data Quality Recommendations ({len(recommendations)} items):")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n📊 PHASE 4: Statistical Insights")
    print("-" * 50)
    
    # Key statistical insights
    print("🔍 Key Statistical Insights:")
    
    # Fraud rate by key categories
    if 'vendor_category' in df_enhanced.columns:
        vendor_fraud = df_enhanced.groupby('vendor_category')['is_fraud'].mean().sort_values(ascending=False)
        print(f"  • Highest fraud rate by vendor category: {vendor_fraud.index[0]} ({vendor_fraud.iloc[0]:.4f})")
        print(f"  • Lowest fraud rate by vendor category: {vendor_fraud.index[-1]} ({vendor_fraud.iloc[-1]:.4f})")
    
    if 'country' in df_enhanced.columns:
        country_fraud = df_enhanced.groupby('country')['is_fraud'].mean().sort_values(ascending=False)
        print(f"  • Highest fraud rate by country: {country_fraud.index[0]} ({country_fraud.iloc[0]:.4f})")
        print(f"  • Lowest fraud rate by country: {country_fraud.index[-1]} ({country_fraud.iloc[-1]:.4f})")
    
    # Amount analysis
    legit_mean = df_enhanced[df_enhanced['is_fraud'] == False]['amount'].mean()
    fraud_mean = df_enhanced[df_enhanced['is_fraud'] == True]['amount'].mean()
    print(f"  • Average legitimate transaction: ${legit_mean:.2f}")
    print(f"  • Average fraudulent transaction: ${fraud_mean:.2f}")
    print(f"  • Fraud transactions are {fraud_mean/legit_mean:.1f}x larger on average")
    
    # Temporal patterns
    if 'hour' in df_enhanced.columns:
        hourly_fraud = df_enhanced.groupby('hour')['is_fraud'].mean()
        peak_hour = hourly_fraud.idxmax()
        low_hour = hourly_fraud.idxmin()
        print(f"  • Peak fraud hour: {peak_hour}:00 ({hourly_fraud[peak_hour]:.4f} fraud rate)")
        print(f"  • Lowest fraud hour: {low_hour}:00 ({hourly_fraud[low_hour]:.4f} fraud rate)")
    
    if 'is_weekend_extracted' in df_enhanced.columns:
        weekend_fraud = df_enhanced.groupby('is_weekend_extracted')['is_fraud'].mean()
        print(f"  • Weekend fraud rate: {weekend_fraud[True]:.4f}")
        print(f"  • Weekday fraud rate: {weekend_fraud[False]:.4f}")
    
    print("\n📊 PHASE 5: Preprocessing Pipeline Setup")
    print("-" * 50)
    
    # Create preprocessing pipeline
    pipeline = explorer.create_preprocessing_pipeline(df_enhanced)
    
    print("🔧 Preprocessing Pipeline Components:")
    print(f"  • Numerical columns: {len(pipeline['numerical_preprocessing']['columns'])}")
    print(f"    - Columns: {pipeline['numerical_preprocessing']['columns']}")
    print(f"    - Imputer: {type(pipeline['numerical_preprocessing']['imputer']).__name__}")
    print(f"    - Scaler: {type(pipeline['numerical_preprocessing']['scaler']).__name__}")
    
    print(f"  • Categorical columns: {len(pipeline['categorical_preprocessing']['columns'])}")
    print(f"    - Columns: {pipeline['categorical_preprocessing']['columns'][:5]}...")  # Show first 5
    print(f"    - Imputer: {type(pipeline['categorical_preprocessing']['imputer']).__name__}")
    print(f"    - Encoder: {pipeline['categorical_preprocessing']['encoder']}")
    
    print(f"  • Feature engineering options:")
    for feature, enabled in pipeline['feature_engineering'].items():
        print(f"    - {feature}: {'✅' if enabled else '❌'}")
    
    print("\n📊 PHASE 6: Visualization Generation")
    print("-" * 50)
    
    # Generate sample visualizations (without showing them in demo)
    print("🎨 Generating comprehensive visualizations...")
    
    # Create visualizations directory
    viz_dir = "fraud_analysis_visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        # Generate all visualizations
        visualizer.generate_all_visualizations(df_enhanced, viz_dir)
        print(f"✅ All visualizations saved to {viz_dir}/")
        
        # List generated files
        viz_files = list(Path(viz_dir).glob("*.png"))
        print(f"📁 Generated visualization files:")
        for file in viz_files:
            print(f"  • {file.name}")
            
    except Exception as e:
        print(f"⚠️  Visualization generation encountered an issue: {e}")
        print("📊 Visualizations can be generated manually using the FraudVisualizationSuite")
    
    print("\n📊 PHASE 7: Summary and Next Steps")
    print("-" * 50)
    
    print("✅ Fraud Data Exploration Framework Setup Complete!")
    print("\n📋 Summary:")
    print(f"  • Dataset: {basic_info['shape'][0]:,} transactions with {basic_info['shape'][1]} features")
    print(f"  • Fraud rate: {basic_info['fraud_rate']*100:.2f}%")
    print(f"  • Data quality: {'Excellent' if len(recommendations) <= 2 else 'Good' if len(recommendations) <= 5 else 'Needs attention'}")
    missing_val_text = 'None' if basic_info['missing_values'] == 0 else f"{basic_info['missing_values']:,}"
    print(f"  • Missing values: {missing_val_text}")
    print(f"  • Preprocessing pipeline: Ready")
    print(f"  • Visualizations: Generated")
    
    print("\n🔬 Ready for Statistical Hypothesis Testing:")
    print("  1. Normality tests for continuous variables")
    print("  2. Independence tests for categorical variables") 
    print("  3. Correlation analysis")
    print("  4. Hypothesis testing for fraud patterns")
    print("  5. Advanced statistical modeling")
    
    print("\n📚 Available Tools:")
    print("  • FraudDataExplorer: Complete data analysis framework")
    print("  • FraudVisualizationSuite: Comprehensive visualization tools")
    print("  • Statistical libraries: scipy, statsmodels, pingouin, diptest")
    print("  • ML libraries: scikit-learn with preprocessing pipelines")
    
    print(f"\n🎯 Framework successfully initialized and ready for statistical analysis!")
    
    return df_enhanced, quality_report, pipeline


if __name__ == "__main__":
    df, report, pipeline = main()