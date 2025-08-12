#!/usr/bin/env python3
"""
Fraud Transaction Data Visualization Module

This module provides comprehensive visualization functions for fraud transaction analysis.

Author: Statistical Analysis Framework
Date: 2025-01-11
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FraudVisualizationSuite:
    """
    Comprehensive visualization suite for fraud transaction analysis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the visualization suite
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size for plots
        """
        self.figsize = figsize
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['font.size'] = 10
        
        print("✅ FraudVisualizationSuite initialized successfully!")
    
    def plot_fraud_overview_dashboard(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create comprehensive fraud overview dashboard
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset with fraud data
        save_path : Optional[str]
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Fraud Transaction Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Overall fraud distribution
        fraud_counts = df['is_fraud'].value_counts()
        colors = ['lightblue', 'salmon']
        wedges, texts, autotexts = axes[0, 0].pie(fraud_counts.values, 
                                                  labels=['Legitimate', 'Fraudulent'], 
                                                  autopct='%1.2f%%', 
                                                  startangle=90, 
                                                  colors=colors,
                                                  explode=(0, 0.1))
        axes[0, 0].set_title('Overall Fraud Distribution', fontsize=14, fontweight='bold')
        
        # 2. Amount distribution by fraud status
        legitimate_amounts = df[df['is_fraud'] == False]['amount']
        fraudulent_amounts = df[df['is_fraud'] == True]['amount']
        
        # Use log scale for better visualization
        axes[0, 1].hist(np.log1p(legitimate_amounts), bins=50, alpha=0.7, 
                        label='Legitimate', color='lightblue', density=True)
        axes[0, 1].hist(np.log1p(fraudulent_amounts), bins=50, alpha=0.7, 
                        label='Fraudulent', color='salmon', density=True)
        axes[0, 1].set_xlabel('Log(Amount + 1)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Amount Distribution by Fraud Status (Log Scale)', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        
        # 3. Fraud rate by hour of day
        if 'hour' in df.columns:
            hourly_fraud = df.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
            bars = axes[0, 2].bar(hourly_fraud['hour'], hourly_fraud['mean'], 
                                  color='coral', alpha=0.8, edgecolor='darkred')
            axes[0, 2].set_xlabel('Hour of Day')
            axes[0, 2].set_ylabel('Fraud Rate')
            axes[0, 2].set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
            axes[0, 2].set_xticks(range(0, 24, 4))
            axes[0, 2].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Fraud rate by vendor category
        vendor_fraud = df.groupby('vendor_category')['is_fraud'].mean().sort_values(ascending=False)
        bars = axes[1, 0].bar(range(len(vendor_fraud)), vendor_fraud.values, 
                              color='lightcoral', edgecolor='darkred')
        axes[1, 0].set_xlabel('Vendor Category')
        axes[1, 0].set_ylabel('Fraud Rate')
        axes[1, 0].set_title('Fraud Rate by Vendor Category', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(len(vendor_fraud)))
        axes[1, 0].set_xticklabels(vendor_fraud.index, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 5. Fraud rate by country
        country_fraud = df.groupby('country')['is_fraud'].mean().sort_values(ascending=False)
        bars = axes[1, 1].bar(range(len(country_fraud)), country_fraud.values, 
                              color='lightgreen', edgecolor='darkgreen')
        axes[1, 1].set_xlabel('Country')
        axes[1, 1].set_ylabel('Fraud Rate')
        axes[1, 1].set_title('Fraud Rate by Country', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(range(len(country_fraud)))
        axes[1, 1].set_xticklabels(country_fraud.index, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Correlation heatmap for key features
        numerical_cols = ['amount', 'is_fraud']
        if 'hour' in df.columns:
            numerical_cols.extend(['hour', 'day_of_week', 'month'])
        
        # Add boolean columns as numerical
        bool_cols = ['is_card_present', 'is_outside_home_country', 'is_high_risk_vendor', 'is_weekend']
        for col in bool_cols:
            if col in df.columns:
                numerical_cols.append(col)
        
        corr_data = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[1, 2])
        axes[1, 2].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Dashboard saved to {save_path}")
        
        plt.show()
    
    def plot_temporal_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create temporal analysis visualizations
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset with temporal features
        save_path : Optional[str]
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Temporal Analysis of Fraud Transactions', fontsize=18, fontweight='bold')
        
        # 1. Daily fraud pattern
        if 'timestamp' in df.columns:
            df['date'] = df['timestamp'].dt.date
            daily_fraud = df.groupby('date')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
            
            axes[0, 0].plot(daily_fraud['date'], daily_fraud['mean'], 
                           color='red', linewidth=2, marker='o', markersize=3)
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Fraud Rate')
            axes[0, 0].set_title('Daily Fraud Rate Trend', fontsize=14, fontweight='bold')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Day of week analysis
        if 'day_of_week' in df.columns:
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            dow_fraud = df.groupby('day_of_week')['is_fraud'].mean()
            
            bars = axes[0, 1].bar(range(7), dow_fraud.values, 
                                  color=['lightblue' if i < 5 else 'orange' for i in range(7)],
                                  edgecolor='navy')
            axes[0, 1].set_xlabel('Day of Week')
            axes[0, 1].set_ylabel('Fraud Rate')
            axes[0, 1].set_title('Fraud Rate by Day of Week', fontsize=14, fontweight='bold')
            axes[0, 1].set_xticks(range(7))
            axes[0, 1].set_xticklabels(dow_names)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Hour bin analysis
        if 'hour_bin' in df.columns:
            hourbin_fraud = df.groupby('hour_bin')['is_fraud'].mean()
            
            bars = axes[1, 0].bar(range(len(hourbin_fraud)), hourbin_fraud.values,
                                  color=['darkblue', 'gold', 'orange', 'purple'],
                                  edgecolor='black')
            axes[1, 0].set_xlabel('Time Period')
            axes[1, 0].set_ylabel('Fraud Rate')
            axes[1, 0].set_title('Fraud Rate by Time Period', fontsize=14, fontweight='bold')
            axes[1, 0].set_xticks(range(len(hourbin_fraud)))
            axes[1, 0].set_xticklabels(hourbin_fraud.index)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Business hours vs non-business hours
        if 'is_business_hours' in df.columns:
            bh_fraud = df.groupby('is_business_hours')['is_fraud'].mean()
            
            bars = axes[1, 1].bar(['Non-Business Hours', 'Business Hours'], bh_fraud.values,
                                  color=['red', 'green'], alpha=0.7, edgecolor='black')
            axes[1, 1].set_ylabel('Fraud Rate')
            axes[1, 1].set_title('Fraud Rate: Business vs Non-Business Hours', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Temporal analysis saved to {save_path}")
        
        plt.show()
    
    def plot_amount_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create detailed amount analysis visualizations
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        save_path : Optional[str]
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Transaction Amount Analysis', fontsize=18, fontweight='bold')
        
        # 1. Box plot of amounts by fraud status
        fraud_labels = ['Legitimate', 'Fraudulent']
        amount_data = [df[df['is_fraud'] == False]['amount'], df[df['is_fraud'] == True]['amount']]
        
        bp = axes[0, 0].boxplot(amount_data, labels=fraud_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('salmon')
        axes[0, 0].set_ylabel('Transaction Amount')
        axes[0, 0].set_title('Amount Distribution by Fraud Status', fontsize=14, fontweight='bold')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Amount percentiles comparison
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        legit_percentiles = df[df['is_fraud'] == False]['amount'].quantile([p/100 for p in percentiles])
        fraud_percentiles = df[df['is_fraud'] == True]['amount'].quantile([p/100 for p in percentiles])
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        bars1 = axes[0, 1].bar(x - width/2, legit_percentiles.values, width, 
                               label='Legitimate', color='lightblue', edgecolor='navy')
        bars2 = axes[0, 1].bar(x + width/2, fraud_percentiles.values, width,
                               label='Fraudulent', color='salmon', edgecolor='darkred')
        
        axes[0, 1].set_xlabel('Percentiles')
        axes[0, 1].set_ylabel('Amount')
        axes[0, 1].set_title('Amount Percentiles Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'{p}th' for p in percentiles])
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Amount bins analysis
        amount_bins = pd.cut(df['amount'], bins=10, labels=[f'Bin_{i+1}' for i in range(10)])
        bin_fraud = df.groupby(amount_bins)['is_fraud'].mean()
        
        bars = axes[1, 0].bar(range(len(bin_fraud)), bin_fraud.values,
                              color='purple', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Amount Bins (Deciles)')
        axes[1, 0].set_ylabel('Fraud Rate')
        axes[1, 0].set_title('Fraud Rate by Amount Deciles', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(len(bin_fraud)))
        axes[1, 0].set_xticklabels(bin_fraud.index, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Currency analysis
        if 'currency' in df.columns:
            currency_fraud = df.groupby('currency')['is_fraud'].mean().sort_values(ascending=False)
            
            bars = axes[1, 1].bar(range(len(currency_fraud)), currency_fraud.values,
                                  color='gold', edgecolor='orange')
            axes[1, 1].set_xlabel('Currency')
            axes[1, 1].set_ylabel('Fraud Rate')
            axes[1, 1].set_title('Fraud Rate by Currency', fontsize=14, fontweight='bold')
            axes[1, 1].set_xticks(range(len(currency_fraud)))
            axes[1, 1].set_xticklabels(currency_fraud.index)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Amount analysis saved to {save_path}")
        
        plt.show()
    
    def plot_categorical_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create categorical variables analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        save_path : Optional[str]
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Categorical Variables Analysis', fontsize=18, fontweight='bold')
        
        # 1. Card type analysis
        if 'card_type' in df.columns:
            card_fraud = df.groupby('card_type')['is_fraud'].mean().sort_values(ascending=False)
            
            bars = axes[0, 0].bar(range(len(card_fraud)), card_fraud.values,
                                  color='lightcoral', edgecolor='darkred')
            axes[0, 0].set_xlabel('Card Type')
            axes[0, 0].set_ylabel('Fraud Rate')
            axes[0, 0].set_title('Fraud Rate by Card Type', fontsize=14, fontweight='bold')
            axes[0, 0].set_xticks(range(len(card_fraud)))
            axes[0, 0].set_xticklabels(card_fraud.index, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Channel analysis
        if 'channel' in df.columns:
            channel_fraud = df.groupby('channel')['is_fraud'].mean().sort_values(ascending=False)
            
            bars = axes[0, 1].bar(range(len(channel_fraud)), channel_fraud.values,
                                  color='lightgreen', edgecolor='darkgreen')
            axes[0, 1].set_xlabel('Channel')
            axes[0, 1].set_ylabel('Fraud Rate')
            axes[0, 1].set_title('Fraud Rate by Channel', fontsize=14, fontweight='bold')
            axes[0, 1].set_xticks(range(len(channel_fraud)))
            axes[0, 1].set_xticklabels(channel_fraud.index)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Device analysis
        if 'device' in df.columns:
            device_fraud = df.groupby('device')['is_fraud'].mean().sort_values(ascending=False)
            
            bars = axes[1, 0].bar(range(len(device_fraud)), device_fraud.values,
                                  color='gold', edgecolor='orange')
            axes[1, 0].set_xlabel('Device')
            axes[1, 0].set_ylabel('Fraud Rate')
            axes[1, 0].set_title('Fraud Rate by Device', fontsize=14, fontweight='bold')
            axes[1, 0].set_xticks(range(len(device_fraud)))
            axes[1, 0].set_xticklabels(device_fraud.index, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Risk factors comparison
        risk_factors = ['is_card_present', 'is_outside_home_country', 'is_high_risk_vendor', 'is_weekend']
        risk_data = []
        risk_labels = []
        
        for factor in risk_factors:
            if factor in df.columns:
                fraud_rate_true = df[df[factor] == True]['is_fraud'].mean()
                fraud_rate_false = df[df[factor] == False]['is_fraud'].mean()
                risk_data.extend([fraud_rate_false, fraud_rate_true])
                risk_labels.extend([f'{factor}\n(False)', f'{factor}\n(True)'])
        
        if risk_data:
            colors = ['lightblue', 'salmon'] * (len(risk_data) // 2)
            bars = axes[1, 1].bar(range(len(risk_data)), risk_data, color=colors, edgecolor='black')
            axes[1, 1].set_xlabel('Risk Factors')
            axes[1, 1].set_ylabel('Fraud Rate')
            axes[1, 1].set_title('Fraud Rate by Risk Factors', fontsize=14, fontweight='bold')
            axes[1, 1].set_xticks(range(len(risk_data)))
            axes[1, 1].set_xticklabels(risk_labels, rotation=45, ha='right', fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Categorical analysis saved to {save_path}")
        
        plt.show()
    
    def generate_all_visualizations(self, df: pd.DataFrame, output_dir: str = "visualizations"):
        """
        Generate all visualization plots and save them
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset
        output_dir : str
            Directory to save all plots
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🎨 Generating all visualizations in {output_dir}/...")
        
        # Generate all plots
        self.plot_fraud_overview_dashboard(df, f"{output_dir}/fraud_overview_dashboard.png")
        self.plot_temporal_analysis(df, f"{output_dir}/temporal_analysis.png")
        self.plot_amount_analysis(df, f"{output_dir}/amount_analysis.png")
        self.plot_categorical_analysis(df, f"{output_dir}/categorical_analysis.png")
        
        print(f"✅ All visualizations generated and saved in {output_dir}/")


def main():
    """
    Main function to demonstrate the visualization suite
    """
    print("🎨 Starting Fraud Visualization Suite Demo")
    print("=" * 50)
    
    # This would typically be called with actual data
    # For demo purposes, we'll show how to use it
    print("📊 Visualization suite ready for use!")
    print("Usage example:")
    print("  from fraud_visualization import FraudVisualizationSuite")
    print("  viz = FraudVisualizationSuite()")
    print("  viz.generate_all_visualizations(df)")


if __name__ == "__main__":
    main()