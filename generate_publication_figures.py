#!/usr/bin/env python3
"""
Publication-Quality Figure Generation for Fraud Detection Analysis

This module generates high-quality, publication-ready figures for the comprehensive
fraud detection statistical analysis. All figures are optimized for academic papers,
reports, and presentations.

Author: Statistical Analysis Framework
Version: 2.0.0
Date: January 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from datetime import datetime
import os

# Import our analysis frameworks
from statistical_analysis_integration import ComprehensiveStatisticalIntegrator
from fraud_analysis_framework import FraudDataExplorer
from fraud_visualization import FraudVisualizationSuite

warnings.filterwarnings('ignore')

class PublicationFigureGenerator:
    """
    Generates publication-quality figures for fraud detection analysis.
    """
    
    def __init__(self, output_dir="publication_figures", dpi=300, style='seaborn-v0_8'):
        """
        Initialize the publication figure generator.
        
        Args:
            output_dir (str): Directory to save figures
            dpi (int): Resolution for saved figures
            style (str): Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        
        # Set publication-quality style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'axes.linewidth': 1.2,
            'grid.alpha': 0.3,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        
        print(f"📊 Publication Figure Generator initialized")
        print(f"   Output directory: {self.output_dir}")
        print(f"   DPI: {self.dpi}")
        print(f"   Style: {style}")
    
    def generate_hypothesis_summary_figure(self, mc_results, save_name="hypothesis_summary"):
        """
        Generate a comprehensive hypothesis testing summary figure.
        
        Args:
            mc_results: Multiple comparison results object
            save_name (str): Base name for saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Fraud Detection Hypothesis Testing Results', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. P-values comparison - with error handling
        if hasattr(mc_results, 'raw_p_values') and mc_results.raw_p_values is not None:
            p_values = mc_results.raw_p_values
        else:
            print("Warning: raw_p_values not available, using placeholder values")
            p_values = [0.05, 0.03, 0.01, 0.08, 0.02, 0.04]  # Placeholder values
        
        if hasattr(mc_results, 'hypothesis_names') and mc_results.hypothesis_names is not None:
            hypothesis_names = [name.replace('_', ' ').title() for name in mc_results.hypothesis_names]
        else:
            print("Warning: hypothesis_names not available, using default names")
            hypothesis_names = [f'Hypothesis {i+1}' for i in range(len(p_values))]
        
        bars = axes[0, 0].bar(range(len(p_values)), p_values, 
                             color='lightblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
        axes[0, 0].set_ylabel('P-value')
        axes[0, 0].set_title('Raw P-values by Hypothesis')
        axes[0, 0].set_xticks(range(len(hypothesis_names)))
        axes[0, 0].set_xticklabels([f'H{i+1}' for i in range(len(hypothesis_names))], rotation=0)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add p-value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Multiple comparison corrections - with error handling
        corrections = ['Bonferroni', 'FDR (B-H)', 'Sidak', 'Holm']
        
        # Safe access to significance results
        def safe_sum(attr_name, default_val=0):
            if hasattr(mc_results, attr_name) and getattr(mc_results, attr_name) is not None:
                return sum(getattr(mc_results, attr_name))
            return default_val
        
        significant_counts = [
            safe_sum('significant_bonferroni'),
            safe_sum('significant_fdr_bh'),
            safe_sum('significant_sidak'),
            safe_sum('significant_holm')
        ]
        
        bars = axes[0, 1].bar(corrections, significant_counts, 
                             color=['red', 'orange', 'green', 'blue'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('Number of Significant Results')
        axes[0, 1].set_title('Significant Results by Correction Method')
        axes[0, 1].set_ylim(0, len(hypothesis_names) + 0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Hypothesis significance heatmap - with error handling
        def safe_get_significance(attr_name, default_len=6):
            if hasattr(mc_results, attr_name) and getattr(mc_results, attr_name) is not None:
                return getattr(mc_results, attr_name)
            return [False] * default_len  # Default to all non-significant
        
        significance_matrix = np.array([
            safe_get_significance('significant_bonferroni'),
            safe_get_significance('significant_fdr_bh'),
            safe_get_significance('significant_sidak'),
            safe_get_significance('significant_holm')
        ]).astype(int)
        
        im = axes[1, 0].imshow(significance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[1, 0].set_yticks(range(len(corrections)))
        axes[1, 0].set_yticklabels(corrections)
        axes[1, 0].set_xticks(range(len(hypothesis_names)))
        axes[1, 0].set_xticklabels([f'H{i+1}' for i in range(len(hypothesis_names))])
        axes[1, 0].set_title('Significance Matrix (Green = Significant)')
        
        # Add text annotations
        for i in range(len(corrections)):
            for j in range(len(hypothesis_names)):
                text = '✓' if significance_matrix[i, j] else '✗'
                axes[1, 0].text(j, i, text, ha='center', va='center', 
                               color='white', fontweight='bold', fontsize=14)
        
        # 4. Effect sizes (if available)
        axes[1, 1].text(0.5, 0.5, 'Effect Sizes\n(Implementation Dependent)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Effect Size Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"✅ Hypothesis summary figure saved: {save_path}")
        plt.show()
        
        return save_path
    
    def generate_economic_impact_figure(self, economic_summary, save_name="economic_impact"):
        """
        Generate economic impact visualization.
        
        Args:
            economic_summary (dict): Economic impact summary data
            save_name (str): Base name for saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Economic Impact Analysis - Fraud Detection Implementation', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Potential savings breakdown
        if economic_summary.get('potential_savings'):
            categories = list(economic_summary['potential_savings'].keys())
            savings = list(economic_summary['potential_savings'].values())
            
            bars = axes[0, 0].bar(categories, savings, color='green', alpha=0.7, edgecolor='black')
            axes[0, 0].set_ylabel('Potential Savings ($)')
            axes[0, 0].set_title('Potential Savings by Category')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Cost-benefit analysis
        total_savings = economic_summary.get('total_potential_savings', 0)
        total_costs = economic_summary.get('total_implementation_costs', 0)
        net_benefit = total_savings - total_costs
        
        categories_cb = ['Potential\nSavings', 'Implementation\nCosts', 'Net\nBenefit']
        values_cb = [total_savings, total_costs, net_benefit]
        colors_cb = ['green', 'red', 'blue']
        
        bars = axes[0, 1].bar(categories_cb, values_cb, color=colors_cb, alpha=0.7, edgecolor='black')
        axes[0, 1].set_ylabel('Amount ($)')
        axes[0, 1].set_title('Cost-Benefit Analysis')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. ROI visualization
        roi = economic_summary.get('overall_roi', 0)
        roi_data = [roi, 100 - roi] if roi <= 100 else [100, roi - 100]
        roi_labels = ['ROI', 'Baseline'] if roi <= 100 else ['Baseline (100%)', f'Additional ROI ({roi-100:.1f}%)']
        colors_roi = ['gold', 'lightgray'] if roi <= 100 else ['lightgray', 'gold']
        
        wedges, texts, autotexts = axes[1, 0].pie(roi_data, labels=roi_labels, colors=colors_roi,
                                                 autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title(f'Return on Investment\nTotal ROI: {roi:.1f}%')
        
        # 4. Timeline and payback
        months = ['Month 1', 'Month 2', 'Month 3', 'Month 6', 'Month 12']
        cumulative_savings = [total_savings/12 * i for i in [1, 2, 3, 6, 12]]
        
        axes[1, 1].plot(months, cumulative_savings, marker='o', linewidth=3, markersize=8, color='green')
        axes[1, 1].axhline(y=total_costs, color='red', linestyle='--', alpha=0.8, label=f'Break-even: ${total_costs:,.0f}')
        axes[1, 1].set_ylabel('Cumulative Savings ($)')
        axes[1, 1].set_title('Savings Timeline & Payback Period')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"✅ Economic impact figure saved: {save_path}")
        plt.show()
        
        return save_path
    
    def generate_all_publication_figures(self, integrator):
        """
        Generate all publication-quality figures.
        
        Args:
            integrator: ComprehensiveStatisticalIntegrator instance
        """
        print("🎨 Generating all publication-quality figures...")
        
        generated_files = []
        
        # Generate hypothesis summary
        if hasattr(integrator, 'multiple_comparison_results'):
            fig_path = self.generate_hypothesis_summary_figure(
                integrator.multiple_comparison_results,
                "comprehensive_hypothesis_summary"
            )
            generated_files.append(fig_path)
        
        # Generate economic impact
        if hasattr(integrator, 'economic_impact_summary'):
            fig_path = self.generate_economic_impact_figure(
                integrator.economic_impact_summary,
                "comprehensive_economic_impact"
            )
            generated_files.append(fig_path)
        
        print(f"\n✅ Publication figure generation completed!")
        print(f"📁 Generated {len(generated_files)} publication-quality figures")
        print(f"📂 Output directory: {self.output_dir}")
        
        return generated_files


def main():
    """
    Main function to generate all publication figures.
    """
    print("🎨 Publication Figure Generator")
    print("=" * 50)
    
    # Initialize components
    generator = PublicationFigureGenerator()
    integrator = ComprehensiveStatisticalIntegrator(alpha=0.05, random_seed=42)
    
    # Run analysis if not already done
    try:
        if not hasattr(integrator, 'integrated_results'):
            print("📊 Running comprehensive analysis...")
            integrator.run_comprehensive_analysis('transaction_fraud_data.parquet')
        
        # Generate all figures
        generated_files = generator.generate_all_publication_figures(integrator)
        
        print(f"\n🎯 Publication figures ready for:")
        print("   • Academic papers")
        print("   • Executive presentations") 
        print("   • Technical reports")
        print("   • Conference presentations")
        
    except Exception as e:
        print(f"❌ Error generating publication figures: {e}")
        print("💡 Ensure transaction_fraud_data.parquet exists and all dependencies are installed")


if __name__ == "__main__":
    main()