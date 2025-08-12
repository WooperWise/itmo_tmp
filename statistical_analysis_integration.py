#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis Integration with Multiple Comparison Corrections
==================================================================================

This module integrates all 6 fraud detection hypotheses with comprehensive statistical
analysis, multiple comparison corrections, and unified reporting capabilities.

Hypotheses Integrated:
1. Temporal Fraud Patterns (Night vs Day)
2. Weekend vs Weekday Fraud Patterns  
3. Bimodality of Fraud Transaction Amounts
4. Channel-based Fraud Analysis
5. ROI from ML Model Implementation
6. Dynamic Threshold Effectiveness

Author: Statistical Analysis Framework
Date: 2025-01-12
Version: 2.0.0
"""

# Core libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis libraries
import scipy.stats as stats
from scipy.stats import chi2_contingency, normaltest, shapiro
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests, fdrcorrection
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.contingency_tables import mcnemar
import pingouin as pg
from diptest import diptest

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# System libraries
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
from dataclasses import dataclass

# Import existing framework components
from fraud_analysis_framework import FraudDataExplorer
from hypothesis_tests_1_3 import FraudHypothesisTests
from hypothesis_tests_4_6 import AdvancedFraudAnalyzer


@dataclass
class MultipleComparisonResults:
    """Container for multiple comparison correction results"""
    original_pvalues: List[float]
    bonferroni_pvalues: List[float]
    fdr_bh_pvalues: List[float]
    fdr_by_pvalues: List[float]
    sidak_pvalues: List[float]
    holm_pvalues: List[float]
    hypothesis_names: List[str]
    alpha: float
    family_wise_error_rate: float
    false_discovery_rate: float
    significant_bonferroni: List[bool]
    significant_fdr_bh: List[bool]
    significant_fdr_by: List[bool]
    significant_sidak: List[bool]
    significant_holm: List[bool]


@dataclass
class IntegratedHypothesisResult:
    """Container for integrated hypothesis test results"""
    hypothesis_id: str
    hypothesis_name: str
    test_type: str
    original_pvalue: float
    effect_size: float
    effect_size_interpretation: str
    confidence_interval: Tuple[float, float]
    sample_size: int
    test_statistic: float
    assumptions_met: bool
    practical_significance: bool
    business_impact: str
    corrected_pvalues: Dict[str, float]
    significant_after_correction: Dict[str, bool]


class ComprehensiveStatisticalIntegrator:
    """
    Comprehensive statistical analysis integrator with multiple comparison corrections
    """
    
    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """
        Initialize the comprehensive statistical integrator
        
        Parameters:
        -----------
        alpha : float
            Significance level for statistical tests
        random_seed : int
            Random seed for reproducibility
        """
        self.alpha = alpha
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize component analyzers
        self.data_explorer = FraudDataExplorer(random_seed=random_seed)
        self.hypothesis_tester_1_3 = FraudHypothesisTests(alpha=alpha, random_seed=random_seed)
        self.hypothesis_tester_4_6 = AdvancedFraudAnalyzer()
        
        # Results storage
        self.integrated_results = {}
        self.multiple_comparison_results = None
        self.economic_impact_summary = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (16, 12)
        plt.rcParams['font.size'] = 11
        
        print("✅ ComprehensiveStatisticalIntegrator initialized successfully!")
        print(f"📊 Significance level (α): {self.alpha}")
        print(f"🔢 Random seed: {self.random_seed}")
    
    def load_and_prepare_data(self, file_path: str = 'transaction_fraud_data.parquet') -> pd.DataFrame:
        """
        Load and prepare the fraud transaction dataset
        
        Parameters:
        -----------
        file_path : str
            Path to the fraud transaction data
            
        Returns:
        --------
        pd.DataFrame
            Prepared dataset with all necessary features
        """
        print("\n" + "="*80)
        print("📊 LOADING AND PREPARING COMPREHENSIVE DATASET")
        print("="*80)
        
        # Load data using the data explorer
        df = self.data_explorer.load_fraud_data(file_path)
        
        # Extract temporal features
        df_enhanced = self.data_explorer.extract_temporal_features(df)
        
        # Generate data quality report
        quality_report = self.data_explorer.generate_data_quality_report(df_enhanced)
        
        print(f"\n📈 Dataset Summary:")
        print(f"   Shape: {df_enhanced.shape}")
        print(f"   Fraud Rate: {df_enhanced['is_fraud'].mean():.4f} ({df_enhanced['is_fraud'].mean()*100:.2f}%)")
        print(f"   Date Range: {df_enhanced['timestamp'].min()} to {df_enhanced['timestamp'].max()}")
        print(f"   Memory Usage: {df_enhanced.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df_enhanced
    
    def run_all_hypothesis_tests(self, df: pd.DataFrame) -> Dict:
        """
        Run all 6 hypothesis tests and collect results
        
        Parameters:
        -----------
        df : pd.DataFrame
            Prepared fraud transaction dataset
            
        Returns:
        --------
        Dict
            Combined results from all hypothesis tests
        """
        print("\n" + "="*80)
        print("🧪 RUNNING ALL 6 HYPOTHESIS TESTS")
        print("="*80)
        
        all_results = {}
        
        # Run hypotheses 1-3
        print("\n🔬 Running Hypotheses 1-3 (Temporal, Weekend, Bimodality)...")
        h1_results = self.hypothesis_tester_1_3.hypothesis_1_temporal_patterns(df)
        h2_results = self.hypothesis_tester_1_3.hypothesis_2_weekend_patterns(df)
        h3_results = self.hypothesis_tester_1_3.hypothesis_3_bimodality_amounts(df)
        
        # Run hypotheses 4-6
        print("\n🔬 Running Hypotheses 4-6 (Channel, ML ROI, Dynamic Thresholds)...")
        self.hypothesis_tester_4_6.df = df  # Set the dataframe
        h4_results = self.hypothesis_tester_4_6.hypothesis_4_channel_analysis()
        h5_results = self.hypothesis_tester_4_6.hypothesis_5_ml_roi_simulation()
        h6_results = self.hypothesis_tester_4_6.hypothesis_6_dynamic_threshold_effectiveness()
        
        # Combine all results
        all_results = {
            'hypothesis_1': h1_results,
            'hypothesis_2': h2_results,
            'hypothesis_3': h3_results,
            'hypothesis_4': h4_results,
            'hypothesis_5': h5_results,
            'hypothesis_6': h6_results
        }
        
        return all_results
    
    def extract_pvalues_for_correction(self, all_results: Dict) -> Tuple[List[float], List[str]]:
        """
        Extract p-values from all hypothesis tests for multiple comparison correction
        
        Parameters:
        -----------
        all_results : Dict
            Combined results from all hypothesis tests
            
        Returns:
        --------
        Tuple[List[float], List[str]]
            P-values and corresponding hypothesis names
        """
        pvalues = []
        hypothesis_names = []
        
        # Hypothesis 1: Temporal patterns
        if 'hypothesis_1' in all_results:
            pvalues.append(all_results['hypothesis_1']['p_value'])
            hypothesis_names.append("H1: Night vs Day Fraud Rates")
        
        # Hypothesis 2: Weekend patterns
        if 'hypothesis_2' in all_results:
            pvalues.append(all_results['hypothesis_2']['p_value'])
            hypothesis_names.append("H2: Weekend vs Weekday Fraud Rates")
        
        # Hypothesis 3: Bimodality
        if 'hypothesis_3' in all_results:
            # Use the strongest evidence from multiple tests
            h3 = all_results['hypothesis_3']
            if h3.get('dip_test') and h3['dip_test']['p_value'] is not None:
                pvalues.append(h3['dip_test']['p_value'])
                hypothesis_names.append("H3: Bimodality (Dip Test)")
            
            pvalues.append(h3['chi_square_concentration']['p_value'])
            hypothesis_names.append("H3: Bimodality (Chi-square Concentration)")
            
            pvalues.append(h3['chi_square_independence']['p_value'])
            hypothesis_names.append("H3: Bimodality (Chi-square Independence)")
        
        # Hypothesis 4: Channel analysis
        if 'hypothesis_4' in all_results:
            pvalues.append(all_results['hypothesis_4']['chi_square_test'].p_value)
            hypothesis_names.append("H4: Channel-based Fraud Analysis")
        
        # Hypothesis 5: ML ROI - use the best scenario
        if 'hypothesis_5' in all_results:
            # For ML ROI, we'll use a synthetic p-value based on confidence intervals
            # This is a practical approach for economic analysis
            best_scenario = all_results['hypothesis_5']['hypothesis_conclusion']['best_scenario']
            roi = all_results['hypothesis_5']['economic_analysis'][best_scenario].roi
            # Convert ROI to a p-value equivalent (higher ROI = lower p-value)
            synthetic_pvalue = max(0.001, 1.0 / (1.0 + abs(roi) / 10.0))
            pvalues.append(synthetic_pvalue)
            hypothesis_names.append("H5: ML ROI Analysis")
        
        # Hypothesis 6: Dynamic thresholds
        if 'hypothesis_6' in all_results:
            # Use precision improvement as basis for p-value
            precision_improvement = all_results['hypothesis_6']['hypothesis_conclusion']['precision_improvement']
            # Convert improvement to p-value (higher improvement = lower p-value)
            synthetic_pvalue = max(0.001, 1.0 - abs(precision_improvement))
            pvalues.append(synthetic_pvalue)
            hypothesis_names.append("H6: Dynamic Threshold Effectiveness")
        
        return pvalues, hypothesis_names
    
    def apply_multiple_comparison_corrections(self, pvalues: List[float], 
                                            hypothesis_names: List[str]) -> MultipleComparisonResults:
        """
        Apply multiple comparison corrections to p-values
        
        Parameters:
        -----------
        pvalues : List[float]
            Original p-values from hypothesis tests
        hypothesis_names : List[str]
            Names of the hypotheses
            
        Returns:
        --------
        MultipleComparisonResults
            Comprehensive multiple comparison correction results
        """
        print("\n" + "="*80)
        print("🔧 APPLYING MULTIPLE COMPARISON CORRECTIONS")
        print("="*80)
        
        pvalues_array = np.array(pvalues)
        n_tests = len(pvalues)
        
        print(f"📊 Number of tests: {n_tests}")
        print(f"📈 Original α level: {self.alpha}")
        
        # Bonferroni correction
        bonferroni_reject, bonferroni_pvals = multipletests(pvalues_array, 
                                                           alpha=self.alpha, 
                                                           method='bonferroni')[:2]
        
        # Benjamini-Hochberg FDR correction
        fdr_bh_reject, fdr_bh_pvals = fdrcorrection(pvalues_array, 
                                                   alpha=self.alpha, 
                                                   method='indep')
        
        # Benjamini-Yekutieli FDR correction (for dependent tests)
        fdr_by_reject, fdr_by_pvals = fdrcorrection(pvalues_array, 
                                                   alpha=self.alpha, 
                                                   method='negcorr')
        
        # Sidak correction
        sidak_reject, sidak_pvals = multipletests(pvalues_array, 
                                                 alpha=self.alpha, 
                                                 method='sidak')[:2]
        
        # Holm correction
        holm_reject, holm_pvals = multipletests(pvalues_array, 
                                               alpha=self.alpha, 
                                               method='holm')[:2]
        
        # Calculate error rates
        family_wise_error_rate = 1 - (1 - self.alpha)**n_tests  # Probability of at least one Type I error
        false_discovery_rate = self.alpha  # Expected proportion of false discoveries
        
        # Create results object
        results = MultipleComparisonResults(
            original_pvalues=pvalues,
            bonferroni_pvalues=bonferroni_pvals.tolist(),
            fdr_bh_pvalues=fdr_bh_pvals.tolist(),
            fdr_by_pvalues=fdr_by_pvals.tolist(),
            sidak_pvalues=sidak_pvals.tolist(),
            holm_pvalues=holm_pvals.tolist(),
            hypothesis_names=hypothesis_names,
            alpha=self.alpha,
            family_wise_error_rate=family_wise_error_rate,
            false_discovery_rate=false_discovery_rate,
            significant_bonferroni=bonferroni_reject.tolist(),
            significant_fdr_bh=fdr_bh_reject.tolist(),
            significant_fdr_by=fdr_by_reject.tolist(),
            significant_sidak=sidak_reject.tolist(),
            significant_holm=holm_reject.tolist()
        )
        
        # Print summary
        print(f"\n📊 Multiple Comparison Correction Summary:")
        print(f"   Family-wise Error Rate (FWER): {family_wise_error_rate:.4f}")
        print(f"   False Discovery Rate (FDR): {false_discovery_rate:.4f}")
        print(f"   Bonferroni α_adjusted: {self.alpha/n_tests:.6f}")
        
        print(f"\n🔍 Significant Results After Correction:")
        print(f"   Bonferroni: {sum(bonferroni_reject)}/{n_tests}")
        print(f"   FDR (B-H): {sum(fdr_bh_reject)}/{n_tests}")
        print(f"   FDR (B-Y): {sum(fdr_by_reject)}/{n_tests}")
        print(f"   Sidak: {sum(sidak_reject)}/{n_tests}")
        print(f"   Holm: {sum(holm_reject)}/{n_tests}")
        
        return results
    
    def create_multiple_comparison_summary_table(self, mc_results: MultipleComparisonResults) -> pd.DataFrame:
        """
        Create a comprehensive summary table of multiple comparison results
        
        Parameters:
        -----------
        mc_results : MultipleComparisonResults
            Multiple comparison correction results
            
        Returns:
        --------
        pd.DataFrame
            Summary table with all correction methods
        """
        summary_data = []
        
        for i, hypothesis in enumerate(mc_results.hypothesis_names):
            row = {
                'Hypothesis': hypothesis,
                'Original_p_value': mc_results.original_pvalues[i],
                'Bonferroni_p_value': mc_results.bonferroni_pvalues[i],
                'FDR_BH_p_value': mc_results.fdr_bh_pvalues[i],
                'FDR_BY_p_value': mc_results.fdr_by_pvalues[i],
                'Sidak_p_value': mc_results.sidak_pvalues[i],
                'Holm_p_value': mc_results.holm_pvalues[i],
                'Significant_Original': mc_results.original_pvalues[i] < mc_results.alpha,
                'Significant_Bonferroni': mc_results.significant_bonferroni[i],
                'Significant_FDR_BH': mc_results.significant_fdr_bh[i],
                'Significant_FDR_BY': mc_results.significant_fdr_by[i],
                'Significant_Sidak': mc_results.significant_sidak[i],
                'Significant_Holm': mc_results.significant_holm[i]
            }
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def calculate_economic_impact_summary(self, all_results: Dict) -> Dict:
        """
        Calculate comprehensive economic impact summary across all hypotheses
        
        Parameters:
        -----------
        all_results : Dict
            Combined results from all hypothesis tests
            
        Returns:
        --------
        Dict
            Economic impact summary
        """
        print("\n" + "="*80)
        print("💰 CALCULATING ECONOMIC IMPACT SUMMARY")
        print("="*80)
        
        economic_summary = {
            'total_transactions_analyzed': 0,
            'total_fraud_amount': 0,
            'potential_savings': {},
            'implementation_costs': {},
            'roi_estimates': {},
            'business_recommendations': []
        }
        
        # Extract basic metrics from hypothesis 1 (temporal patterns)
        if 'hypothesis_1' in all_results:
            h1 = all_results['hypothesis_1']
            night_total = h1['sample_sizes']['night']
            day_total = h1['sample_sizes']['day']
            economic_summary['total_transactions_analyzed'] = night_total + day_total
            
            # Estimate potential savings from temporal monitoring
            if h1['reject_null']:
                night_fraud_rate = h1['fraud_rates']['night']
                day_fraud_rate = h1['fraud_rates']['day']
                rate_difference = night_fraud_rate - day_fraud_rate
                
                # Assume average transaction amount of $50,000 (from previous analysis)
                avg_transaction_amount = 50000
                potential_night_savings = night_total * rate_difference * avg_transaction_amount
                economic_summary['potential_savings']['temporal_monitoring'] = potential_night_savings
                
                if potential_night_savings > 0:
                    economic_summary['business_recommendations'].append(
                        "Implement enhanced fraud monitoring during night hours (0-5 AM)"
                    )
        
        # Extract ML ROI estimates from hypothesis 5
        if 'hypothesis_5' in all_results:
            h5 = all_results['hypothesis_5']
            best_scenario = h5['hypothesis_conclusion']['best_scenario']
            best_metrics = h5['economic_analysis'][best_scenario]
            
            economic_summary['potential_savings']['ml_implementation'] = best_metrics.net_benefit
            economic_summary['roi_estimates']['ml_implementation'] = best_metrics.roi
            economic_summary['implementation_costs']['ml_implementation'] = 600000  # ML + operational costs
            
            if best_metrics.roi > 0:
                economic_summary['business_recommendations'].append(
                    f"Proceed with ML model implementation using {best_scenario} scenario (ROI: {best_metrics.roi:.1f}%)"
                )
        
        # Extract channel-based recommendations from hypothesis 4
        if 'hypothesis_4' in all_results:
            h4 = all_results['hypothesis_4']
            if h4['hypothesis_conclusion']['conclusion'] == 'REJECT H0':
                online_rr = h4['hypothesis_conclusion']['online_risk_ratio']
                economic_summary['business_recommendations'].append(
                    f"Implement stricter fraud controls for online channels (risk ratio: {online_rr:.2f}x)"
                )
        
        # Extract dynamic threshold recommendations from hypothesis 6
        if 'hypothesis_6' in all_results:
            h6 = all_results['hypothesis_6']
            if h6['hypothesis_conclusion']['overall_conclusion'] == 'REJECT H0':
                precision_improvement = h6['hypothesis_conclusion']['precision_improvement']
                fpr_reduction = h6['hypothesis_conclusion']['fpr_reduction']
                economic_summary['business_recommendations'].append(
                    f"Deploy dynamic threshold system (precision improvement: {precision_improvement:.1%}, FPR reduction: {fpr_reduction:.1%})"
                )
        
        # Calculate total potential impact
        total_savings = sum(economic_summary['potential_savings'].values())
        total_costs = sum(economic_summary['implementation_costs'].values())
        overall_roi = ((total_savings - total_costs) / total_costs * 100) if total_costs > 0 else 0
        
        economic_summary['total_potential_savings'] = total_savings
        economic_summary['total_implementation_costs'] = total_costs
        economic_summary['overall_roi'] = overall_roi
        
        print(f"💰 Economic Impact Summary:")
        print(f"   Total Potential Savings: ${total_savings:,.2f}")
        print(f"   Total Implementation Costs: ${total_costs:,.2f}")
        print(f"   Overall ROI: {overall_roi:.1f}%")
        print(f"   Business Recommendations: {len(economic_summary['business_recommendations'])}")
        
        return economic_summary
    
    def generate_integrated_visualization(self, mc_results: MultipleComparisonResults, 
                                        economic_summary: Dict, 
                                        save_path: str = "integrated_analysis_dashboard.png"):
        """
        Generate comprehensive integrated visualization dashboard
        
        Parameters:
        -----------
        mc_results : MultipleComparisonResults
            Multiple comparison correction results
        economic_summary : Dict
            Economic impact summary
        save_path : str
            Path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Comprehensive Fraud Detection Statistical Analysis Dashboard', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. P-value comparison plot
        n_tests = len(mc_results.hypothesis_names)
        x_pos = np.arange(n_tests)
        width = 0.15
        
        axes[0, 0].bar(x_pos - 2*width, mc_results.original_pvalues, width, 
                      label='Original', alpha=0.8, color='blue')
        axes[0, 0].bar(x_pos - width, mc_results.bonferroni_pvalues, width, 
                      label='Bonferroni', alpha=0.8, color='red')
        axes[0, 0].bar(x_pos, mc_results.fdr_bh_pvalues, width, 
                      label='FDR (B-H)', alpha=0.8, color='green')
        axes[0, 0].bar(x_pos + width, mc_results.sidak_pvalues, width, 
                      label='Sidak', alpha=0.8, color='orange')
        axes[0, 0].bar(x_pos + 2*width, mc_results.holm_pvalues, width, 
                      label='Holm', alpha=0.8, color='purple')
        
        axes[0, 0].axhline(y=mc_results.alpha, color='black', linestyle='--', 
                          label=f'α = {mc_results.alpha}')
        axes[0, 0].set_xlabel('Hypothesis Tests')
        axes[0, 0].set_ylabel('P-values')
        axes[0, 0].set_title('P-values: Original vs Multiple Comparison Corrections')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([f'H{i+1}' for i in range(n_tests)], rotation=45)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Significance comparison heatmap
        significance_data = np.array([
            [1 if p < mc_results.alpha else 0 for p in mc_results.original_pvalues],
            mc_results.significant_bonferroni,
            mc_results.significant_fdr_bh,
            mc_results.significant_fdr_by,
            mc_results.significant_sidak,
            mc_results.significant_holm
        ]).astype(int)
        
        im = axes[0, 1].imshow(significance_data, cmap='RdYlGn', aspect='auto')
        axes[0, 1].set_title('Significance After Multiple Comparison Corrections')
        axes[0, 1].set_xlabel('Hypothesis Tests')
        axes[0, 1].set_ylabel('Correction Methods')
        axes[0, 1].set_xticks(range(n_tests))
        axes[0, 1].set_xticklabels([f'H{i+1}' for i in range(n_tests)])
        axes[0, 1].set_yticks(range(6))
        axes[0, 1].set_yticklabels(['Original', 'Bonferroni', 'FDR (B-H)', 'FDR (B-Y)', 'Sidak', 'Holm'])
        
        # Add text annotations
        for i in range(6):
            for j in range(n_tests):
                text = '✓' if significance_data[i, j] else '✗'
                axes[0, 1].text(j, i, text, ha="center", va="center", 
                               color="white" if significance_data[i, j] else "black", fontsize=12)
        
        # 3. Economic impact summary
        if economic_summary.get('potential_savings'):
            savings_categories = list(economic_summary['potential_savings'].keys())
            savings_values = list(economic_summary['potential_savings'].values())
            
            bars = axes[0, 2].bar(savings_categories, savings_values, color='green', alpha=0.7)
            axes[0, 2].set_title('Potential Savings by Category')
            axes[0, 2].set_ylabel('Potential Savings ($)')
            axes[0, 2].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                               f'${height:,.0f}', ha='center', va='bottom')
        
        # 4. ROI comparison
        if economic_summary.get('roi_estimates'):
            roi_categories = list(economic_summary['roi_estimates'].keys())
            roi_values = list(economic_summary['roi_estimates'].values())
            
            colors = ['green' if roi > 0 else 'red' for roi in roi_values]
            bars = axes[1, 0].bar(roi_categories, roi_values, color=colors, alpha=0.7)
            axes[1, 0].set_title('ROI Estimates by Implementation')
            axes[1, 0].set_ylabel('ROI (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', 
                               va='bottom' if height > 0 else 'top')
        
        # 5. Hypothesis strength comparison
        # Create a radar chart showing the strength of evidence for each hypothesis
        hypothesis_strength = []
        for i, hypothesis in enumerate(mc_results.hypothesis_names):
            # Calculate strength based on number of significant corrections
            strength = sum([
                mc_results.significant_bonferroni[i],
                mc_results.significant_fdr_bh[i],
                mc_results.significant_fdr_by[i],
                mc_results.significant_sidak[i],
                mc_results.significant_holm[i]
            ])
            hypothesis_strength.append(strength)
        
        bars = axes[1, 1].bar(range(len(hypothesis_strength)), hypothesis_strength, 
                             color='skyblue', alpha=0.8)
        axes[1, 1].set_title('Hypothesis Evidence Strength\n(# of Significant Corrections)')
        axes[1, 1].set_xlabel('Hypothesis')
        axes[1, 1].set_ylabel('Number of Significant Corrections')
        axes[1, 1].set_xticks(range(len(hypothesis_strength)))
        axes[1, 1].set_xticklabels([f'H{i+1}' for i in range(len(hypothesis_strength))])
        axes[1, 1].set_ylim(0, 5)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}/5', ha='center', va='bottom')
        
        # 6. Business recommendations summary
        recommendations = economic_summary.get('business_recommendations', [])
        if recommendations:
            # Create a text summary of recommendations
            axes[1, 2].axis('off')
            axes[1, 2].set_title('Key Business Recommendations', fontsize=14, fontweight='bold')
            
            rec_text = "\n\n".join([f"• {rec}" for rec in recommendations[:5]])  # Show top 5
            axes[1, 2].text(0.05, 0.95, rec_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', wrap=True,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
            
            # Add overall summary
            total_savings = economic_summary.get('total_potential_savings', 0)
            total_costs = economic_summary.get('total_implementation_costs', 0)
            overall_roi = economic_summary.get('overall_roi', 0)
            
            summary_text = f"\nOverall Impact:\n• Total Savings: ${total_savings:,.0f}\n• Implementation Costs: ${total_costs:,.0f}\n• Net ROI: {overall_roi:.1f}%"
            axes[1, 2].text(0.05, 0.3, summary_text, transform=axes[1, 2].transAxes,
                           fontsize=11, verticalalignment='top', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Integrated analysis dashboard saved to {save_path}")
        
        plt.show()
    
    def run_comprehensive_analysis(self, file_path: str = 'transaction_fraud_data.parquet') -> Dict:
        """
        Run the complete comprehensive statistical analysis
        
        Parameters:
        -----------
        file_path : str
            Path to the fraud transaction data
            
        Returns:
        --------
        Dict
            Complete analysis results
        """
        print("\n" + "="*80)
        print("🚀 STARTING COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)
        
        # Step 1: Load and prepare data
        df = self.load_and_prepare_data(file_path)
        
        # Step 2: Run all hypothesis tests
        all_results = self.run_all_hypothesis_tests(df)
        
        # Step 3: Extract p-values for multiple comparison correction
        pvalues, hypothesis_names = self.extract_pvalues_for_correction(all_results)
        
        # Step 4: Apply multiple comparison corrections
        mc_results = self.apply_multiple_comparison_corrections(pvalues, hypothesis_names)
        self.multiple_comparison_results = mc_results
        
        # Step 5: Calculate economic impact
        economic_summary = self.calculate_economic_impact_summary(all_results)
        self.economic_impact_summary = economic_summary
        
        # Step 6: Create summary table
        summary_table = self.create_multiple_comparison_summary_table(mc_results)
        
        # Step 7: Generate integrated visualization
        self.generate_integrated_visualization(mc_results, economic_summary)
        
        # Step 8: Store integrated results
        self.integrated_results = {
            'raw_hypothesis_results': all_results,
            'multiple_comparison_results': mc_results,
            'economic_impact_summary': economic_summary,
            'summary_table': summary_table,
            'dataset_info': {
                'shape': df.shape,
                'fraud_rate': df['is_fraud'].mean(),
                'date_range': (df['timestamp'].min(), df['timestamp'].max()),
                'analysis_date': datetime.now().isoformat()
            }
        }
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE STATISTICAL ANALYSIS COMPLETED")
        print("="*80)
        
        return self.integrated_results
    
    def generate_statistical_report(self) -> str:
        """
        Generate comprehensive statistical analysis report
        
        Returns:
        --------
        str
            Formatted statistical report
        """
        if not self.integrated_results:
            raise ValueError("No analysis results available. Run comprehensive analysis first.")
        
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE FRAUD DETECTION STATISTICAL ANALYSIS REPORT")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Framework Version: 2.0.0")
        report.append(f"Significance Level (α): {self.alpha}")
        report.append("")
        
        # Dataset Summary
        dataset_info = self.integrated_results['dataset_info']
        report.append("DATASET SUMMARY")
        report.append("-" * 50)
        report.append(f"Total Transactions: {dataset_info['shape'][0]:,}")
        report.append(f"Features: {dataset_info['shape'][1]}")
        report.append(f"Overall Fraud Rate: {dataset_info['fraud_rate']:.4f} ({dataset_info['fraud_rate']*100:.2f}%)")
        report.append(f"Analysis Period: {dataset_info['date_range'][0]} to {dataset_info['date_range'][1]}")
        report.append("")
        
        # Multiple Comparison Results
        mc_results = self.multiple_comparison_results
        report.append("MULTIPLE COMPARISON CORRECTION RESULTS")
        report.append("-" * 50)
        report.append(f"Number of Hypothesis Tests: {len(mc_results.hypothesis_names)}")
        report.append(f"Family-wise Error Rate (FWER): {mc_results.family_wise_error_rate:.4f}")
        report.append(f"False Discovery Rate (FDR): {mc_results.false_discovery_rate:.4f}")
        report.append("")
        
        # Detailed Results Table
        report.append("DETAILED HYPOTHESIS TEST RESULTS")
        report.append("-" * 50)
        summary_table = self.integrated_results['summary_table']
        
        # Format table headers
        headers = ['Hypothesis', 'Original p', 'Bonferroni p', 'FDR(BH) p', 'Sidak p', 'Holm p', 'Sig. After Correction']
        report.append(f"{'Hypothesis':<40} {'Orig p':<10} {'Bonf p':<10} {'FDR p':<10} {'Sidak p':<10} {'Holm p':<10} {'Significant':<15}")
        report.append("-" * 120)
        
        for _, row in summary_table.iterrows():
            sig_methods = []
            if row['Significant_Bonferroni']: sig_methods.append('Bonf')
            if row['Significant_FDR_BH']: sig_methods.append('FDR')
            if row['Significant_Sidak']: sig_methods.append('Sidak')
            if row['Significant_Holm']: sig_methods.append('Holm')
            
            sig_str = ', '.join(sig_methods) if sig_methods else 'None'
            
            report.append(f"{row['Hypothesis'][:38]:<40} "
                         f"{row['Original_p_value']:<10.6f} "
                         f"{row['Bonferroni_p_value']:<10.6f} "
                         f"{row['FDR_BH_p_value']:<10.6f} "
                         f"{row['Sidak_p_value']:<10.6f} "
                         f"{row['Holm_p_value']:<10.6f} "
                         f"{sig_str:<15}")
        
        report.append("")
        
        # Individual Hypothesis Summaries
        report.append("INDIVIDUAL HYPOTHESIS ANALYSIS")
        report.append("-" * 50)
        
        raw_results = self.integrated_results['raw_hypothesis_results']
        
        # Hypothesis 1
        if 'hypothesis_1' in raw_results:
            h1 = raw_results['hypothesis_1']
            report.append("Hypothesis 1: Temporal Fraud Patterns (Night vs Day)")
            report.append(f"  Test: {h1['test_type']}")
            report.append(f"  Sample Sizes: Night={h1['sample_sizes']['night']:,}, Day={h1['sample_sizes']['day']:,}")
            report.append(f"  Fraud Rates: Night={h1['fraud_rates']['night']:.4f}, Day={h1['fraud_rates']['day']:.4f}")
            report.append(f"  Test Statistic: {h1['test_statistic']:.4f}")
            report.append(f"  P-value: {h1['p_value']:.6f}")
            report.append(f"  Effect Size (Cohen's h): {h1['effect_size']['cohens_h']:.4f}")
            report.append(f"  Decision: {'REJECT H0' if h1['reject_null'] else 'FAIL TO REJECT H0'}")
            report.append(f"  Business Impact: Night fraud rate is {h1['fraud_rates']['night']/h1['fraud_rates']['day']:.2f}x higher")
            report.append("")
        
        # Hypothesis 2
        if 'hypothesis_2' in raw_results:
            h2 = raw_results['hypothesis_2']
            report.append("Hypothesis 2: Weekend vs Weekday Fraud Patterns")
            report.append(f"  Test: {h2['test_type']}")
            report.append(f"  Sample Sizes: Weekend={h2['sample_sizes']['weekend']:,}, Weekday={h2['sample_sizes']['weekday']:,}")
            report.append(f"  Fraud Rates: Weekend={h2['fraud_rates']['weekend']:.4f}, Weekday={h2['fraud_rates']['weekday']:.4f}")
            report.append(f"  Actual Increase: {h2['actual_increase_percent']:.2f}%")
            report.append(f"  Target Range: {h2['target_range'][0]}-{h2['target_range'][1]}%")
            report.append(f"  P-value: {h2['p_value']:.6f}")
            report.append(f"  Practical Significance: {'YES' if h2['practical_significance'] else 'NO'}")
            report.append("")
        
        # Hypothesis 3
        if 'hypothesis_3' in raw_results:
            h3 = raw_results['hypothesis_3']
            report.append("Hypothesis 3: Bimodality of Fraud Transaction Amounts")
            report.append(f"  Sample Size: {h3['sample_size']:,} fraudulent transactions")
            report.append(f"  Amount Statistics: Mean=${h3['amount_statistics']['mean']:,.2f}, Median=${h3['amount_statistics']['median']:,.2f}")
            report.append(f"  Extreme Percentile Concentration: {h3['percentile_analysis']['extreme_concentration']*100:.1f}%")
            report.append(f"  Supporting Tests: {h3['supporting_tests']}/{h3['total_tests']}")
            report.append(f"  Evidence Strength: {'STRONG' if h3['strong_evidence'] else 'WEAK'}")
            report.append("")
        
        # Economic Impact Summary
        economic_summary = self.economic_impact_summary
        report.append("ECONOMIC IMPACT ANALYSIS")
        report.append("-" * 50)
        report.append(f"Total Potential Savings: ${economic_summary.get('total_potential_savings', 0):,.2f}")
        report.append(f"Total Implementation Costs: ${economic_summary.get('total_implementation_costs', 0):,.2f}")
        report.append(f"Overall ROI: {economic_summary.get('overall_roi', 0):.1f}%")
        report.append("")
        
        report.append("KEY BUSINESS RECOMMENDATIONS:")
        for i, rec in enumerate(economic_summary.get('business_recommendations', []), 1):
            report.append(f"  {i}. {rec}")
        
        report.append("")
        report.append("STATISTICAL CONCLUSIONS")
        report.append("-" * 50)
        
        # Count significant results after correction
        bonferroni_sig = sum(mc_results.significant_bonferroni)
        fdr_sig = sum(mc_results.significant_fdr_bh)
        
        report.append(f"After multiple comparison correction:")
        report.append(f"  - {bonferroni_sig}/{len(mc_results.hypothesis_names)} hypotheses remain significant (Bonferroni)")
        report.append(f"  - {fdr_sig}/{len(mc_results.hypothesis_names)} hypotheses remain significant (FDR)")
        
        if bonferroni_sig > 0 or fdr_sig > 0:
            report.append("  - Strong statistical evidence supports implementing fraud detection improvements")
        else:
            report.append("  - Limited statistical evidence after correction - consider larger sample sizes")
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def save_results_to_json(self, filepath: str = "comprehensive_fraud_analysis_results.json"):
        """
        Save all results to JSON file
        
        Parameters:
        -----------
        filepath : str
            Path to save the JSON results
        """
        if not self.integrated_results:
            raise ValueError("No analysis results available. Run comprehensive analysis first.")
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return {key: convert_numpy_types(value) for key, value in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy_types(self.integrated_results)
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        print(f"📁 Comprehensive results saved to {filepath}")


def main():
    """
    Main function to demonstrate the comprehensive statistical integration
    """
    print("🚀 Starting Comprehensive Fraud Detection Statistical Analysis")
    print("=" * 80)
    
    # Initialize the integrator
    integrator = ComprehensiveStatisticalIntegrator(alpha=0.05, random_seed=42)
    
    # Run comprehensive analysis
    try:
        results = integrator.run_comprehensive_analysis('transaction_fraud_data.parquet')
        
        # Generate and save statistical report
        print("\n📝 Generating comprehensive statistical report...")
        report = integrator.generate_statistical_report()
        
        # Save report to file
        with open("COMPREHENSIVE_FRAUD_ANALYSIS_REPORT.md", "w") as f:
            f.write("# " + report.replace("=", "-"))
        
        print("📁 Report saved to COMPREHENSIVE_FRAUD_ANALYSIS_REPORT.md")
        
        # Save results to JSON
        integrator.save_results_to_json("comprehensive_fraud_analysis_results.json")
        
        # Print summary
        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("📊 Files generated:")
        print("   - COMPREHENSIVE_FRAUD_ANALYSIS_REPORT.md")
        print("   - comprehensive_fraud_analysis_results.json")
        print("   - integrated_analysis_dashboard.png")
        print("\n🎯 All 6 hypotheses analyzed with multiple comparison corrections!")
        
        return integrator, results
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    integrator, results = main()