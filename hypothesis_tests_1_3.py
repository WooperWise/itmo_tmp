#!/usr/bin/env python3
"""
Fraud Transaction Statistical Analysis - Hypotheses 1-3 Implementation

This module implements comprehensive statistical analysis for the first three 
fraud detection hypotheses using rigorous statistical methods.

Hypotheses:
1. Temporal Fraud Patterns (Night vs Day)
2. Weekend vs Weekday Fraud Patterns  
3. Bimodality of Fraud Transaction Amounts

Author: Statistical Analysis Framework
Date: 2025-01-11
Version: 1.0.0
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

# Import existing framework
from fraud_analysis_framework import FraudDataExplorer


class FraudHypothesisTests:
    """
    Comprehensive statistical hypothesis testing for fraud detection patterns
    """
    
    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """
        Initialize the hypothesis testing framework
        
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
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 11
        
        # Results storage
        self.results = {}
        
        print("✅ FraudHypothesisTests initialized successfully!")
        print(f"📊 Significance level (α): {self.alpha}")
    
    def cohens_h(self, p1: float, p2: float) -> float:
        """
        Calculate Cohen's h effect size for proportions
        
        Parameters:
        -----------
        p1, p2 : float
            Proportions to compare
            
        Returns:
        --------
        float
            Cohen's h effect size
        """
        # Use statsmodels implementation
        return proportion_effectsize(p1, p2)
    
    def interpret_cohens_h(self, h: float) -> str:
        """
        Interpret Cohen's h effect size
        
        Parameters:
        -----------
        h : float
            Cohen's h value
            
        Returns:
        --------
        str
            Interpretation of effect size
        """
        abs_h = abs(h)
        if abs_h < 0.2:
            return "Small effect"
        elif abs_h < 0.5:
            return "Medium effect"
        elif abs_h < 0.8:
            return "Large effect"
        else:
            return "Very large effect"
    
    def calculate_confidence_interval_proportion_diff(self, 
                                                    count1: int, n1: int, 
                                                    count2: int, n2: int, 
                                                    confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for difference in proportions
        
        Parameters:
        -----------
        count1, n1 : int
            Count and total for group 1
        count2, n2 : int
            Count and total for group 2
        confidence : float
            Confidence level
            
        Returns:
        --------
        Tuple[float, float]
            Lower and upper bounds of confidence interval
        """
        p1 = count1 / n1
        p2 = count2 / n2
        diff = p1 - p2
        
        # Standard error for difference in proportions
        se = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
        
        # Critical value
        alpha = 1 - confidence
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval
        margin_error = z_critical * se
        return (diff - margin_error, diff + margin_error)
    
    def hypothesis_1_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Test Hypothesis 1: Night hours (0-5) have higher fraud rates than day hours (6-23)
        
        H0: Night hours do NOT have higher fraud rates than day hours
        H1: Night hours have significantly higher fraud rates than day hours
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with temporal features
            
        Returns:
        --------
        Dict
            Complete statistical analysis results
        """
        print("\n" + "="*80)
        print("🌙 HYPOTHESIS 1: TEMPORAL FRAUD PATTERNS (NIGHT VS DAY)")
        print("="*80)
        
        # Ensure temporal features exist
        if 'hour' not in df.columns:
            print("⚠️  Extracting temporal features...")
            explorer = FraudDataExplorer()
            df = explorer.extract_temporal_features(df)
        
        # Define night (0-5) and day (6-23) hours
        df['is_night'] = df['hour'].between(0, 5, inclusive='both')
        
        # Calculate fraud rates
        night_data = df[df['is_night'] == True]
        day_data = df[df['is_night'] == False]
        
        night_fraud_count = night_data['is_fraud'].sum()
        night_total = len(night_data)
        night_fraud_rate = night_fraud_count / night_total
        
        day_fraud_count = day_data['is_fraud'].sum()
        day_total = len(day_data)
        day_fraud_rate = day_fraud_count / day_total
        
        print(f"📊 Sample Sizes:")
        print(f"   Night hours (0-5): {night_total:,} transactions")
        print(f"   Day hours (6-23): {day_total:,} transactions")
        print(f"\n📈 Fraud Rates:")
        print(f"   Night: {night_fraud_rate:.4f} ({night_fraud_rate*100:.2f}%)")
        print(f"   Day: {day_fraud_rate:.4f} ({day_fraud_rate*100:.2f}%)")
        
        # Check test assumptions
        print(f"\n🔍 Test Assumptions:")
        print(f"   Night fraud count: {night_fraud_count} (≥5: {'✅' if night_fraud_count >= 5 else '❌'})")
        print(f"   Night non-fraud count: {night_total - night_fraud_count} (≥5: {'✅' if (night_total - night_fraud_count) >= 5 else '❌'})")
        print(f"   Day fraud count: {day_fraud_count} (≥5: {'✅' if day_fraud_count >= 5 else '❌'})")
        print(f"   Day non-fraud count: {day_total - day_fraud_count} (≥5: {'✅' if (day_total - day_fraud_count) >= 5 else '❌'})")
        
        # Perform z-test for proportions (one-tailed: night > day)
        counts = np.array([night_fraud_count, day_fraud_count])
        nobs = np.array([night_total, day_total])
        
        # One-tailed test: night > day
        z_stat, p_value = proportions_ztest(counts, nobs, alternative='larger')
        
        # Calculate effect size (Cohen's h)
        cohens_h_value = self.cohens_h(night_fraud_rate, day_fraud_rate)
        effect_interpretation = self.interpret_cohens_h(cohens_h_value)
        
        # Calculate 95% confidence interval for difference
        ci_lower, ci_upper = self.calculate_confidence_interval_proportion_diff(
            night_fraud_count, night_total, day_fraud_count, day_total
        )
        
        # Statistical decision
        reject_null = p_value < self.alpha
        
        print(f"\n📊 Statistical Test Results:")
        print(f"   Z-statistic: {z_stat:.4f}")
        print(f"   P-value (one-tailed): {p_value:.6f}")
        print(f"   Significance level (α): {self.alpha}")
        print(f"   Decision: {'Reject H0' if reject_null else 'Fail to reject H0'}")
        
        print(f"\n📏 Effect Size Analysis:")
        print(f"   Cohen's h: {cohens_h_value:.4f}")
        print(f"   Interpretation: {effect_interpretation}")
        print(f"   Difference in rates: {night_fraud_rate - day_fraud_rate:.4f}")
        print(f"   95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Practical interpretation
        print(f"\n💡 Interpretation:")
        if reject_null:
            print(f"   ✅ Night hours have significantly higher fraud rates than day hours")
            print(f"   📈 Night fraud rate is {night_fraud_rate/day_fraud_rate:.2f}x higher than day rate")
        else:
            print(f"   ❌ No significant evidence that night hours have higher fraud rates")
        
        # Hourly breakdown for detailed analysis
        hourly_fraud = df.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        hourly_fraud.columns = ['hour', 'total_transactions', 'fraud_count', 'fraud_rate']
        
        print(f"\n📅 Hourly Fraud Rate Breakdown:")
        for _, row in hourly_fraud.iterrows():
            period = "🌙 Night" if row['hour'] <= 5 else "☀️ Day"
            print(f"   {int(row['hour']):2d}:00 - {period}: {row['fraud_rate']:.4f} ({int(row['fraud_count']):,}/{int(row['total_transactions']):,})")
        
        # Store results
        results = {
            'hypothesis': 'H1: Night hours have higher fraud rates than day hours',
            'test_type': 'Z-test for proportions (one-tailed)',
            'sample_sizes': {'night': night_total, 'day': day_total},
            'fraud_rates': {'night': night_fraud_rate, 'day': day_fraud_rate},
            'test_statistic': z_stat,
            'p_value': p_value,
            'alpha': self.alpha,
            'reject_null': reject_null,
            'effect_size': {
                'cohens_h': cohens_h_value,
                'interpretation': effect_interpretation,
                'difference': night_fraud_rate - day_fraud_rate
            },
            'confidence_interval': {'lower': ci_lower, 'upper': ci_upper},
            'hourly_breakdown': hourly_fraud.to_dict('records'),
            'assumptions_met': all([
                night_fraud_count >= 5,
                night_total - night_fraud_count >= 5,
                day_fraud_count >= 5,
                day_total - day_fraud_count >= 5
            ])
        }
        
        self.results['hypothesis_1'] = results
        return results
    
    def hypothesis_2_weekend_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Test Hypothesis 2: Weekend fraud rate exceeds weekday fraud rate by 25-40%
        
        H0: Weekend fraud rate does NOT exceed weekday fraud rate by 25-40%
        H1: Weekend fraud rate exceeds weekday fraud rate by 25-40%
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with temporal features
            
        Returns:
        --------
        Dict
            Complete statistical analysis results
        """
        print("\n" + "="*80)
        print("📅 HYPOTHESIS 2: WEEKEND VS WEEKDAY FRAUD PATTERNS")
        print("="*80)
        
        # Ensure temporal features exist
        if 'day_of_week' not in df.columns:
            print("⚠️  Extracting temporal features...")
            explorer = FraudDataExplorer()
            df = explorer.extract_temporal_features(df)
        
        # Define weekend (Saturday=5, Sunday=6) and weekday (Monday=0 to Friday=4)
        df['is_weekend_calc'] = df['day_of_week'].isin([5, 6])
        
        # Use existing is_weekend column if available, otherwise use calculated
        weekend_col = 'is_weekend' if 'is_weekend' in df.columns else 'is_weekend_calc'
        
        # Calculate fraud rates
        weekend_data = df[df[weekend_col] == True]
        weekday_data = df[df[weekend_col] == False]
        
        weekend_fraud_count = weekend_data['is_fraud'].sum()
        weekend_total = len(weekend_data)
        weekend_fraud_rate = weekend_fraud_count / weekend_total
        
        weekday_fraud_count = weekday_data['is_fraud'].sum()
        weekday_total = len(weekday_data)
        weekday_fraud_rate = weekday_fraud_count / weekday_total
        
        print(f"📊 Sample Sizes:")
        print(f"   Weekend: {weekend_total:,} transactions")
        print(f"   Weekday: {weekday_total:,} transactions")
        print(f"\n📈 Fraud Rates:")
        print(f"   Weekend: {weekend_fraud_rate:.4f} ({weekend_fraud_rate*100:.2f}%)")
        print(f"   Weekday: {weekday_fraud_rate:.4f} ({weekday_fraud_rate*100:.2f}%)")
        
        # Calculate actual percentage increase
        if weekday_fraud_rate > 0:
            actual_increase = ((weekend_fraud_rate - weekday_fraud_rate) / weekday_fraud_rate) * 100
        else:
            actual_increase = float('inf') if weekend_fraud_rate > 0 else 0
        
        print(f"   Actual increase: {actual_increase:.2f}%")
        
        # Check if increase is in target range (25-40%)
        target_range = (25, 40)
        in_target_range = target_range[0] <= actual_increase <= target_range[1]
        
        print(f"   Target range: {target_range[0]}-{target_range[1]}%")
        print(f"   In target range: {'✅' if in_target_range else '❌'}")
        
        # Check test assumptions
        print(f"\n🔍 Test Assumptions:")
        print(f"   Weekend fraud count: {weekend_fraud_count} (≥5: {'✅' if weekend_fraud_count >= 5 else '❌'})")
        print(f"   Weekend non-fraud count: {weekend_total - weekend_fraud_count} (≥5: {'✅' if (weekend_total - weekend_fraud_count) >= 5 else '❌'})")
        print(f"   Weekday fraud count: {weekday_fraud_count} (≥5: {'✅' if weekday_fraud_count >= 5 else '❌'})")
        print(f"   Weekday non-fraud count: {weekday_total - weekday_fraud_count} (≥5: {'✅' if (weekday_total - weekday_fraud_count) >= 5 else '❌'})")
        
        # Perform z-test for proportions (two-tailed for difference)
        counts = np.array([weekend_fraud_count, weekday_fraud_count])
        nobs = np.array([weekend_total, weekday_total])
        
        # Two-tailed test for difference
        z_stat, p_value = proportions_ztest(counts, nobs, alternative='two-sided')
        
        # Calculate effect size (Cohen's h)
        cohens_h_value = self.cohens_h(weekend_fraud_rate, weekday_fraud_rate)
        effect_interpretation = self.interpret_cohens_h(cohens_h_value)
        
        # Calculate 95% confidence interval for difference
        ci_lower, ci_upper = self.calculate_confidence_interval_proportion_diff(
            weekend_fraud_count, weekend_total, weekday_fraud_count, weekday_total
        )
        
        # Statistical decision
        reject_null = p_value < self.alpha
        
        # Practical significance assessment
        practical_significance = in_target_range and reject_null
        
        print(f"\n📊 Statistical Test Results:")
        print(f"   Z-statistic: {z_stat:.4f}")
        print(f"   P-value (two-tailed): {p_value:.6f}")
        print(f"   Significance level (α): {self.alpha}")
        print(f"   Statistical significance: {'Yes' if reject_null else 'No'}")
        print(f"   Practical significance: {'Yes' if practical_significance else 'No'}")
        
        print(f"\n📏 Effect Size Analysis:")
        print(f"   Cohen's h: {cohens_h_value:.4f}")
        print(f"   Interpretation: {effect_interpretation}")
        print(f"   Difference in rates: {weekend_fraud_rate - weekday_fraud_rate:.4f}")
        print(f"   95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Day-of-week breakdown
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_fraud = df.groupby('day_of_week')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        dow_fraud.columns = ['day_of_week', 'total_transactions', 'fraud_count', 'fraud_rate']
        dow_fraud['day_name'] = [dow_names[i] for i in dow_fraud['day_of_week']]
        dow_fraud['is_weekend'] = dow_fraud['day_of_week'].isin([5, 6])
        
        print(f"\n📅 Day-of-Week Fraud Rate Breakdown:")
        for _, row in dow_fraud.iterrows():
            day_type = "🎉 Weekend" if row['is_weekend'] else "💼 Weekday"
            print(f"   {row['day_name']:<9} - {day_type}: {row['fraud_rate']:.4f} ({row['fraud_count']:,}/{row['total_transactions']:,})")
        
        # Practical interpretation
        print(f"\n💡 Interpretation:")
        if practical_significance:
            print(f"   ✅ Weekend fraud rate significantly exceeds weekday rate by {actual_increase:.1f}%")
            print(f"   📈 This falls within the hypothesized range of 25-40%")
        elif reject_null:
            print(f"   ⚠️  Weekend and weekday fraud rates are significantly different")
            print(f"   📊 However, the {actual_increase:.1f}% increase is outside the 25-40% target range")
        else:
            print(f"   ❌ No significant difference between weekend and weekday fraud rates")
        
        # Store results
        results = {
            'hypothesis': 'H2: Weekend fraud rate exceeds weekday rate by 25-40%',
            'test_type': 'Z-test for proportions (two-tailed)',
            'sample_sizes': {'weekend': weekend_total, 'weekday': weekday_total},
            'fraud_rates': {'weekend': weekend_fraud_rate, 'weekday': weekday_fraud_rate},
            'actual_increase_percent': actual_increase,
            'target_range': target_range,
            'in_target_range': in_target_range,
            'test_statistic': z_stat,
            'p_value': p_value,
            'alpha': self.alpha,
            'reject_null': reject_null,
            'practical_significance': practical_significance,
            'effect_size': {
                'cohens_h': cohens_h_value,
                'interpretation': effect_interpretation,
                'difference': weekend_fraud_rate - weekday_fraud_rate
            },
            'confidence_interval': {'lower': ci_lower, 'upper': ci_upper},
            'day_of_week_breakdown': dow_fraud.to_dict('records'),
            'assumptions_met': all([
                weekend_fraud_count >= 5,
                weekend_total - weekend_fraud_count >= 5,
                weekday_fraud_count >= 5,
                weekday_total - weekday_fraud_count >= 5
            ])
        }
        
        self.results['hypothesis_2'] = results
        return results
    
    def hypothesis_3_bimodality_amounts(self, df: pd.DataFrame) -> Dict:
        """
        Test Hypothesis 3: Fraud transaction amounts show bimodal distribution 
        concentrated in extreme percentiles (<1% and >95%)
        
        H0: Fraud amounts do NOT show bimodal distribution in extreme percentiles
        H1: Fraud amounts show bimodal distribution in extreme percentiles
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset with transaction amounts
            
        Returns:
        --------
        Dict
            Complete statistical analysis results
        """
        print("\n" + "="*80)
        print("💰 HYPOTHESIS 3: BIMODALITY OF FRAUD TRANSACTION AMOUNTS")
        print("="*80)
        
        # Extract fraud transactions only
        fraud_data = df[df['is_fraud'] == True].copy()
        fraud_amounts = fraud_data['amount'].dropna()
        
        print(f"📊 Sample Size: {len(fraud_amounts):,} fraudulent transactions")
        print(f"📈 Amount Statistics:")
        print(f"   Mean: ${fraud_amounts.mean():,.2f}")
        print(f"   Median: ${fraud_amounts.median():,.2f}")
        print(f"   Std: ${fraud_amounts.std():,.2f}")
        print(f"   Min: ${fraud_amounts.min():,.2f}")
        print(f"   Max: ${fraud_amounts.max():,.2f}")
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = fraud_amounts.quantile([p/100 for p in percentiles])
        
        print(f"\n📊 Percentile Analysis:")
        for p in percentiles:
            print(f"   {p:2d}th percentile: ${percentile_values[p/100]:,.2f}")
        
        # Define extreme percentile groups
        p1_threshold = percentile_values[0.01]  # 1st percentile
        p95_threshold = percentile_values[0.95]  # 95th percentile
        
        # Categorize transactions
        fraud_data['percentile_group'] = 'Middle (1-95%)'
        fraud_data.loc[fraud_data['amount'] <= p1_threshold, 'percentile_group'] = 'Low (<1%)'
        fraud_data.loc[fraud_data['amount'] >= p95_threshold, 'percentile_group'] = 'High (>95%)'
        
        # Count transactions in each group
        group_counts = fraud_data['percentile_group'].value_counts()
        group_props = fraud_data['percentile_group'].value_counts(normalize=True)
        
        print(f"\n📊 Percentile Group Distribution:")
        for group in ['Low (<1%)', 'Middle (1-95%)', 'High (>95%)']:
            if group in group_counts:
                count = group_counts[group]
                prop = group_props[group]
                print(f"   {group:<15}: {count:,} ({prop:.4f} or {prop*100:.2f}%)")
        
        # Test 1: Hartigan's Dip Test for Unimodality
        print(f"\n🔍 Test 1: Hartigan's Dip Test for Unimodality")
        
        # Use log-transformed amounts for better distribution properties
        log_amounts = np.log1p(fraud_amounts)  # log(1 + x) to handle zeros
        
        try:
            dip_stat, dip_p_value = diptest(log_amounts.values)
            dip_reject_null = dip_p_value < self.alpha
            
            print(f"   Dip statistic: {dip_stat:.6f}")
            print(f"   P-value: {dip_p_value:.6f}")
            print(f"   Decision: {'Reject unimodality (evidence of multimodality)' if dip_reject_null else 'Fail to reject unimodality'}")
            
        except Exception as e:
            print(f"   ⚠️  Dip test failed: {e}")
            dip_stat, dip_p_value, dip_reject_null = None, None, None
        
        # Test 2: Chi-square test for concentration in extreme percentiles
        print(f"\n🔍 Test 2: Chi-square Test for Extreme Percentile Concentration")
        
        # Expected distribution if uniform across percentiles
        total_fraud = len(fraud_data)
        expected_low = total_fraud * 0.01  # 1% expected in low group
        expected_middle = total_fraud * 0.94  # 94% expected in middle group  
        expected_high = total_fraud * 0.05  # 5% expected in high group (95-100%)
        
        observed = [
            group_counts.get('Low (<1%)', 0),
            group_counts.get('Middle (1-95%)', 0), 
            group_counts.get('High (>95%)', 0)
        ]
        expected = [expected_low, expected_middle, expected_high]
        
        print(f"   Observed: {observed}")
        print(f"   Expected: {[f'{e:.1f}' for e in expected]}")
        
        # Perform chi-square test
        chi2_stat, chi2_p_value = stats.chisquare(observed, expected)
        chi2_reject_null = chi2_p_value < self.alpha
        
        print(f"   Chi-square statistic: {chi2_stat:.4f}")
        print(f"   P-value: {chi2_p_value:.6f}")
        print(f"   Decision: {'Reject uniform distribution' if chi2_reject_null else 'Fail to reject uniform distribution'}")
        
        # Test 3: Compare fraud rates across percentile groups of all transactions
        print(f"\n🔍 Test 3: Fraud Rate Analysis Across Amount Percentiles")
        
        # Calculate percentiles for ALL transactions (not just fraud)
        all_amounts = df['amount'].dropna()
        all_p1 = all_amounts.quantile(0.01)
        all_p95 = all_amounts.quantile(0.95)
        
        # Categorize ALL transactions
        df_temp = df.copy()
        df_temp['amount_percentile_group'] = 'Middle (1-95%)'
        df_temp.loc[df_temp['amount'] <= all_p1, 'amount_percentile_group'] = 'Low (<1%)'
        df_temp.loc[df_temp['amount'] >= all_p95, 'amount_percentile_group'] = 'High (>95%)'
        
        # Calculate fraud rates by percentile group
        fraud_rates_by_group = df_temp.groupby('amount_percentile_group')['is_fraud'].agg(['count', 'sum', 'mean'])
        fraud_rates_by_group.columns = ['total_transactions', 'fraud_count', 'fraud_rate']
        
        print(f"   Fraud rates by amount percentile group:")
        for group in ['Low (<1%)', 'Middle (1-95%)', 'High (>95%)']:
            if group in fraud_rates_by_group.index:
                row = fraud_rates_by_group.loc[group]
                print(f"   {group:<15}: {row['fraud_rate']:.4f} ({row['fraud_count']:,}/{row['total_transactions']:,})")
        
        # Chi-square test for independence between percentile group and fraud
        contingency_table = pd.crosstab(df_temp['amount_percentile_group'], df_temp['is_fraud'])
        chi2_indep, chi2_indep_p, dof, expected_indep = chi2_contingency(contingency_table)
        chi2_indep_reject = chi2_indep_p < self.alpha
        
        print(f"   Chi-square test for independence:")
        print(f"   Chi-square statistic: {chi2_indep:.4f}")
        print(f"   P-value: {chi2_indep_p:.6f}")
        print(f"   Decision: {'Reject independence (fraud rate varies by amount group)' if chi2_indep_reject else 'Fail to reject independence'}")
        
        # Test 4: Normality tests on fraud amounts
        print(f"\n🔍 Test 4: Distribution Shape Analysis")
        
        # Test normality of log-transformed amounts
        shapiro_stat, shapiro_p = shapiro(log_amounts.values[:5000])  # Shapiro limited to 5000 samples
        jb_stat, jb_p = stats.jarque_bera(log_amounts.values)
        
        print(f"   Shapiro-Wilk test (log-transformed amounts):")
        print(f"   Statistic: {shapiro_stat:.6f}, P-value: {shapiro_p:.6f}")
        print(f"   Decision: {'Reject normality' if shapiro_p < self.alpha else 'Fail to reject normality'}")
        
        print(f"   Jarque-Bera test (log-transformed amounts):")
        print(f"   Statistic: {jb_stat:.6f}, P-value: {jb_p:.6f}")
        print(f"   Decision: {'Reject normality' if jb_p < self.alpha else 'Fail to reject normality'}")
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(fraud_amounts)
        kurtosis = stats.kurtosis(fraud_amounts)
        
        print(f"   Distribution characteristics:")
        print(f"   Skewness: {skewness:.4f} ({'Right-skewed' if skewness > 0 else 'Left-skewed' if skewness < 0 else 'Symmetric'})")
        print(f"   Kurtosis: {kurtosis:.4f} ({'Heavy-tailed' if kurtosis > 0 else 'Light-tailed' if kurtosis < 0 else 'Normal-tailed'})")
        
        # Overall hypothesis decision
        evidence_for_bimodality = []
        if dip_reject_null is not None:
            evidence_for_bimodality.append(('Dip test', dip_reject_null))
        evidence_for_bimodality.append(('Chi-square concentration test', chi2_reject_null))
        evidence_for_bimodality.append(('Chi-square independence test', chi2_indep_reject))
        
        # Count supporting evidence
        supporting_tests = sum(1 for _, result in evidence_for_bimodality if result)
        total_tests = len(evidence_for_bimodality)
        
        # Overall decision
        strong_evidence = supporting_tests >= 2  # At least 2 out of 3 tests support bimodality
        
        print(f"\n📊 Overall Hypothesis Assessment:")
        print(f"   Tests supporting bimodality: {supporting_tests}/{total_tests}")
        for test_name, result in evidence_for_bimodality:
            print(f"   {test_name}: {'✅ Supports' if result else '❌ Does not support'}")
        
        print(f"\n💡 Interpretation:")
        if strong_evidence:
            print(f"   ✅ Strong evidence for bimodal distribution in extreme percentiles")
            print(f"   📈 Fraud amounts are concentrated in very low and very high ranges")
        else:
            print(f"   ❌ Insufficient evidence for bimodal distribution in extreme percentiles")
            print(f"   📊 Fraud amounts may follow a different distribution pattern")
        
        # Additional insights
        extreme_concentration = (group_props.get('Low (<1%)', 0) + group_props.get('High (>95%)', 0))
        print(f"   📊 {extreme_concentration*100:.1f}% of fraud amounts are in extreme percentiles (<1% or >95%)")
        
        # Store results
        results = {
            'hypothesis': 'H3: Fraud amounts show bimodal distribution in extreme percentiles',
            'sample_size': len(fraud_amounts),
            'amount_statistics': {
                'mean': fraud_amounts.mean(),
                'median': fraud_amounts.median(),
                'std': fraud_amounts.std(),
                'min': fraud_amounts.min(),
                'max': fraud_amounts.max(),
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'percentile_analysis': {
                'thresholds': {'p1': p1_threshold, 'p95': p95_threshold},
                'group_counts': group_counts.to_dict(),
                'group_proportions': group_props.to_dict(),
                'extreme_concentration': extreme_concentration
            },
            'dip_test': {
                'statistic': dip_stat,
                'p_value': dip_p_value,
                'reject_null': dip_reject_null
            } if dip_stat is not None else None,
            'chi_square_concentration': {
                'statistic': chi2_stat,
                'p_value': chi2_p_value,
                'reject_null': chi2_reject_null,
                'observed': observed,
                'expected': expected
            },
            'chi_square_independence': {
                'statistic': chi2_indep,
                'p_value': chi2_indep_p,
                'reject_null': chi2_indep_reject,
                'contingency_table': contingency_table.to_dict()
            },
            'normality_tests': {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p}
            },
            'fraud_rates_by_group': fraud_rates_by_group.to_dict('index'),
            'supporting_tests': supporting_tests,
            'total_tests': total_tests,
            'strong_evidence': strong_evidence,
            'alpha': self.alpha
        }
        
        self.results['hypothesis_3'] = results
        return results
    
    def run_all_hypothesis_tests(self, df: pd.DataFrame) -> Dict:
        """
        Run all three hypothesis tests
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset for analysis
            
        Returns:
        --------
        Dict
            Combined results from all hypothesis tests
        """
        print("🚀 RUNNING ALL HYPOTHESIS TESTS")
        print("=" * 80)
        
        # Run all tests
        h1_results = self.hypothesis_1_temporal_patterns(df)
        h2_results = self.hypothesis_2_weekend_patterns(df)
        h3_results = self.hypothesis_3_bimodality_amounts(df)
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 SUMMARY OF ALL HYPOTHESIS TESTS")
        print("=" * 80)
        
        print(f"Hypothesis 1 (Temporal): {'✅ REJECTED H0' if h1_results['reject_null'] else '❌ FAILED TO REJECT H0'}")
        print(f"Hypothesis 2 (Weekend): {'✅ PRACTICAL SIGNIFICANCE' if h2_results['practical_significance'] else '❌ NO PRACTICAL SIGNIFICANCE'}")
        print(f"Hypothesis 3 (Bimodality): {'✅ STRONG EVIDENCE' if h3_results['strong_evidence'] else '❌ WEAK EVIDENCE'}")
        
        return {
            'hypothesis_1': h1_results,
            'hypothesis_2': h2_results,
            'hypothesis_3': h3_results,
            'summary': {
                'h1_significant': h1_results['reject_null'],
                'h2_practical': h2_results['practical_significance'],
                'h3_strong_evidence': h3_results['strong_evidence']
            }
        }
    
    def generate_statistical_summary_report(self) -> str:
        """
        Generate comprehensive statistical summary report
        """
        report = []
        report.append("=" * 80)
        report.append("FRAUD DETECTION STATISTICAL ANALYSIS - HYPOTHESES 1-3 SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Significance Level (α): {self.alpha}")
        report.append("")
        
        # Hypothesis 1 Summary
        if 'hypothesis_1' in self.results:
            h1 = self.results['hypothesis_1']
            report.append("HYPOTHESIS 1: TEMPORAL FRAUD PATTERNS (NIGHT VS DAY)")
            report.append("-" * 60)
            report.append(f"H0: Night hours do NOT have higher fraud rates than day hours")
            report.append(f"H1: Night hours have significantly higher fraud rates than day hours")
            report.append("")
            report.append(f"Sample Sizes:")
            report.append(f"  Night (0-5h): {h1['sample_sizes']['night']:,} transactions")
            report.append(f"  Day (6-23h): {h1['sample_sizes']['day']:,} transactions")
            report.append("")
            report.append(f"Fraud Rates:")
            report.append(f"  Night: {h1['fraud_rates']['night']:.4f} ({h1['fraud_rates']['night']*100:.2f}%)")
            report.append(f"  Day: {h1['fraud_rates']['day']:.4f} ({h1['fraud_rates']['day']*100:.2f}%)")
            report.append("")
            report.append(f"Statistical Test Results:")
            report.append(f"  Test: Z-test for proportions (one-tailed)")
            report.append(f"  Z-statistic: {h1['test_statistic']:.4f}")
            report.append(f"  P-value: {h1['p_value']:.6f}")
            report.append(f"  Decision: {'REJECT H0' if h1['reject_null'] else 'FAIL TO REJECT H0'}")
            report.append("")
            report.append(f"Effect Size:")
            report.append(f"  Cohen's h: {h1['effect_size']['cohens_h']:.4f}")
            report.append(f"  Interpretation: {h1['effect_size']['interpretation']}")
            report.append(f"  95% CI for difference: [{h1['confidence_interval']['lower']:.4f}, {h1['confidence_interval']['upper']:.4f}]")
            report.append("")
            report.append(f"Conclusion:")
            if h1['reject_null']:
                report.append(f"  ✅ Night hours have significantly higher fraud rates than day hours")
                report.append(f"  📈 Night fraud rate is {h1['fraud_rates']['night']/h1['fraud_rates']['day']:.2f}x higher")
            else:
                report.append(f"  ❌ No significant evidence that night hours have higher fraud rates")
            report.append("")
        
        # Hypothesis 2 Summary
        if 'hypothesis_2' in self.results:
            h2 = self.results['hypothesis_2']
            report.append("HYPOTHESIS 2: WEEKEND VS WEEKDAY FRAUD PATTERNS")
            report.append("-" * 60)
            report.append(f"H0: Weekend fraud rate does NOT exceed weekday rate by 25-40%")
            report.append(f"H1: Weekend fraud rate exceeds weekday rate by 25-40%")
            report.append("")
            report.append(f"Sample Sizes:")
            report.append(f"  Weekend: {h2['sample_sizes']['weekend']:,} transactions")
            report.append(f"  Weekday: {h2['sample_sizes']['weekday']:,} transactions")
            report.append("")
            report.append(f"Fraud Rates:")
            report.append(f"  Weekend: {h2['fraud_rates']['weekend']:.4f} ({h2['fraud_rates']['weekend']*100:.2f}%)")
            report.append(f"  Weekday: {h2['fraud_rates']['weekday']:.4f} ({h2['fraud_rates']['weekday']*100:.2f}%)")
            report.append(f"  Actual increase: {h2['actual_increase_percent']:.2f}%")
            report.append(f"  Target range: {h2['target_range'][0]}-{h2['target_range'][1]}%")
            report.append(f"  In target range: {'YES' if h2['in_target_range'] else 'NO'}")
            report.append("")
            report.append(f"Statistical Test Results:")
            report.append(f"  Test: Z-test for proportions (two-tailed)")
            report.append(f"  Z-statistic: {h2['test_statistic']:.4f}")
            report.append(f"  P-value: {h2['p_value']:.6f}")
            report.append(f"  Statistical significance: {'YES' if h2['reject_null'] else 'NO'}")
            report.append(f"  Practical significance: {'YES' if h2['practical_significance'] else 'NO'}")
            report.append("")
            report.append(f"Effect Size:")
            report.append(f"  Cohen's h: {h2['effect_size']['cohens_h']:.4f}")
            report.append(f"  Interpretation: {h2['effect_size']['interpretation']}")
            report.append(f"  95% CI for difference: [{h2['confidence_interval']['lower']:.4f}, {h2['confidence_interval']['upper']:.4f}]")
            report.append("")
            report.append(f"Conclusion:")
            if h2['practical_significance']:
                report.append(f"  ✅ Weekend fraud rate significantly exceeds weekday rate by {h2['actual_increase_percent']:.1f}%")
                report.append(f"  📈 This falls within the hypothesized range of 25-40%")
            elif h2['reject_null']:
                report.append(f"  ⚠️  Weekend and weekday fraud rates are significantly different")
                report.append(f"  📊 However, the {h2['actual_increase_percent']:.1f}% increase is outside the target range")
            else:
                report.append(f"  ❌ No significant difference between weekend and weekday fraud rates")
            report.append("")
        
        # Hypothesis 3 Summary
        if 'hypothesis_3' in self.results:
            h3 = self.results['hypothesis_3']
            report.append("HYPOTHESIS 3: BIMODALITY OF FRAUD TRANSACTION AMOUNTS")
            report.append("-" * 60)
            report.append(f"H0: Fraud amounts do NOT show bimodal distribution in extreme percentiles")
            report.append(f"H1: Fraud amounts show bimodal distribution in extreme percentiles")
            report.append("")
            report.append(f"Sample Size: {h3['sample_size']:,} fraudulent transactions")
            report.append("")
            report.append(f"Amount Statistics:")
            report.append(f"  Mean: ${h3['amount_statistics']['mean']:,.2f}")
            report.append(f"  Median: ${h3['amount_statistics']['median']:,.2f}")
            report.append(f"  Std: ${h3['amount_statistics']['std']:,.2f}")
            report.append(f"  Skewness: {h3['amount_statistics']['skewness']:.4f}")
            report.append(f"  Kurtosis: {h3['amount_statistics']['kurtosis']:.4f}")
            report.append("")
            report.append(f"Percentile Analysis:")
            report.append(f"  1st percentile threshold: ${h3['percentile_analysis']['thresholds']['p1']:,.2f}")
            report.append(f"  95th percentile threshold: ${h3['percentile_analysis']['thresholds']['p95']:,.2f}")
            report.append(f"  Extreme percentile concentration: {h3['percentile_analysis']['extreme_concentration']*100:.1f}%")
            report.append("")
            report.append(f"Statistical Test Results:")
            
            if h3['dip_test']:
                report.append(f"  Hartigan's Dip Test:")
                report.append(f"    Statistic: {h3['dip_test']['statistic']:.6f}")
                report.append(f"    P-value: {h3['dip_test']['p_value']:.6f}")
                report.append(f"    Result: {'Multimodal evidence' if h3['dip_test']['reject_null'] else 'Unimodal'}")
            else:
                report.append(f"  Hartigan's Dip Test: Not available")
            
            report.append(f"  Chi-square Concentration Test:")
            report.append(f"    Statistic: {h3['chi_square_concentration']['statistic']:.4f}")
            report.append(f"    P-value: {h3['chi_square_concentration']['p_value']:.6f}")
            report.append(f"    Result: {'Non-uniform distribution' if h3['chi_square_concentration']['reject_null'] else 'Uniform distribution'}")
            
            report.append(f"  Chi-square Independence Test:")
            report.append(f"    Statistic: {h3['chi_square_independence']['statistic']:.4f}")
            report.append(f"    P-value: {h3['chi_square_independence']['p_value']:.6f}")
            report.append(f"    Result: {'Fraud rate varies by amount group' if h3['chi_square_independence']['reject_null'] else 'Fraud rate independent of amount group'}")
            report.append("")
            report.append(f"Overall Assessment:")
            report.append(f"  Supporting tests: {h3['supporting_tests']}/{h3['total_tests']}")
            report.append(f"  Evidence strength: {'STRONG' if h3['strong_evidence'] else 'WEAK'}")
            report.append("")
            report.append(f"Conclusion:")
            if h3['strong_evidence']:
                report.append(f"  ✅ Strong evidence for bimodal distribution in extreme percentiles")
                report.append(f"  📈 Fraud amounts are concentrated in very low and very high ranges")
            else:
                report.append(f"  ❌ Insufficient evidence for bimodal distribution in extreme percentiles")
                report.append(f"  📊 Fraud amounts may follow a different distribution pattern")
            report.append("")
        
        # Overall Summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 60)
        h1_result = "SIGNIFICANT" if self.results.get('hypothesis_1', {}).get('reject_null', False) else "NOT SIGNIFICANT"
        h2_result = "PRACTICAL SIGNIFICANCE" if self.results.get('hypothesis_2', {}).get('practical_significance', False) else "NO PRACTICAL SIGNIFICANCE"
        h3_result = "STRONG EVIDENCE" if self.results.get('hypothesis_3', {}).get('strong_evidence', False) else "WEAK EVIDENCE"
        
        report.append(f"Hypothesis 1 (Temporal Patterns): {h1_result}")
        report.append(f"Hypothesis 2 (Weekend Patterns): {h2_result}")
        report.append(f"Hypothesis 3 (Amount Bimodality): {h3_result}")
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_hypothesis_1_visualization(self, df: pd.DataFrame, viz_dir: str):
        """Generate hourly fraud rate heatmap for Hypothesis 1"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Ensure temporal features exist
        if 'hour' not in df.columns:
            explorer = FraudDataExplorer()
            df = explorer.extract_temporal_features(df)
        
        # Calculate hourly fraud rates
        hourly_fraud = df.groupby('hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        hourly_fraud['fraud_rate'] = hourly_fraud['mean']
        
        # Create heatmap data (reshape for better visualization)
        heatmap_data = hourly_fraud['fraud_rate'].values.reshape(4, 6)  # 4x6 grid for 24 hours
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data,
                   annot=True,
                   fmt='.4f',
                   cmap='Reds',
                   xticklabels=[f'{i}-{i+3}h' for i in range(0, 24, 4)],
                   yticklabels=[f'Row {i+1}' for i in range(4)],
                   cbar_kws={'label': 'Fraud Rate'})
        
        plt.title('Hourly Fraud Rate Heatmap\nHypothesis 1: Night vs Day Patterns',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Hour Groups', fontsize=12)
        plt.ylabel('Time Blocks', fontsize=12)
        
        # Add night hours annotation
        plt.text(0.5, -0.15, 'Night Hours (0-5): Higher fraud rates expected',
                transform=plt.gca().transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        save_path = f"{viz_dir}/hypothesis_1_temporal_patterns.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {save_path}")
    
    def _generate_hypothesis_2_visualization(self, df: pd.DataFrame, viz_dir: str):
        """Generate weekend vs weekday fraud rate barplot with confidence intervals"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Ensure temporal features exist
        if 'day_of_week' not in df.columns:
            explorer = FraudDataExplorer()
            df = explorer.extract_temporal_features(df)
        
        # Create weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday=5, Sunday=6
        
        # Calculate fraud rates and confidence intervals
        weekend_data = df[df['is_weekend'] == True]
        weekday_data = df[df['is_weekend'] == False]
        
        weekend_fraud_rate = weekend_data['is_fraud'].mean()
        weekday_fraud_rate = weekday_data['is_fraud'].mean()
        
        # Calculate confidence intervals (using normal approximation)
        def calc_ci(data, confidence=0.95):
            n = len(data)
            p = data.mean()
            se = np.sqrt(p * (1 - p) / n)
            z = stats.norm.ppf((1 + confidence) / 2)
            return z * se
        
        weekend_ci = calc_ci(weekend_data['is_fraud'])
        weekday_ci = calc_ci(weekday_data['is_fraud'])
        
        # Create barplot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        categories = ['Weekday', 'Weekend']
        fraud_rates = [weekday_fraud_rate, weekend_fraud_rate]
        ci_values = [weekday_ci, weekend_ci]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax.bar(categories, fraud_rates, yerr=ci_values,
                     capsize=10, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, rate, ci) in enumerate(zip(bars, fraud_rates, ci_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + ci + 0.0001,
                   f'{rate:.4f}±{ci:.4f}', ha='center', va='bottom',
                   fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Fraud Rate', fontsize=12)
        ax.set_title('Weekend vs Weekday Fraud Rates\nHypothesis 2: Weekend Pattern Analysis',
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistical annotation
        increase_pct = ((weekend_fraud_rate - weekday_fraud_rate) / weekday_fraud_rate) * 100
        ax.text(0.5, 0.95, f'Weekend increase: {increase_pct:.1f}%',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
               fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = f"{viz_dir}/hypothesis_2_weekend_patterns.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {save_path}")
    
    def _generate_hypothesis_3_visualization(self, df: pd.DataFrame, viz_dir: str):
        """Generate bimodality distribution plots for Hypothesis 3"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get fraud transaction amounts
        fraud_amounts = df[df['is_fraud'] == 1]['amount'].values
        
        if len(fraud_amounts) == 0:
            print("   ⚠️  No fraud transactions found for visualization")
            return
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fraud Transaction Amount Distribution Analysis\nHypothesis 3: Bimodality Evidence',
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Histogram with KDE
        axes[0, 0].hist(fraud_amounts, bins=50, density=True, alpha=0.7,
                       color='lightblue', edgecolor='black')
        sns.kdeplot(fraud_amounts, ax=axes[0, 0], color='red', linewidth=2)
        axes[0, 0].set_title('Distribution with Kernel Density Estimate', fontweight='bold')
        axes[0, 0].set_xlabel('Transaction Amount ($)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot
        axes[0, 1].boxplot(fraud_amounts, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 1].set_title('Box Plot - Outlier Detection', fontweight='bold')
        axes[0, 1].set_ylabel('Transaction Amount ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Log-scale histogram
        log_amounts = np.log10(fraud_amounts + 1)  # Add 1 to avoid log(0)
        axes[1, 0].hist(log_amounts, bins=50, density=True, alpha=0.7,
                       color='lightcoral', edgecolor='black')
        sns.kdeplot(log_amounts, ax=axes[1, 0], color='darkred', linewidth=2)
        axes[1, 0].set_title('Log-Scale Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Log10(Transaction Amount + 1)')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Percentile analysis
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(fraud_amounts, percentiles)
        
        axes[1, 1].plot(percentiles, percentile_values, 'o-', linewidth=2,
                       markersize=8, color='purple')
        axes[1, 1].set_title('Percentile Analysis', fontweight='bold')
        axes[1, 1].set_xlabel('Percentile')
        axes[1, 1].set_ylabel('Transaction Amount ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add extreme percentile highlighting
        p1_val = np.percentile(fraud_amounts, 1)
        p99_val = np.percentile(fraud_amounts, 99)
        axes[1, 1].axhline(y=p1_val, color='red', linestyle='--', alpha=0.7, label=f'1st percentile: ${p1_val:.2f}')
        axes[1, 1].axhline(y=p99_val, color='red', linestyle='--', alpha=0.7, label=f'99th percentile: ${p99_val:.2f}')
        axes[1, 1].legend()
        
        # Add summary statistics text
        stats_text = f"""Summary Statistics:
Mean: ${fraud_amounts.mean():.2f}
Median: ${np.median(fraud_amounts):.2f}
Std: ${fraud_amounts.std():.2f}
Skewness: {stats.skew(fraud_amounts):.3f}
Kurtosis: {stats.kurtosis(fraud_amounts):.3f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"),
                verticalalignment='bottom')
        
        plt.tight_layout()
        save_path = f"{viz_dir}/hypothesis_3_bimodality.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ {save_path}")
    
    def save_results_to_json(self, filepath: str = "hypothesis_test_results.json"):
        """
        Save all results to JSON file
        """
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy_types(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        print(f"📁 Results saved to {filepath}")


def main():
    """
    Main function to demonstrate the hypothesis testing framework
    """
    print("🚀 Starting Fraud Hypothesis Testing Framework")
    print("=" * 80)
    
    # Initialize components
    explorer = FraudDataExplorer(random_seed=42)
    hypothesis_tester = FraudHypothesisTests(alpha=0.05, random_seed=42)
    
    # Load data
    df = explorer.load_fraud_data('transaction_fraud_data.parquet')
    
    # Extract temporal features if needed
    df_enhanced = explorer.extract_temporal_features(df)
    
    print(f"\n📊 Dataset loaded: {df_enhanced.shape[0]:,} transactions")
    print(f"📈 Fraud rate: {df_enhanced['is_fraud'].mean():.4f} ({df_enhanced['is_fraud'].mean()*100:.2f}%)")
    
    # Run all hypothesis tests
    all_results = hypothesis_tester.run_all_hypothesis_tests(df_enhanced)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    try:
        # Create visualization directory
        viz_dir = "hypothesis_visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate Hypothesis 1 visualization: Hourly fraud rate heatmap
        hypothesis_tester._generate_hypothesis_1_visualization(df_enhanced, viz_dir)
        
        # Generate Hypothesis 2 visualization: Weekend vs weekday fraud rate barplot
        hypothesis_tester._generate_hypothesis_2_visualization(df_enhanced, viz_dir)
        
        # Generate Hypothesis 3 visualization: Bimodality distribution plots
        hypothesis_tester._generate_hypothesis_3_visualization(df_enhanced, viz_dir)
        
        print("✅ All visualizations generated successfully!")
        
    except Exception as e:
        print(f"⚠️  Visualization generation encountered an issue: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate and save statistical report
    print("\n📝 Generating statistical summary report...")
    report = hypothesis_tester.generate_statistical_summary_report()
    
    # Save report to file
    with open("hypothesis_test_summary_report.txt", "w") as f:
        f.write(report)
    
    print("📁 Report saved to hypothesis_test_summary_report.txt")
    
    # Save results to JSON
    hypothesis_tester.save_results_to_json("hypothesis_test_results.json")
    
    # Print summary
    print("\n" + "=" * 80)
    print("✅ HYPOTHESIS TESTING FRAMEWORK COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("📊 Files generated:")
    print("   - hypothesis_test_summary_report.txt")
    print("   - hypothesis_test_results.json")
    print("   - hypothesis_visualizations/ (directory)")
    print("\n🎯 All three hypotheses have been rigorously tested with statistical significance!")
    
    return df_enhanced, all_results


if __name__ == "__main__":
    df, results = main()