#!/usr/bin/env python3
"""
Test script for generate_publication_figures.py
"""

import sys
import os
sys.path.append('.')

from generate_publication_figures import PublicationFigureGenerator
from statistical_analysis_integration import ComprehensiveStatisticalIntegrator
from fraud_analysis_framework import FraudDataExplorer

def test_publication_figures():
    """Test the publication figure generation with mock data"""
    print("🧪 Testing Publication Figure Generation...")
    
    try:
        # Initialize components
        explorer = FraudDataExplorer(random_seed=42)
        integrator = ComprehensiveStatisticalIntegrator(alpha=0.05, random_seed=42)
        
        # Load data
        df = explorer.load_fraud_data('transaction_fraud_data.parquet')
        df_enhanced = explorer.extract_temporal_features(df)
        
        # Run comprehensive analysis with file path
        print("📊 Running comprehensive analysis...")
        results = integrator.run_comprehensive_analysis('transaction_fraud_data.parquet')
        
        # Initialize figure generator
        fig_gen = PublicationFigureGenerator()
        
        # Test hypothesis summary figure
        print("📈 Generating hypothesis summary figure...")
        mc_results = results['multiple_comparison_results']
        fig_gen.generate_hypothesis_summary_figure(mc_results)
        
        # Test economic impact figure
        print("💰 Generating economic impact figure...")
        economic_summary = results['economic_impact_summary']
        fig_gen.generate_economic_impact_figure(economic_summary)
        
        print("✅ Publication figure generation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_publication_figures()
    sys.exit(0 if success else 1)