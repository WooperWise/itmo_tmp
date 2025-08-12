# ----------------------------------------------------------------------------------------------------
COMPREHENSIVE FRAUD DETECTION STATISTICAL ANALYSIS REPORT
----------------------------------------------------------------------------------------------------
Generated: 2025-08-12 10:19:53
Analysis Framework Version: 2.0.0
Significance Level (α): 0.05

DATASET SUMMARY
--------------------------------------------------
Total Transactions: 7,483,766
Features: 34
Overall Fraud Rate: 0.2187 (21.87%)
Analysis Period: 2024-09-30 00:00:01.034820 to 2024-10-30 23:59:59.101885

MULTIPLE COMPARISON CORRECTION RESULTS
--------------------------------------------------
Number of Hypothesis Tests: 8
Family-wise Error Rate (FWER): 0.3366
False Discovery Rate (FDR): 0.0500

DETAILED HYPOTHESIS TEST RESULTS
--------------------------------------------------
Hypothesis                               Orig p     Bonf p     FDR p      Sidak p    Holm p     Significant    
------------------------------------------------------------------------------------------------------------------------
H1: Night vs Day Fraud Rates             0.000000   0.000000   0.000000   0.000000   0.000000   Bonf, FDR, Sidak, Holm
H2: Weekend vs Weekday Fraud Rates       0.943494   1.000000   0.943494   1.000000   1.000000   None           
H3: Bimodality (Dip Test)                0.000000   0.000000   0.000000   0.000000   0.000000   Bonf, FDR, Sidak, Holm
H3: Bimodality (Chi-square Concentrati   0.919543   1.000000   0.943494   1.000000   1.000000   None           
H3: Bimodality (Chi-square Independenc   0.000000   0.000000   0.000000   0.000000   0.000000   Bonf, FDR, Sidak, Holm
H4: Channel-based Fraud Analysis         0.000000   0.000000   0.000000   0.000000   0.000000   Bonf, FDR, Sidak, Holm
H5: ML ROI Analysis                      0.001000   0.008000   0.001600   0.007972   0.004000   Bonf, FDR, Sidak, Holm
H6: Dynamic Threshold Effectiveness      0.710451   1.000000   0.943494   0.999951   1.000000   None           

INDIVIDUAL HYPOTHESIS ANALYSIS
--------------------------------------------------
Hypothesis 1: Temporal Fraud Patterns (Night vs Day)
  Test: Z-test for proportions (one-tailed)
  Sample Sizes: Night-1,486,393, Day-5,997,373
  Fraud Rates: Night-0.5028, Day-0.1246
  Test Statistic: 1032.5336
  P-value: 0.000000
  Effect Size (Cohen's h): 0.8549
  Decision: REJECT H0
  Business Impact: Night fraud rate is 4.04x higher

Hypothesis 2: Weekend vs Weekday Fraud Patterns
  Test: Z-test for proportions (two-tailed)
  Sample Sizes: Weekend-1,929,663, Weekday-5,554,103
  Fraud Rates: Weekend-0.1997, Weekday-0.1997
  Actual Increase: 0.01%
  Target Range: 25-40%
  P-value: 0.943494
  Practical Significance: NO

Hypothesis 3: Bimodality of Fraud Transaction Amounts
  Sample Size: 1,494,719 fraudulent transactions
  Amount Statistics: Mean-$118,773.59, Median-$5,626.06
  Extreme Percentile Concentration: 6.0%
  Supporting Tests: 2/3
  Evidence Strength: STRONG

ECONOMIC IMPACT ANALYSIS
--------------------------------------------------
Total Potential Savings: $137,187,797,456.10
Total Implementation Costs: $600,000.00
Overall ROI: 22864532.9%

KEY BUSINESS RECOMMENDATIONS:
  1. Implement enhanced fraud monitoring during night hours (0-5 AM)
  2. Proceed with ML model implementation using optimistic scenario (ROI: 4396973.0%)

STATISTICAL CONCLUSIONS
--------------------------------------------------
After multiple comparison correction:
  - 5/8 hypotheses remain significant (Bonferroni)
  - 5/8 hypotheses remain significant (FDR)
  - Strong statistical evidence supports implementing fraud detection improvements

----------------------------------------------------------------------------------------------------