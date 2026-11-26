# Day 9: Error Analysis and Explainability

## Objective
Investigate where the Logistic Regression model makes incorrect predictions and understand feature importance.

## Dataset
- Total records: 1000
- Features: 5
- Class 0: 894
- Class 1: 106
- Training records: 800
- Testing records: 200

## Model Performance
- Model: Logistic Regression
- Accuracy: 97.5%
- False Positives: 0
- False Negatives: 5

## Error Analysis
The model correctly classified all 179 Class 0 records but missed 5 out of 21 Class 1 records.

This shows that overall accuracy alone does not fully describe model performance on an imbalanced dataset.

## Feature Importance
Permutation importance using F1-score identified:
- Feature 3 as the most important feature.
- Feature 5 as the second most important feature.
- Other features had little or negative measured importance.

Permutation importance indicates how much model performance changes when a feature is shuffled. It does not establish causation.

## Proposed Improvements

1. **Class Weighting:** Use balanced class weights to improve minority-class recognition and evaluate the precision-recall trade-off.

2. **Threshold Tuning:** Test different classification thresholds to improve minority-class recall while monitoring precision.

## Output Files
- `error_analysis.csv`
- `feature_importance.csv`
- `group_performance.csv`
- `Day_09_Error_Analysis_Explainability_New.ipynb`

## Conclusion
The analysis revealed that the model performs well on the majority class but misses some minority-class records. Error analysis and feature importance provide deeper insights into model behaviour and help identify possible improvements.