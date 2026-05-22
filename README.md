# # Credit Risk Prediction in P2P Lending

A statistical learning project on the Bondora public loan dataset.

This project studies credit risk prediction in peer-to-peer lending using the Bondora public loan dataset. The goal is not only to compare predictive models, but to diagnose why standard credit-risk pipelines can fail: default labels are constructed, calibration depends on the prediction horizon, and observed outcomes can be affected by censoring. The project evaluates logistic regression, LightGBM, TabPFN-3, and Bondora's own Probability of Default score, and uses conformal prediction as a methodology layer for uncertainty quantification.

## Research Questions

1. How sensitive are credit-risk results to the definition of default?
2. Why can a model have acceptable AUC but poor probability calibration?
3. How does prediction horizon affect the interpretation of default probability?
4. Can conformal prediction provide a more honest uncertainty framework under distribution shift?

## Dataset

The analysis uses the Bondora public loan dataset.

- Period: 2009–2024 H2
- Sample: 264,369 closed loans
- Countries included: Estonia (EE), Finland (FI), Spain (ES)
- Countries excluded: Netherlands (NL) and Slovakia (SK), due to small sample size
- Unit of observation: loan-level record

