> **Demonstration run** — scenario 12 of 20: Elevated credit risk  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-12_high_risk_credit` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0x6f38 (Ransdellj, Entrepreneur) |
| Assessed | 2026-08-16 19:51 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 671 transaction(s), 179 days on file |
| CDA | Review | risk 0.4492 vs 0.28 threshold; 2 rule(s) breached (DELINQ-COUNT, UTIL-30-GUIDANCE) |
| AML | No action | score 0.0 vs 0.9; roll-up 0/1 flagged, SAR aggregate False |

## Summary
The credit application of Ransdellj, an existing customer, was assessed and determined to be high risk, with a model probability of 44.9% exceeding the threshold of 28%. The primary reason for this assessment is the customer's history of delayed payments, with 11 observed instances exceeding the threshold of 3 occurrences. 

## Credit Assessment
The credit model assigned a probability of 44.9% to this application, which is above the model threshold of 28%, resulting in a "High Risk" label and a decision to review the application. The top SHAP features driving this decision include a delay from the due date of 30 days, which increases the risk, and an interest rate of 12%, outstanding debt of $463.07, and a good credit mix, which all decrease the risk to varying extents.

## Policy and Standards Breaches
Two policy rules were breached in this assessment. The first breach is related to the number of delayed payments on file, with 11 observed instances exceeding the threshold of 3 occurrences. This rule is in place because a repeated pattern of late payments is a strong predictor of default. The second breach is related to revolving utilization, with an observed value of 37.03% exceeding the threshold of 30%. This rule is in place because high utilization can negatively impact credit scores, although it is considered a soft signal that is more relevant when combined with delinquency.

## KYC and AML Findings
The KYC agent found that the customer is an existing customer with 8 credit applications and 671 transactions on file over 179 days. However, sanctions and watchlist screening was not performed. The AML agent did not flag any suspicious activity, with a model probability of 0.0% and no breached rules. The top SHAP features for the AML model include a low degree of pair transactions, hop destination pass ratio, and degree pair share, which all lower suspicion.

## Data Quality Caveats
There are no rules listed as not evaluated, indicating that the assessment is based on complete data.

## Recommended Next Steps
Based on the high-risk label and policy breaches, it is recommended that the reviewer manually review the application, paying close attention to the customer's history of delayed payments and high revolving utilization. Additionally, the reviewer should consider verifying the customer's credit information and assessing their overall creditworthiness before making a decision on the application.