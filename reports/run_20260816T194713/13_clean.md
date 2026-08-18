> **Demonstration run** — scenario 13 of 20: Established customer, clean profile  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-13_clean` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0x1032 (Wahbap, Lawyer) |
| Assessed | 2026-08-16 19:52 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 441 transaction(s), 180 days on file |
| CDA | Review | risk 0.5639 vs 0.28 threshold; 2 rule(s) breached (DELINQ-COUNT, MINPAY-ONLY) |
| AML | Escalate for analyst review | score 0.0 vs 0.9; SAR-5K; roll-up 0/1 flagged, SAR aggregate False |

## Summary
The credit application of Wahbap, an existing customer with 8 prior credit applications and 441 transactions on file, has been assessed as high risk with a model probability of 56.4%, exceeding the threshold of 28%. The primary reason for this assessment is the customer's history of delayed payments and paying only the minimum amount due. 

## Credit Assessment
The model score for this application is 0.5639, which is above the model threshold of 0.28, resulting in a "High Risk" label and a decision to "Review". The top SHAP features driving this decision include Outstanding Debt, which increases the risk, and Credit Mix Standard, Num Bank Accounts, and Interest Rate, which decrease the risk. Payment Behaviour, specifically low-value payments, also increases the risk.

## Policy and Standards Breaches
Two policy breaches were identified: 
- More than 3 delayed payments on file, with an observed value of 9.0, exceeding the threshold of 3 occurrences. This standard exists because repeated late payments are a strong predictor of default.
- Paying only the minimum amount due, with an observed value of "Yes", exceeding the threshold of "Payment_of_Min_Amount = Yes". This standard exists because paying only the minimum barely amortizes principal and is a marker of revolving-debt dependence.

## KYC and AML Findings
The AML agent found one breached rule: 
- Activity aggregating $5,000 or more, with an observed value of $57,354.54, exceeding the threshold of $5,000 aggregate. This standard exists because transactions above $5,000 may involve illicit funds or be designed to evade BSA requirements.

## Data Quality Caveats
No rules were listed as not evaluated, indicating that the assessment is based on complete data.

## Recommended Next Steps
Based on the high-risk assessment and policy breaches, it is recommended that the application be escalated for analyst review to verify the customer's creditworthiness and ensure compliance with regulatory requirements. The analyst should carefully examine the customer's payment history, credit utilization, and transaction behavior to determine the appropriate course of action.