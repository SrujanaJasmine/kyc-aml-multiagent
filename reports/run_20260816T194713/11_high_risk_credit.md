> **Demonstration run** — scenario 11 of 20: Elevated credit risk  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-11_high_risk_credit` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0xacbe (Henryg, Media_Manager) |
| Assessed | 2026-08-16 19:51 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 217 transaction(s), 181 days on file |
| CDA | Decline | risk 0.7868 vs 0.28 threshold; 4 rule(s) breached (DELINQ-COUNT, UTIL-30-GUIDANCE, MINPAY-ONLY, LOAN-COUNT) |
| AML | No action | score 0.0 vs 0.9; roll-up 0/1 flagged, SAR aggregate False |

## Summary
The credit application of Henryg, an existing customer, was assessed and resulted in a decline decision due to a high risk score of 78.7%, which exceeds the auto-decline band, and the breach of one high-severity rule and three medium-severity rules. The most significant reason for this decision is the customer's history of delayed payments, with 9 observed instances, which is a strong predictor of default.

## Credit Assessment
The model score is 0.7868, which is above the model threshold of 0.28, indicating a high risk. The decision to decline the application is based on this high risk score and the breach of multiple rules. The top SHAP features driving this decision are Outstanding Debt, which increases the risk, and Interest Rate, which also increases the risk. Additionally, Payment Behaviour Low spent Small value payments and Delay from due date increase the risk, while Num Bank Accounts decreases the risk.

## Policy and Standards Breaches
The following rules were breached:
* More than 3 delayed payments on file: The observed value is 9.0, which exceeds the threshold of 3 occurrences. This rule is in place because a repeated pattern of late payments is a strong predictor of default.
* Revolving utilization above 30%: The observed value is 34.175190036257355, which exceeds the threshold of 30%. This rule is in place because high utilization can indicate a higher risk of default.
* Paying only the minimum amount due: The observed value is Yes, which exceeds the threshold of Payment_of_Min_Amount = Yes. This rule is in place because paying only the minimum amount due can indicate a higher risk of default.
* More than 5 active loans: The observed value is 6.0, which exceeds the threshold of 5 loans. This rule is in place because numerous simultaneous obligations can fragment cash flow and make total exposure harder to verify.

## KYC and AML Findings
The KYC agent found that the customer is an existing customer with 8 credit applications and 217 transactions on file over 181 days. The AML agent did not flag any suspicious activity, with a model probability of 0.0 and no breached rules.

## Data Quality Caveats
There are no rules listed as not evaluated, indicating that all necessary data was available for the assessment.

## Recommended Next Steps
Based on the findings, the recommended next steps are to decline the credit application due to the high risk score and the breach of multiple rules. Additionally, the customer's payment behaviour and credit utilization should be closely monitored to prevent further defaults. The customer may also be required to provide additional information or documentation to support their creditworthiness.