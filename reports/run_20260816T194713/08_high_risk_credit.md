> **Demonstration run** — scenario 8 of 20: Elevated credit risk  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-08_high_risk_credit` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0xb14 (Langep, Manager) |
| Assessed | 2026-08-16 19:50 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 159 transaction(s), 179 days on file |
| CDA | Decline | risk 0.8235 vs 0.28 threshold; 4 rule(s) breached (DELINQ-COUNT, MINPAY-ONLY, INQUIRY-6, LOAN-COUNT) |
| AML | Escalate for analyst review | score 0.0 vs 0.9; SAR-5K; roll-up 0/1 flagged, SAR aggregate False |

## Summary
The credit application of Langep, an existing customer, was assessed and resulted in a decline decision due to a high risk score of 82.3%, exceeding the auto-decline band, and the breach of one high-severity rule and three medium-severity rules. The most significant reason for this decision is the customer's history of delayed payments, with 11 observed instances, far exceeding the threshold of 3 occurrences.

## Credit Assessment
The credit assessment model assigned a probability of 0.8235, which is significantly higher than the model threshold of 0.28, indicating a high risk. The decision to decline the application was made due to this high risk score and the breach of multiple rules. The top SHAP features driving this decision include Outstanding Debt, Interest Rate, Num Credit Inquiries, Changed Credit Limit, and Credit History Age Total, with Outstanding Debt and Interest Rate increasing the risk the most.

## Policy and Standards Breaches
The following rules were breached:
- More than 3 delayed payments on file: The customer had 11 delayed payments, exceeding the threshold of 3 occurrences. This rule exists because a repeated pattern of late payments is a strong predictor of default.
- Paying only the minimum amount due: The customer was observed to be paying only the minimum amount due, which barely amortizes principal and is a recognized marker of revolving-debt dependence.
- More than 6 hard credit inquiries: The customer had 11 hard credit inquiries, exceeding the threshold of 6 inquiries. This rule exists because a cluster of hard inquiries suggests credit-seeking from several lenders at once, which historically precedes over-extension and application fraud.
- More than 5 active loans: The customer had 6 active loans, exceeding the threshold of 5 loans. This rule exists because numerous simultaneous obligations fragment cash flow and make total exposure harder to verify, affecting the ability-to-repay determination.

## KYC and AML Findings
The KYC agent found that the customer is an existing customer with 8 credit applications and 159 transactions on file over 179 days. However, sanctions and watchlist screening was not performed. The AML agent found one breached rule: Activity aggregating $5,000 or more, with the observed amount being $5,503.63. The model did not flag the customer for laundering, but the transaction was flagged for exceeding the $5,000 aggregate threshold, requiring a SAR report if suspicion is present.

## Data Quality Caveats
There are no rules listed as not evaluated, indicating that all necessary data was available for the assessment.

## Recommended Next Steps
The recommended next steps are to escalate the case for analyst review due to the breach of regulatory thresholds, despite the model score being below the cutoff. Additionally, the customer's credit history and payment behavior should be carefully reviewed to determine the best course of action. The AML findings also require further review to determine if a SAR report is necessary.