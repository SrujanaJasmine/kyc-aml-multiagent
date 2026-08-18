> **Demonstration run** — scenario 4 of 20: Laundering activity on file  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-04_laundering` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0x3c39 (Jonathan Stempelb, Developer) |
| Assessed | 2026-08-16 19:48 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 233 transaction(s), 180 days on file |
| CDA | Review | risk 0.3116 vs 0.28 threshold; 7 rule(s) breached (DELINQ-30-REPORTABLE, DELINQ-COUNT, MINPAY-ONLY, MIX-BAD) |
| AML | No action | score 0.0936 vs 0.9; roll-up 0/20 flagged, SAR aggregate False |

## Summary
The credit application of Jonathan Stempelb, an existing customer, has been assessed and flagged for review due to a high risk of default. The primary reason for this decision is the customer's history of delayed payments, with 15 instances of late payments, exceeding the threshold of 3 occurrences. 

## Credit Assessment
The credit assessment model assigned a probability of 31.16% to this application, which exceeds the threshold of 28%, indicating a high risk of default. The decision to review this application is based on this high risk and the breach of 7 policy rules. The top SHAP features driving this decision include the interest rate, which increases the risk, and the changed credit limit, which decreases the risk.

## Policy and Standards Breaches
The following policy rules were breached:
- Payment more than 30 days past due: The customer had a payment 38 days past due, exceeding the threshold of 30 days. This rule exists to identify delinquent payments that are reportable and may indicate a higher risk of default.
- More than 3 delayed payments on file: The customer had 15 delayed payments, exceeding the threshold of 3 occurrences. This rule exists to identify a pattern of late payments, which is a strong predictor of default.
- Paying only the minimum amount due: The customer paid only the minimum amount due, which is a marker of revolving-debt dependence and may indicate a higher risk of default.
- Poor credit mix: The customer has a poor credit mix, which limits the evidence of their ability to manage different credit forms and lowers confidence in their credit score.
- More than 6 hard credit inquiries: The customer had 7 hard credit inquiries, exceeding the threshold of 6 inquiries. This rule exists to identify credit-seeking behavior that may indicate over-extension or application fraud.
- More than 5 active loans: The customer had 9 active loans, exceeding the threshold of 5 loans. This rule exists to identify numerous simultaneous obligations that may fragment cash flow and make total exposure harder to verify.
- Credit history shorter than 24 months: The customer's credit history is shorter than 24 months, which may indicate insufficient repayment history for a statistically stable credit score.

## KYC and AML Findings
The KYC agent found that the customer is an existing customer with 8 credit applications and 233 transactions on file over 180 days. However, sanctions and watchlist screening was not performed. The AML agent found no breached rules and assigned a low probability of money laundering to the customer's transaction.

## Data Quality Caveats
There are no rules listed as not evaluated, indicating that all necessary data was available for the assessment.

## Recommended Next Steps
The reviewer should manually review the customer's credit application, taking into account the breached policy rules and the customer's credit history. The reviewer should also verify the customer's income and obligations to determine their ability to repay the loan. Additionally, the reviewer may want to consider obtaining additional information about the customer's credit-seeking behavior and their ability to manage multiple credit obligations.