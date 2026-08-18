> **Demonstration run** — scenario 9 of 20: Elevated credit risk  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-09_high_risk_credit` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0xb986 (Simon Jessope, Developer) |
| Assessed | 2026-08-16 19:50 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 209 transaction(s), 180 days on file |
| CDA | Review | risk 0.5334 vs 0.28 threshold; 6 rule(s) breached (DELINQ-30-REPORTABLE, DELINQ-COUNT, UTIL-30-GUIDANCE, MINPAY-ONLY) |
| AML | No action | score 0.0 vs 0.9; roll-up 0/1 flagged, SAR aggregate False |

## Summary
The credit application of Simon Jessope, a 14-year-old existing customer with a history of 8 prior applications and 209 transactions, has been assessed. The outcome is a "Review" decision due to a high risk label, primarily driven by the customer's payment history and credit utilization. The most significant reason for this decision is the customer's delayed payments, with 57 days past due, exceeding the 30-day threshold.

## Credit Assessment
The credit model score is 0.5334, which is above the threshold of 0.28, indicating a high risk. The decision to review the application is based on this score, which falls in the manual-review band. The top SHAP features driving this decision are the delay from the due date, the number of credit cards, and the number of credit inquiries, all of which increase the risk.

## Policy and Standards Breaches
The customer has breached several rules, including:
* Payment more than 30 days past due, with an observed value of 57 days, exceeding the threshold of 30 days. This rule exists to identify delinquency, which becomes reportable after 30 days.
* More than 3 delayed payments on file, with an observed value of 21, exceeding the threshold of 3 occurrences. This rule exists to identify a repeated pattern of late payments, which is a strong predictor of default.
* Revolving utilization above 30%, with an observed value of 30.19%, slightly exceeding the threshold of 30%. This rule exists to advise keeping credit utilization at or below 30% of the total limit.
* Paying only the minimum amount due, with an observed value of "Yes", exceeding the threshold of "Payment_of_Min_Amount = Yes". This rule exists to identify minimum-payment behavior, which is a marker of revolving-debt dependence.
* Poor credit mix, with an observed value of "Bad", exceeding the threshold of "Credit_Mix = Bad". This rule exists to identify a narrow or poorly performing mix of account types, which limits evidence of the borrower's ability to manage different credit forms.
* More than 6 hard credit inquiries, with an observed value of 10, exceeding the threshold of 6 inquiries. This rule exists to identify credit-seeking behavior, which historically precedes over-extension and application fraud.

## KYC and AML Findings
The KYC agent found that the customer is an existing customer with a history of transactions and applications. The AML agent did not flag any suspicious activity, with a model probability of 0.0 and no breached rules.

## Data Quality Caveats
There are no rules listed as not evaluated, indicating that all relevant data was available for assessment.

## Recommended Next Steps
Based on the high risk label and policy breaches, it is recommended that the reviewer manually reviews the application, focusing on the customer's payment history, credit utilization, and credit mix. The reviewer should also consider the customer's age and occupation, as well as the number of credit cards and inquiries, to determine the best course of action. Additionally, the reviewer may want to consider verifying the customer's income and employment status to ensure that they can afford the credit being applied for.