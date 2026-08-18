> **Demonstration run** — scenario 17 of 20: Established customer, clean profile  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-17_clean` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0x10f9 (Langep, Lawyer) |
| Assessed | 2026-08-16 19:53 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 251 transaction(s), 180 days on file |
| CDA | Review | risk 0.0067 vs 0.28 threshold; 2 rule(s) breached (DELINQ-COUNT, UTIL-30-GUIDANCE) |
| AML | Escalate for analyst review | score 0.0 vs 0.9; SAR-5K; roll-up 0/1 flagged, SAR aggregate False |

## Summary
The credit application of Langep, an existing customer with 8 prior credit applications and 251 transactions on file, has been assessed. Despite the low model risk score of 0.67%, the application has been flagged for review due to the breach of two rules: more than 3 delayed payments on file and revolving utilization above 30%. The most significant reason for this decision is the repeated pattern of delayed payments, which is a strong predictor of default.

## Credit Assessment
The model score is 0.0067, which is below the threshold of 0.28, indicating a low risk. However, the decision is to review the application due to the breach of two rules. The top SHAP features driving this decision are the number of credit cards, outstanding debt, delay from due date, interest rate, and credit history age, all of which decrease the risk. However, the presence of 14 delayed payments and a revolving utilization of 30.01% outweigh these factors.

## Policy and Standards Breaches
Two rules were breached: 
- More than 3 delayed payments on file: The observed value is 14.0, which exceeds the threshold of 3 occurrences. This rule exists because a repeated pattern of delayed payments is a strong predictor of default.
- Revolving utilization above 30%: The observed value is 30.01%, which exceeds the threshold of 30%. This rule exists because the CFPB advises keeping usage at or below 30% of the total limit, and amounts owed is 30% of a FICO Score.

## KYC and AML Findings
The KYC check confirmed that the customer is an existing customer with a history of 8 credit applications and 251 transactions on file. The AML check flagged one regulatory threshold: activity aggregating $5,000 or more. The observed value is $11,717.59, which exceeds the threshold of $5,000. This rule exists because crossing $5,000 brings the activity within the SAR reporting range, and suspicion must also be present.

## Data Quality Caveats
There are no rules listed as not evaluated, and no data is missing. However, the sanctions and watchlist screening was not performed, which may weaken the assessment.

## Recommended Next Steps
The recommended next steps are to escalate the application for analyst review due to the breach of two rules and the crossing of one regulatory threshold. The analyst should review the application and consider the repeated pattern of delayed payments and the high revolving utilization in making a decision. Additionally, the analyst should consider the AML findings and determine if the transaction is suspicious and requires further investigation.