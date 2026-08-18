> **Demonstration run** — scenario 14 of 20: Established customer, clean profile  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-14_clean` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0x104a (Leahk, Mechanic) |
| Assessed | 2026-08-16 19:52 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 275 transaction(s), 179 days on file |
| CDA | Review | risk 0.097 vs 0.28 threshold; 1 rule(s) breached (DELINQ-COUNT) |
| AML | Escalate for analyst review | score 0.0109 vs 0.9; SAR-5K; roll-up 0/1 flagged, SAR aggregate False |

## Summary
The credit application of Leahk, an existing customer, was assessed and determined to require a review due to a low model risk score of 9.7% but with one breached rule. The primary reason for this decision is the presence of more than three delayed payments on file, which is a significant behavioral predictor of default.

## Credit Assessment
The credit model assigned a probability of 9.7% to this application, which is below the threshold of 28%. However, the decision to review the application was made due to the breach of one rule. The top SHAP features driving this decision include Delay from due date, Outstanding Debt, Interest Rate, Credit Mix Standard, and Number of Credit Cards, which respectively decrease or increase the risk.

## Policy and Standards Breaches
One rule was breached: More than 3 delayed payments on file. The observed value was 6.0, exceeding the threshold of 3 occurrences. This standard exists because a repeated pattern of late payments is a strong predictor of default, and manual review is normally required regardless of the model score.

## KYC and AML Findings
The KYC agent found that Leahk is an existing customer with 8 credit applications and 275 transactions on file over 179 days. The AML agent identified a transaction aggregating $7,302.04, which exceeds the $5,000 threshold, requiring a Suspicious Activity Report (SAR) filing if suspicion of illicit funds is present. Although the model score is low (1.1%), the transaction's value crosses a regulatory threshold, prompting a recommendation for analyst review.

## Data Quality Caveats
No data quality issues were reported, as all necessary information for the assessment was available.

## Recommended Next Steps
The reviewer should manually review the application, focusing on the delayed payments and the large transaction identified by the AML agent. Additionally, the reviewer should consider the customer's history and the model's low risk score when making a final decision. The AML finding should be escalated for analyst review to determine if a SAR filing is required.