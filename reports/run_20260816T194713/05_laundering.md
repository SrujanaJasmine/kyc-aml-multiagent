> **Demonstration run** — scenario 5 of 20: Laundering activity on file  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-05_laundering` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0xa73b (Niveditak, Scientist) |
| Assessed | 2026-08-16 19:48 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 430 transaction(s), 180 days on file |
| CDA | Review | risk 0.0761 vs 0.28 threshold; 3 rule(s) breached (DELINQ-COUNT, UTIL-30-GUIDANCE, MINPAY-ONLY) |
| AML | Enhanced monitoring | score 0.9992 vs 0.9; LAYERING-CYCLE; roll-up 1/46 flagged, SAR aggregate False |

## Summary
The credit application of Niveditak, an existing customer, was assessed and determined to require review despite a low model risk score of 7.6%. The primary reason for this decision is the breach of three rules: more than three delayed payments on file, revolving utilization above 30%, and paying only the minimum amount due. 

## Credit Assessment
The credit model assigned a probability of 7.6% to this application, which is below the threshold of 28%, indicating a good/low risk. However, the decision to review the application was made due to the breach of three rules. The top SHAP features driving this decision include Outstanding Debt, which decreases the risk, Interest Rate, which decreases the risk, Credit History Age Total, which decreases the risk, Delay from due date, which decreases the risk, and Num Credit Card, which increases the risk.

## Policy and Standards Breaches
The application breached three rules:
- **More than 3 delayed payments on file**: The observed value is 8.0, exceeding the threshold of 3 occurrences. This rule exists because repeated late payments are a strong predictor of default.
- **Revolving utilization above 30%**: The observed value is 40.51%, exceeding the threshold of 30%. This rule is based on CFPB guidance, which advises keeping credit utilization at or below 30% of the total limit.
- **Paying only the minimum amount due**: The observed value is Yes, which exceeds the threshold. This rule exists because paying only the minimum amount barely amortizes principal and is a recognized marker of revolving-debt dependence.

## KYC and AML Findings
The AML agent found one breached rule:
- **Funds returning to their origin**: The observed value is a 3-hop cycle, which exceeds the threshold of membership in a 2- or 3-hop cycle. This rule exists because circular flows serve no commercial purpose and are a classic layering technique intended to obscure the origin of funds.

## Data Quality Caveats
No rules were listed as not evaluated, indicating that the assessment was based on complete data.

## Recommended Next Steps
Based on the findings, the reviewer should:
- Review the application manually to assess the creditworthiness of the customer despite the low model risk score.
- Investigate the breached rules, particularly the high number of delayed payments and the revolving utilization above 30%.
- Consider enhanced monitoring for the customer due to the AML model score of 99.9% and the breached rule related to funds returning to their origin.
- Verify the customer's information and transaction history to ensure accuracy and completeness.