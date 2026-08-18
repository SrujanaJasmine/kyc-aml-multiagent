> **Demonstration run** — scenario 10 of 20: Elevated credit risk  
> Data source: `system` split, held out from all model training.  
> Thread `demo-20260816T194713-10_high_risk_credit` · generated 2026-08-16 19:47 UTC.

# Compliance Assessment Report

| | |
|---|---|
| Customer | CUS_0x29fd (Steven C.g, Developer) |
| Assessed | 2026-08-16 19:50 UTC |
| Agents run | KYC, CDA, AML |

## Verdicts at a glance

| Agent | Verdict | Detail |
|---|---|---|
| KYC | Existing | 8 application(s), 251 transaction(s), 180 days on file |
| CDA | Decline | risk 0.7923 vs 0.28 threshold; 6 rule(s) breached (DELINQ-COUNT, UTIL-30-GUIDANCE, MINPAY-ONLY, MIX-BAD) |
| AML | No action | score 0.0 vs 0.9; roll-up 0/1 flagged, SAR aggregate False |

## Summary
The credit application of Steven C.g, an existing customer, was assessed and resulted in a decline decision due to a high model risk score of 79.2%, exceeding the auto-decline band, and the breach of one high-severity regulatory rule. The primary reason for this decision is the customer's history of delayed payments, with 18 observed instances, significantly exceeding the threshold of 3 occurrences.

## Credit Assessment
The credit model assigned a probability of 0.7923, which is above the model threshold of 0.28, indicating a high risk. The decision to decline the application was made due to this high risk score and the breach of multiple rules. The top SHAP features driving this decision include Outstanding Debt, Interest Rate, Number of Credit Cards, Number of Credit Inquiries, and Payment Behaviour, all of which increase the risk.

## Policy and Standards Breaches
Several rules were breached, including:
- More than 3 delayed payments on file: 18 observed instances exceeded the threshold of 3 occurrences. This rule exists because repeated late payments are a strong predictor of default.
- Revolving utilization above 30%: 33.80% observed utilization exceeded the threshold of 30%. This guideline is in place as it is advised by the CFPB to keep credit utilization at or below 30% of the total limit.
- Paying only the minimum amount due: The customer was observed to be paying only the minimum amount due, which barely amortizes principal and is a marker of revolving-debt dependence.
- Poor credit mix: The customer's credit mix was observed to be bad, limiting evidence of their ability to manage different credit forms.
- More than 6 hard credit inquiries: 12 observed inquiries exceeded the threshold of 6 inquiries, suggesting credit-seeking from multiple lenders.
- More than 5 active loans: 6 observed active loans exceeded the threshold of 5 loans, fragmenting cash flow and making total exposure harder to verify.

## KYC and AML Findings
The KYC agent found that the customer is an existing customer with 8 credit applications and 251 transactions on file over 180 days. The AML agent did not flag any suspicious activity, with a model probability of 0.0 and no breached rules.

## Data Quality Caveats
There are no rules listed as not evaluated, indicating that all necessary data was available for the assessment.

## Recommended Next Steps
Based on the findings, it is recommended that the reviewer:
- Manually review the customer's credit history and application to verify the accuracy of the data.
- Consider the customer's ability to repay the loan, given their history of delayed payments and high credit utilization.
- Evaluate the customer's credit mix and consider whether it is suitable for the loan product being applied for.
- Monitor the customer's account activity for any suspicious transactions or behavior.