# US Credit Policy References

Public sources behind every rule in `credit_rules.py`. Each rule carries an
`authority` field indicating which of the three categories below it belongs to.

| Authority | Meaning |
|---|---|
| `regulation` | Binding federal rule or interagency supervisory policy |
| `underwriting` | Investor/agency guideline lenders follow contractually |
| `guidance` | Consumer education or scoring convention with no legal force |

Conflating these would misrepresent the strength of a finding in an audit, which
is why the distinction is carried through into every result the agent emits.

---

## 1. Equal Credit Opportunity Act — Regulation B, 12 CFR § 1002.9

**Issuer:** Consumer Financial Protection Bureau
**Link:** https://www.consumerfinance.gov/rules-policy/regulations/1002/9/
**eCFR:** https://www.ecfr.gov/current/title-12/chapter-X/part-1002/subpart-A/section-1002.9

Requires notification of action taken within 30 days of a completed application.
Where the action is adverse, the creditor must give a statement of the **specific
principal reasons**. Critically:

- Statements that the decision rested on internal standards or policies, or that
  the applicant failed to achieve a qualifying score, are **insufficient**.
- The reasons disclosed must relate to and accurately describe the factors
  actually considered or scored.
- No fixed number is mandated, but the official interpretation notes that
  disclosing **more than four reasons is unlikely to be helpful**.

**Why it shapes this system:** this is the reason the breach determination is a
deterministic threshold comparison rather than semantic retrieval. A vector
store can surface a passage that reads relevant without establishing that
anything was breached, which cannot support a specific-reason disclosure.
`adverse_action_reasons()` caps output at four for the same reason.

---

## 2. Uniform Retail Credit Classification and Account Management Policy

**Issuer:** FFIEC, on behalf of the Federal Reserve, FDIC, OCC and OTS
**Link:** https://www.federalregister.gov/documents/2000/06/12/00-14704/uniform-retail-credit-classification-and-account-management-policy
**OCC Bulletin 2000-20:** https://www.occ.gov/news-issuances/bulletins/2000/bulletin-2000-20.html
**Federal Reserve:** https://www.federalreserve.gov/frrs/guidance/uniform-retail-credit-classification-and-account-management-policy.htm

- Open- and closed-end retail loans **90 cumulative days past due** are
  classified **substandard**.
- **Closed-end** loans must be charged off at **120 days** past due.
- **Open-end** credit (e.g. credit cards) must be charged off at **180 days**.

Also covers classification of delinquent residential mortgage and home equity
loans, charge-off for bankrupt and deceased obligors and fraud, and limits on
re-aging open-end credit.

**Rules citing it:** `DELINQ-90-SUBSTANDARD`, `DELINQ-30-REPORTABLE`, `DELINQ-COUNT`

---

## 3. Fannie Mae Selling Guide B3-6-02 — Debt-to-Income Ratios

**Issuer:** Fannie Mae
**Link:** https://selling-guide.fanniemae.com/sel/b3-6-02/debt-income-ratios

- **36%** — base maximum total DTI for manually underwritten loans.
- **45%** — permitted ceiling where the borrower meets the credit score and
  reserve requirements in the Eligibility Matrix.
- **50%** — maximum for loan casefiles underwritten through Desktop Underwriter.

**Rules citing it:** `DTI-36-MANUAL`, `DTI-45-CEILING`

Two thresholds rather than one, because crossing 36% means the file leaves
standard eligibility while crossing 45% means it is outside agency parameters
altogether. Collapsing them would lose that distinction.

---

## 4. Ability-to-Repay / Qualified Mortgage — Regulation Z, 12 CFR § 1026.43

**Issuer:** Consumer Financial Protection Bureau
**2020 General QM Final Rule:** https://www.federalregister.gov/documents/2020/12/29/2020-27567/qualified-mortgage-definition-under-the-truth-in-lending-act-regulation-z-general-qm-loan-definition
**CFPB rule page:** https://www.consumerfinance.gov/rules-policy/final-rules/qualified-mortgage-definition-under-truth-lending-act-regulation-z-general-qm-loan-definition/

Creditors must make a reasonable, good-faith determination of a consumer's
ability to repay.

> **Important correction to a widely repeated figure.** The General QM
> definition's **43% DTI limit was removed** by the December 2020 final rule and
> replaced with a **price-based threshold**: a loan qualifies where the APR
> exceeds the average prime offer rate for a comparable transaction by less than
> 2.25 percentage points, with higher thresholds for smaller loan amounts,
> certain manufactured-housing loans and subordinate liens. The Bureau's
> reasoning was that price is a more holistic indicator of repayment ability
> than DTI alone.
>
> Do **not** cite 43% DTI as a current QM limit — it is stale. This project uses
> the Fannie Mae figures above for DTI instead, which are current.

**Rules citing it:** `THIN-FILE`, `LOAN-COUNT`

---

## 5. Truth in Lending Act — Regulation Z, 12 CFR § 1026.7(b)(12)

**Issuer:** Consumer Financial Protection Bureau
**Link:** https://www.consumerfinance.gov/rules-policy/regulations/1026/7/

Requires the minimum-payment warning on credit card periodic statements —
disclosing how long repayment takes and what it costs when only the minimum is
paid.

**Rule citing it:** `MINPAY-ONLY`

The rule exists because Congress treated minimum-payment behaviour as
consequential enough to mandate a standing disclosure, which is a reasonable
basis for treating it as a risk marker.

---

## 6. CFPB — How do I get and keep a good credit score?

**Issuer:** Consumer Financial Protection Bureau
**Link:** https://www.consumerfinance.gov/ask-cfpb/how-do-i-get-and-keep-a-good-credit-score-en-318/

Advises keeping credit use at **no more than 30%** of the total credit limit.

> **This is guidance, not a threshold in law.** Utilization is scored on a
> sliding scale and there is no cliff at 30% — 31% is not materially different
> from 29%. Published data shows median utilization above 40% among consumers
> scoring 660–719, so exceeding 30% is common among borrowers with acceptable
> credit. The rule is graded `Medium` severity and flagged `guidance` for
> exactly this reason, and is most meaningful in combination with delinquency
> rather than alone.

**Rule citing it:** `UTIL-30-GUIDANCE`

---

## 7. FICO — Amounts Owed score factor

**Issuer:** Fair Isaac Corporation
**Link:** https://www.myfico.com/credit-education/blog/credit-score-factor-amounts-owed

Amounts owed is **30% of a FICO Score**, the second-largest component. The
highest-scoring consumers typically carry utilization **below 10%**.

**Rules citing it:** `UTIL-70-SEVERE`, `MIX-BAD`, `INQUIRY-6`

---

## Adding your own policy documents

`credit_rules.py` holds thresholds and inline text; `retrieval.py` builds an
optional FAISS index over that text. To index real documents instead, extend
`build_policy_index()` with a document loader and keep the `rule_id` metadata
key so retrieved passages stay tied to the rule they explain.

Thresholds are starting values drawn from published sources. Tune them against
your own portfolio and record why — the `rationale` field on each rule is the
place for that, since it flows through into the generated report.
