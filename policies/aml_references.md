# US AML Policy References

Sources behind every rule in `aml_rules.py`. Each rule carries an `authority`
field: `regulation` (binding — a filing obligation exists) or `guidance`
(published red-flag indicator, informative but creating no obligation).

That distinction is load-bearing. FinCEN has stated explicitly that a transaction
at or near $10,000 does **not** by itself require a SAR — there must be knowledge
or suspicion of evasion. Reporting an advisory red flag as though it were a legal
trigger would overstate the finding, which is why `regulatory_breach_count` is
returned separately from the total.

---

## 1. Currency Transaction Reports — 31 CFR § 1010.311

**Issuer:** FinCEN
**Link:** https://www.ecfr.gov/current/title-31/subtitle-B/chapter-X/part-1010/subpart-C/section-1010.311

A CTR is required for currency transactions of **more than $10,000** by, through,
or to the bank. Multiple currency transactions must be treated as a single
transaction where the institution knows they are by or on behalf of the same
person and together exceed $10,000 in one business day.

**Rule:** `CTR-10K`

> **Scope matters.** This applies to *currency* transactions. A $12,000 ACH
> transfer or wire is not a currency transaction and creates no CTR obligation.
> The rule checks payment format before amount for that reason — a rule that
> fired on every large payment would be wrong more often than it was right.

---

## 2. Suspicious Activity Reports — 31 CFR § 1020.320

**Issuer:** FinCEN
**Link:** https://www.ecfr.gov/current/title-31/subtitle-B/chapter-X/part-1020/subpart-C/section-1020.320

Banks must file a SAR where a transaction (or series) **aggregating $5,000 or
more** is known or suspected to involve illicit funds, to be designed to evade
BSA requirements, or to have no apparent lawful purpose.

**Rule:** `SAR-5K`

The $5,000 test applies to *aggregated* activity, which is why the agent produces
a customer-level roll-up and not only a per-transaction score. The roll-up sums
the value of flagged transactions, not total turnover — the statute concerns
suspicious activity, not volume.

---

## 3. Structuring — 31 U.S.C. § 5324

**Issuer:** United States Code
**Link:** https://uscode.house.gov/view.xhtml?req=granuleid%3AUSC-1999-title31-section5324&num=0&edition=1999
**FFIEC Appendix G (examiner guidance on structuring):** https://bsaaml.ffiec.gov/manual/Appendices/08

Unlawful to structure, assist in structuring, or attempt to structure
transactions to evade BSA reporting. **Criminal even where the underlying funds
are entirely lawful** — the offence is the evasion, not the money. Aggravated
penalties apply where the pattern exceeds $100,000 in twelve months.

**Rule:** `STRUCTURING` — three or more sub-$10,000 deposits into one account
within seven days aggregating past $10,000.

This is the one rule that is a direct computational restatement of a statute: the
`struct_band_cnt_7d` and `struct_band_sum_7d` features exist specifically to
evidence it.

---

## 4. FFIEC BSA/AML Examination Manual, Appendix F — Red Flags

**Issuer:** FFIEC
**Link:** https://bsaaml.ffiec.gov/manual/Appendices/07

Published catalogue of money-laundering and terrorist-financing indicators. The
manual is explicit that the list is not exhaustive and that no single indicator
is conclusive alone — which is why these rules are `guidance` severity and why
the agent reports them as supporting evidence rather than as findings.

**Rules:** `RAPID-PASSTHROUGH`, `VELOCITY-SPIKE`

---

## 5. FinCEN Advisory FIN-2014-A005 — Funnel Accounts and TBML

**Issuer:** FinCEN, 28 May 2014
**Link:** https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2014-a005

Defines a **funnel account**: an individual or business account in one geographic
area that receives multiple cash deposits, *often in amounts below the cash
reporting threshold*, from which funds are withdrawn in a different geographic
area with little time elapsing between deposits and withdrawals.

**Rule:** `FUNNEL-ACCOUNT`

The advisory's own caveat is worth carrying: some funnel-account and TBML red
flags are, in the right circumstances, entirely legitimate activity, and no
single one indicates laundering by itself.

---

## 6. FinCEN Advisory FIN-2020-A003 — Imposter Scams and Money Mule Schemes

**Issuer:** FinCEN, 7 July 2020
**Link:** https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2020-a003

Red-flag indicators for money mule activity — accounts receiving funds from
multiple unrelated parties and rapidly forwarding them onward. Issued in a
COVID-19 fraud context, but the mule indicators generalise.

**Rule:** `MULE-FAN-IN`

FinCEN requested that institutions reference this advisory in SAR field 2 and the
narrative using the key term `COVID19 MM FIN-2020-A003` where applicable — worth
knowing if any of this ever feeds a real filing workflow.

---

## 7. FATF 40 Recommendations

**Issuer:** Financial Action Task Force
**Link:** https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html

International AML/CFT standards. Influential on US supervisory expectations but
**not directly enforceable** in the United States, hence `guidance` authority.

**Rule:** `LAYERING-CYCLE` — circular flows serve no commercial purpose and are a
classic layering technique.

---

## Mapping rules to typologies

| Injected typology | Rules expected to fire |
|---|---|
| `STRUCTURING` | `STRUCTURING`, `SAR-5K`, possibly `FUNNEL-ACCOUNT` |
| `FAN_IN` | `MULE-FAN-IN`, `SAR-5K` |
| `FAN_OUT` | `RAPID-PASSTHROUGH`, `VELOCITY-SPIKE` |
| `SCATTER_GATHER` | `FUNNEL-ACCOUNT`, `RAPID-PASSTHROUGH` |
| `CYCLE` | `LAYERING-CYCLE` |

Useful as a sanity check: if a typology is detected by the model but no rule
fires for it, either the rule thresholds need tuning or the typology as generated
does not resemble the published pattern it is named after.

## Tuning

Thresholds in `aml_rules.py` are starting points drawn from the sources above.
Tune them against your own alert volume and record the reasoning in each rule's
`rationale` field, which flows through into the generated report.
