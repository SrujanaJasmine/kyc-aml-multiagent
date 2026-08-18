# FRAML — Multi-Agent Fraud, AML and Credit Risk Assessment

A LangGraph orchestrator–worker system for financial compliance. A supervisor node
fans a batch of assessment items out to specialist agents in parallel; each agent
combines a machine-learning model with deterministic policy rules; their findings
merge into one plain-English compliance report and a queryable audit trail.

The system answers three questions about a customer at once, *is this party known
to us*, *should this credit application be approved*, and *does this transaction
look like money laundering* and routes the case to a human wherever the model and
the written rules disagree.



## Architecture

```
                        START
                          │
                  route_to_workers                 (Send-based fan-out)
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       kyc_agent     aml_agent     cda_agent
            └─────────────┼─────────────┘
                          ▼
                  explanation_agent                (writes the report)
                          ▼
                     audit_agent                   (persists the trail)
                          ▼
                         END
```

| Agent | Responsibility |
|---|---|
| `kyc_agent` | Establishes whether the party is an existing customer, a new one, or dormant, and summarises the depth of the relationship |
| `cda_agent` | Scores a credit application with XGBoost, tests it against US lending policy, and returns Approve / Review / Decline with Regulation B adverse-action reasons |
| `aml_agent` | Scores a transaction, tests it against BSA/AML obligations, and rolls the customer's recent activity up to the SAR aggregation threshold |
| `explanation_agent` | Synthesises the findings into a report — a deterministic verdict table plus an LLM-written narrative |
| `audit_agent` | Writes every agent call to SQLite and to a human-readable text log |

### The design idea worth knowing

Each risk agent carries **two independent assessors**: a gradient-boosted model and
a set of deterministic rules tied to published regulation. They answer different
questions — *how does this resemble past cases* versus *which written standard does
this breach* — and they fail differently. A model can miss a novel pattern; a
threshold cannot miss a $10,001 cash deposit. A model can flag something
inexplicable; a rule always states the observed value against the published limit.

Where they **disagree**, the case is escalated to a human rather than auto-decided.
That disagreement band is reported as a first-class metric in the evaluation.

Only the rules can support a regulatory disclosure. Under Regulation B a creditor
must give the *specific principal reasons* for adverse action, and "the model scored
it low" is explicitly insufficient.



## Repository layout
.
├── graph.py                    the graph definition + a 20-scenario demonstration
├── evaluate_system.py          full-scale evaluation of the decision layer
├── config.py                   .env loading and shared paths
│
├── agents/
│   ├── state.py                GraphState / WorkerState schemas
│   ├── orchestrator.py         Send-based fan-out to the workers
│   ├── kyc_agent.py            customer standing
│   ├── cda_agent.py            credit assessment
│   ├── aml_agent.py            transaction screening
│   ├── explanation_agent.py    report synthesis
│   └── audit_agent.py          audit persistence
│
├── policies/
│   ├── credit_rules.py         12 lending rules, US sources
│   ├── references.md           full citations for the credit rules
│   ├── aml_rules.py            8 BSA/AML rules
│   ├── aml_references.md       full citations for the AML rules
│   └── retrieval.py            optional FAISS lookup of policy wording
│
├── ml_models/
│   ├── credit_risk_model.py    CreditScorer — the credit model's inference contract
│   ├── feature_mapping.py      customer-table fields → model feature names
│   ├── data_splits.py          deterministic four-way split by customer
│   ├── train_model.py          credit model training
│   ├── evaluate_model.py       credit metrics + figures
│   ├── aml_features.py         54 AML features incl. graph and velocity
│   ├── train_aml_model.py      AML model training
│   ├── evaluate_aml_model.py   AML metrics + figures
│   ├── tune_models.py          Optuna search for both models
│   └── artifacts/              trained models, thresholds, metrics
│
├── database/customer_db.py     schema, CSV loaders, accessors
├── memory/memory.py            LangGraph checkpoints + agent_history table
├── COMMANDS.md                 every command, in order, with expected outputs
├── EDA/                        notebooks: credit EDA, transaction simulation
├── data/                       datasets (gitignored — see data/README.md)
└── reports/                    generated figures, evaluations and demo runs


### Two databases, deliberately separate

- **`database/customer_data.db`** - *what* the system reasons about: customers,
  credit applications, transactions.
- **`memory/system_memory.db`** - *how* the system ran: LangGraph checkpoints and
  the `agent_history` audit table.

Keeping them apart means the audit trail cannot be disturbed by business-data
operations, and either can be rebuilt without the other.



## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add a `GROQ_API_KEY` (free from
[console.groq.com/keys](https://console.groq.com/keys)). Without it the explanation
agent falls back to a deterministic report rather than failing.

Datasets are not in the repository — see [`data/README.md`](data/README.md) for
sources and the expected layout.



## Running it

The short version is below. [`COMMANDS.md`](COMMANDS.md) has the full runbook — every
flag, what each step writes, and what to do when something breaks.

```powershell
# 1. build the database from the CSVs
python -m database.customer_db

# 2. split, train and evaluate the credit model
python -m ml_models.data_splits --write
python -m ml_models.train_model
python -m ml_models.evaluate_model

# 3. build AML features, train and evaluate
python -m ml_models.aml_features --cache
python -m ml_models.train_aml_model
python -m ml_models.evaluate_aml_model

# 4. optional: hyperparameter search for either model
python -m ml_models.tune_models --model credit --trials 100
python -m ml_models.tune_models --model aml --trials 60 --timeout 3600

# 5. full-scale evaluation on the held-out split
python evaluate_system.py

# 6. end-to-end demonstration, 20 scenarios
python graph.py
```

One-off, and worth doing before step 6 — without these indexes each AML assessment
scans the full transaction table:

```powershell
python -c "import sqlite3; c=sqlite3.connect('database/customer_data.db'); c.execute('CREATE INDEX IF NOT EXISTS idx_txn_from ON transactions(from_account)'); c.execute('CREATE INDEX IF NOT EXISTS idx_txn_to ON transactions(to_account)'); c.execute('CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id)'); c.commit()"
```

### What lands where

| Output | Path |
|---|---|
| Demonstration reports (Markdown + text, with an index) | `reports/run_<timestamp>/` |
| Full evaluation, tables and per-item CSVs | `reports/evaluation_<timestamp>/` |
| Model figures — ROC, PR, confusion, thresholds, importance | `reports/figures/` |
| Tuning write-ups | `reports/tuning/` |
| Audit trail | `memory/system_memory.db`, `memory/agent_history.txt` |


## Evaluation design

`ml_models/data_splits.py` partitions the labelled data **by customer** into
train 60% / validation 15% / test 15% / system 10%. Splitting by row would place the
same customer on both sides — the credit data is a monthly panel — and inflate every
metric. Assignment is an MD5 of the customer id, so the partition is identical on
every machine with no seed to drift.

Both models are fitted on `train` only. `system` is touched by nothing but the
end-to-end demonstration, so `graph.py` runs on data neither model has seen.

`evaluate_system.py` reports an **ablation** — model only, rules only, and the
agent's combined logic, scored identically — so the architecture has to demonstrate
what the combination buys rather than assert it.


## Data caveats

These matter for reading any number this project produces.

- **The AML labels are synthetic.** Laundering typologies are injected by
  `EDA/transaction_generator.ipynb`. The results show the pipeline detects the patterns it was
  built to detect; they are not evidence of real-world detection performance.
- **Laundering prevalence is set at 3% of customers** to make the problem trainable.
  That is far above any real portfolio, so precision in particular will not transfer.
- **Credit labels come from the source dataset** and are used as provided.
- **Sanctions screening is not implemented.** `kyc_agent` reports
  `screening_performed: false` rather than returning a plausible-looking empty match
  list, because an empty result would read as "screened, nothing found".
- **Policy thresholds are starting points** drawn from published sources. Several
  fire on a large share of this dataset, which says more about the dataset's
  distributions than about the applicants — the evaluation quantifies this.

---

## Sources

Policy rules are tied to published US sources with full citations in
[`policies/references.md`](policies/references.md) (credit) and
[`policies/aml_references.md`](policies/aml_references.md) (AML) — Regulation B,
the FFIEC BSA/AML Examination Manual, Fannie Mae's Selling Guide, 31 CFR Chapter X,
31 USC 5324, and FinCEN advisories.

Each rule carries an `authority` field — `regulation`, `underwriting` or `guidance` —
so binding obligations are never presented alongside advisory red flags as if they
carried the same weight.
