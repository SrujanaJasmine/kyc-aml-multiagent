# Commands

Every command needed to set up, train, evaluate and run the system, in order.
Written for Windows PowerShell; the only difference on macOS/Linux is the
virtualenv activation line, noted where it matters.

Run everything from the repository root unless a step says otherwise.

---

## Read this first

The datasets are **not in the repository** — they total roughly 22 GB and GitHub
rejects any single file over 100 MB. The trained models *are* committed, so you can
inspect them without retraining, but nothing will actually **run** until the data is
in place, because every entry point reads from `data/` or from the SQLite database
built out of it.

If you only want to see the system work end to end, the shortest honest path is:

1. [Download the two source datasets](#2-get-the-data) — Kaggle, free account
2. [Run the two notebooks](#3-regenerate-the-derived-data) that produce the derived CSVs
3. [Build the database](#4-build-the-database) and [its indexes](#5-add-the-transaction-indexes)
4. [`python graph.py`](#9-end-to-end-demonstration)

Steps 2 and 3 are the slow part. The transaction generator writes ~4.7 million rows
and takes a long time to run; there is no shortcut around it, because the simulated
transactions are what the AML side of the project is built on.

---

## 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10 or newer. `requirements.txt` pins the orchestration stack and leaves the
scientific stack unpinned — install once, confirm everything imports, then pin what
resolved:

```powershell
pip freeze | Select-String "xgboost|scikit-learn|joblib|shap|faiss|sentence-transformers|numpy"
```

### API key

```powershell
Copy-Item .env.example .env
notepad .env
```

Add a `GROQ_API_KEY` — free from [console.groq.com/keys](https://console.groq.com/keys).
It is only used by the explanation agent to write the narrative section of the
report. Without it the agent falls back to a deterministic report and everything else
runs unchanged, so a reviewer who does not want to register for a key can skip this.

---

## 2. Get the data

Two Kaggle datasets, downloaded into `data/`:

| Dataset | Kaggle slug | Lands in |
|---|---|---|
| IBM Transactions for Anti Money Laundering | `ealtman2019/ibm-transactions-for-anti-money-laundering-aml` | `data/aml/` |
| Credit Score Classification | `parisrohan/credit-score-classification` | `data/credit/` |

Only the `LI-*` (lower-illicit) files from the AML dataset are used. `LI-Large_Trans.csv`
alone is ~16 GB — read it in chunks, never whole.

Using the Kaggle CLI:

```powershell
pip install kaggle
# put kaggle.json in %USERPROFILE%\.kaggle\

kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data\aml --unzip
kaggle datasets download -d parisrohan/credit-score-classification -p data\credit --unzip
```

The expected final layout is in [`data/README.md`](data/README.md).

---

## 3. Regenerate the derived data

```powershell
jupyter lab
```

Run these two notebooks in order:

| Notebook | Produces | Notes |
|---|---|---|
| `EDA/credit_eda.ipynb` | `data/cleaned_credit/cleaned_train.csv`, `cleaned_test.csv` | Cleans the credit panel. Minutes. |
| `EDA/transaction_generator.ipynb` | `data/simulated/customer_profile_train.csv`, `customer_transactions_train.csv` | Fits the generative model to the IBM reference data and simulates 6 months of transactions for all 12,500 credit customers. **This is the long one.** It prints progress as it goes. |

`EDA/transaction_generator_v1_superseded.ipynb` is an earlier version kept for
reference. It is **not** the generator described in the paper — do not run it.

The generator ends with an assertion suite covering label integrity, label
learnability, graph structure, amount realism, split integrity and the volume model.
If any assertion fails the notebook stops rather than writing output; that is
intentional.

---

## 4. Build the database

```powershell
python -m database.customer_db
```

Loads the cleaned credit records and the simulated transactions into
`database/customer_data.db`. Expect a multi-GB file and several minutes.

---

## 5. Add the transaction indexes

One-off, and do it before any AML work. Without these, every AML assessment scans the
full 4.7-million-row transaction table.

```powershell
python -c "import sqlite3; c=sqlite3.connect('database/customer_data.db'); c.execute('CREATE INDEX IF NOT EXISTS idx_txn_from ON transactions(from_account)'); c.execute('CREATE INDEX IF NOT EXISTS idx_txn_to ON transactions(to_account)'); c.execute('CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id)'); c.commit(); c.close()"
```

Close every other connection to the database first — Jupyter kernels, DB Browser,
another Python shell. Building an index rewrites a large table, and an interruption
partway through leaves a hot journal that blocks all subsequent reads until SQLite
rolls it back.

If that happens: open the database once with **write** access
(`sqlite3.connect(...).execute("SELECT 1 FROM transactions LIMIT 1")`) and let SQLite
recover it. Do **not** delete the `-journal` file — that is what discards the rollback
and corrupts the table.

`graph.py` checks for these indexes and warns if they are missing. It does not create
them, deliberately: a demonstration should not write to the business database.

---

## 6. Splits

```powershell
python -m ml_models.data_splits --write
```

Partitions every customer into train 60% / validation 15% / test 15% / system 10% by
MD5 of the customer ID. The assignment is deterministic and identical on every
machine — there is no seed to drift. Splitting is by **customer**, not by row,
because the credit data is a monthly panel and a row split would put the same person
on both sides.

Nothing but the end-to-end demonstration and `evaluate_system.py` ever touches
`system`.

---

## 7. Credit model

```powershell
python -m ml_models.train_model
python -m ml_models.evaluate_model
```

`train_model.py` fits XGBoost on `train` and tunes the decision threshold on
`validation` for F1 — 0.28, not 0.5, because the classes are imbalanced. Writes to
`ml_models/artifacts/`:

```
xgboost_credit_model.joblib    feature_columns.joblib
best_threshold.joblib          training_metrics.json
```

`evaluate_model.py` scores the held-out split and writes ROC, precision-recall,
calibration, confusion and feature-importance figures to `reports/figures/`.

| Flag | Effect |
|---|---|
| `--split {test,val,train}` | Which split to score. Default `test`. |
| `--no-figures` | Metrics only, skip plotting. |
| `--tune N` | On `train_model.py`: run N Optuna trials inline. Prefer `tune_models.py` (step 10). |

---

## 8. AML model

```powershell
python -m ml_models.aml_features --cache
python -m ml_models.train_aml_model
python -m ml_models.evaluate_aml_model
```

**`aml_features --cache`** is the expensive step. It builds 54 features over the full
transaction graph — amount, time, format, trailing 1/7/30-day velocity, account
degree, structuring-band activity, and multi-hop pass-through and cycle membership —
and caches them to `data/features/`. Cache once; training and evaluation both read
from it.

The graph features are computed in pure pandas via bounded self-joins on
integer-coded edges rather than with IBM's Snap ML `GraphFeaturePreprocessor`, which
does not support Windows.

**`train_aml_model`** keeps every positive and downsamples negatives 200:1, then tunes
the threshold on the **full, non-downsampled** validation split so it reflects the
true base rate. It optimises F2 rather than F1: a missed filing is an enforcement
matter, a false positive costs an analyst a few minutes. It also prints the alert
volume each candidate threshold would generate across the whole 12,500-customer book,
which is the number that decides whether a threshold is operable.

| Flag | Effect |
|---|---|
| `--neg-per-pos N` | Negatives kept per positive during training. Default 200. |
| `--tune N` | N Optuna trials on validation PR-AUC, keeping the baseline for comparison. |
| `--split {test,val,system}` | On `evaluate_aml_model.py`. Default `test`. |
| `--threshold T` | Override the saved threshold when evaluating. |
| `--no-figures` | Metrics only. |

---

## 9. End-to-end demonstration

```powershell
python graph.py
python graph.py --cases 30
```

Runs 20 scenarios drawn from the `system` split — six with injected laundering, six
with poor credit profiles, six clean, and two unknown parties — through the full
orchestrator: fan-out to the three specialists in one parallel super-step, then
explanation, then audit.

This is the file to open first if you want to understand the architecture. The graph
definition is at the top; the demonstration below it is guarded by `__main__`.

Each run writes a folder under `reports/run_<timestamp>/` containing one Markdown and
one text report per case, plus `00_INDEX.md`. The audit trail goes to
`memory/system_memory.db` and `memory/agent_history.txt`.

---

## 10. Hyperparameter tuning (optional)

```powershell
python -m ml_models.tune_models --model credit --trials 100
python -m ml_models.tune_models --model aml --trials 60 --timeout 3600
python -m ml_models.tune_models --model both --fast
```

Fits a baseline with the default parameters **first**, searches a wide space with
Optuna's TPE sampler, and promotes the tuned model only if it beats that baseline on
validation. Writes comparison figures to `reports/figures/` and a report to
`reports/tuning/`.

| Flag | Effect |
|---|---|
| `--model {credit,aml,both}` | Which model to tune. Default `both`. |
| `--trials N` | Search budget. Default 100. |
| `--timeout S` | Wall-clock cap in seconds; stops mid-search and keeps the best so far. |
| `--fast` | Fewer trials on smaller subsets. Use this first to confirm it runs. |
| `--search-val-rows N` | Validation rows used *inside* the search. Default 120,000. |
| `--search-neg-per-pos N` | Negative ratio inside the search. Default 50. |
| `--always-promote` | Take the tuned model even if it loses to the baseline. Off by default. |

AML tuning is slow, and the reason is worth knowing: XGBoost re-scores the eval set
after every boosting round, so a 720k-row validation split gets scored up to 1,500
times per trial. `--search-val-rows` and `--search-neg-per-pos` shrink only what the
search sees; every reported number comes from a refit scored on the full split.

Start with `--fast` or a `--timeout`. A full `--model both --trials 100` run takes
hours.

---

## 11. Full-scale evaluation

```powershell
python evaluate_system.py
```

Scores all three agents against the entire `system` split — 9,392 credit
applications, 450,354 transactions, 1,174 customers, none of which contributed a row
to training — and ablates each into **model only**, **rules only** and **combined
agent logic**, scored identically.

The ablation is the point. It is what lets the architecture demonstrate what the
combination buys instead of asserting it, and the honest finding is that the rule
layer contributes almost nothing as a detector and a great deal as an abstention
mechanism.

| Flag | Effect |
|---|---|
| `--aml-limit N` | Score only the first N transactions. Use for a quick check. |
| `--skip-kyc` | Skip the KYC pass — it is the slowest component, at one DB round-trip per customer. |
| `--no-figures` | Metrics only. |

Output goes to `reports/evaluation_<timestamp>/`: `metrics.json`, Markdown tables,
per-item CSVs, and confusion / precision-recall figures.

---

## 12. Everything, in order

Copy-paste for a clean machine, after steps 1–3 are done:

```powershell
.venv\Scripts\activate

python -m database.customer_db
python -c "import sqlite3; c=sqlite3.connect('database/customer_data.db'); c.execute('CREATE INDEX IF NOT EXISTS idx_txn_from ON transactions(from_account)'); c.execute('CREATE INDEX IF NOT EXISTS idx_txn_to ON transactions(to_account)'); c.execute('CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id)'); c.commit(); c.close()"

python -m ml_models.data_splits --write

python -m ml_models.train_model
python -m ml_models.evaluate_model

python -m ml_models.aml_features --cache
python -m ml_models.train_aml_model
python -m ml_models.evaluate_aml_model

python evaluate_system.py
python graph.py
```

If the credit side is the only thing that changed, steps 7 and 11 are enough — the
AML feature cache does not need rebuilding.

---

## 13. What lands where

| Output | Path |
|---|---|
| Trained models, thresholds, feature contracts, training metrics | `ml_models/artifacts/` |
| Cached AML features | `data/features/` |
| Model figures — ROC, PR, calibration, confusion, importance, threshold sweeps | `reports/figures/` |
| Tuning write-ups and baseline-vs-tuned comparisons | `reports/tuning/`, `reports/figures/tuning_*` |
| Full evaluation: tables, `metrics.json`, per-item CSVs | `reports/evaluation_<timestamp>/` |
| Demonstration reports, one per case, plus an index | `reports/run_<timestamp>/` |
| Business data — customers, applications, transactions | `database/customer_data.db` |
| Audit trail and LangGraph checkpoints | `memory/system_memory.db`, `memory/agent_history.txt` |

---

## 14. Troubleshooting

**`sqlite3.OperationalError: database is locked` or reads hanging after an
interrupted index build.** A hot journal is present. Open the database once with
write access and let SQLite roll it back — do not delete the `-journal` file:

```powershell
python -c "import sqlite3; c=sqlite3.connect('database/customer_data.db'); print(c.execute('SELECT count(*) FROM transactions').fetchone()); c.close()"
```

**`KeyError` naming a feature column.** The model's feature contract and the frame
being scored have diverged. `ml_models/feature_mapping.py` has a `mapping_coverage()`
diagnostic that reports which columns are missing, extra, or leaking. Retrain if the
schema genuinely changed.

**`FileNotFoundError` on a `.joblib` under `ml_models/artifacts/`.** The model has not
been trained yet. Run step 7 (credit) or step 8 (AML).

**AML feature caching runs out of memory.** Lower the chunk size in
`ml_models/aml_features.py`, or cache one split at a time with `--split`.

**The explanation agent returns a plain report with no narrative.** `GROQ_API_KEY` is
missing or the API call failed. This is the designed fallback — a compliance run must
not depend on an external service's uptime — and everything else in the report is
unaffected.

**Optuna is not installed.** Only `--tune` needs it. Both training scripts run without
it and fall back to their default parameter blocks.

---

## 15. Caveats on any number this produces

- **The AML labels are synthetic.** Typologies are injected by the transaction
  generator. The results show the pipeline detects the patterns it was built to
  detect; they are not evidence of real-world detection performance.
- **Laundering prevalence is set at 3% of customers** to make the problem trainable.
  That is far above any real portfolio, so precision in particular will not transfer.
- **Sanctions screening is not implemented.** `kyc_agent` reports
  `screening_performed: false` rather than returning a plausible-looking empty match
  list, because an empty result would read as "screened, nothing found".
- **Policy thresholds are starting points** drawn from published US sources. Several
  fire on a large share of this dataset, which says more about the dataset's
  distributions than about the applicants. `evaluate_system.py` quantifies exactly
  this.
