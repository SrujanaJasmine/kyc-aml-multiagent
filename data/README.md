# Data

The datasets are **not committed** — they total roughly 22 GB and GitHub rejects
any single file over 100 MB. Download or regenerate them locally into this layout:

```
data/
├── aml/                    IBM synthetic AML transactions (LI-Small / LI-Medium / LI-Large)
│   ├── LI-Small_Trans.csv          LI-Small_accounts.csv
│   ├── LI-Medium_Trans.csv         LI-Medium_accounts.csv
│   └── LI-Large_Trans.csv          LI-Large_accounts.csv
├── credit/                 Credit score classification dataset
│   ├── train.csv
│   └── test.csv
├── cleaned_credit/         Output of EDA/credit_eda.ipynb
│   ├── cleaned_train.csv
│   └── cleaned_test.csv
└── simulated/              Output of EDA/AML_to_Credit_Transaction_Generator.ipynb
    ├── customer_profile_train.csv      customer_transactions_train.csv
    └── customer_profile_test.csv       customer_transactions_test.csv
```

## Sources

- **`aml/`** — IBM Transactions for Anti Money Laundering (Kaggle: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`). Only the LI-* (lower-illicit) files are used.
- **`credit/`** — Credit Score Classification dataset (Kaggle: `parisrohan/credit-score-classification`).

## Regenerating the derived files

1. `EDA/credit_eda.ipynb` → produces `cleaned_credit/`
2. `EDA/AML_to_Credit_Transaction_Generator (2).ipynb` → joins AML transactions onto credit customers, produces `simulated/`
3. `agents/customer_db.py` → loads the CSVs into `customer_data.db`

## A note on size

`LI-Large_Trans.csv` alone is ~16 GB. Read it in chunks
(`pd.read_csv(..., chunksize=...)`) rather than loading it into memory.
