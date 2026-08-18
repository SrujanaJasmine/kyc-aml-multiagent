"""
policies package
================
Credit policy rules, their public US regulatory sources, and optional
retrieval of the underlying policy text.

    credit_rules.py   the rule engine and the thresholds it enforces
    references.md     full citations and links for every source cited
    retrieval.py      optional FAISS index over the policy corpus

Kept out of agents/ because these rules are a compliance artifact in their own
right: they get reviewed, versioned and argued over by people who are not
editing agent code, and an examiner may ask to see them in isolation.

Deliberately no re-exports here. Importing retrieval eagerly would drag in the
embeddings stack (and therefore torch) every time anything touched a rule, so
callers import the specific module they need:

    from policies.credit_rules import evaluate_rules
    from policies.retrieval import retrieve_policy_text
"""
