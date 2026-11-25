# Experimental summary

Selected by validation ROC-AUC (F1 tie-break): RandomForest. No refitting on held-out data.

| model | split | accuracy | precision | recall | f1 | roc_auc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | validation | 0.7724 | 0.7692 | 0.9524 | 0.8511 | 0.6938 | 0.1685 |
| LogisticRegression | evaluation | 0.8618 | 0.8469 | 0.9765 | 0.9071 | 0.8486 | 0.1181 |
| RandomForest | validation | 0.7805 | 0.7714 | 0.9643 | 0.8571 | 0.7299 | 0.1727 |
| RandomForest | evaluation | 0.8455 | 0.83 | 0.9765 | 0.8973 | 0.808 | 0.1395 |

| method | evaluated_cases | returned_candidates | coverage | validity | actionability | proximity | sparsity | uncertainty | stability | reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiCE | 23 | 12 | 0.0435 | 1.0 | 1.0 | 0.6757 | 1.3333 | 0.1029 | 0.9983 | 0.3333 |
| Proposed | 23 | 3 | 0.0435 | 1.0 | 1.0 | 0.1736 | 1.3333 | 0.078 | 1.0 | 1.0 |

See README for formulas, assumptions, interpretation and limitations. Empty proposed metrics mean no accepted recommendations, not zero quality. Candidate metrics are averaged within case, then across cases with outputs; coverage uses all evaluated rejected cases.
