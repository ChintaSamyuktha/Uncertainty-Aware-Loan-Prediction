# Research extension: measured results

| family | budget | method | runs | evaluated_applicant_runs | accepted | coverage | coverage_sd | mean_cost | audit_joint_success | audit_transfer_success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.25 | Certified | 3 | 70 | 0 | 0.0 | 0.0 | undefined | undefined | undefined |
| LogisticRegression | 0.25 | Joint empirical | 3 | 70 | 0 | 0.0 | 0.0 | undefined | undefined | undefined |
| LogisticRegression | 0.25 | Nominal | 3 | 70 | 2 | 0.0267 | 0.0462 | 0.15 | 0.0 | 0.0 |
| LogisticRegression | 0.25 | Point stability | 3 | 70 | 1 | 0.0133 | 0.0231 | 0.05 | 0.0 | 0.0 |
| LogisticRegression | 0.25 | Uncertainty | 3 | 70 | 2 | 0.0267 | 0.0462 | 0.15 | 0.0 | 0.0 |
| LogisticRegression | 0.5 | Certified | 3 | 70 | 2 | 0.0267 | 0.0462 | 0.425 | 1.0 | 0.809 |
| LogisticRegression | 0.5 | Joint empirical | 3 | 70 | 2 | 0.0267 | 0.0462 | 0.425 | 1.0 | 0.809 |
| LogisticRegression | 0.5 | Nominal | 3 | 70 | 3 | 0.0433 | 0.0404 | 0.2125 | 0.0 | 0.157 |
| LogisticRegression | 0.5 | Point stability | 3 | 70 | 3 | 0.0433 | 0.0404 | 0.25 | 0.0 | 0.5 |
| LogisticRegression | 0.5 | Uncertainty | 3 | 70 | 3 | 0.0433 | 0.0404 | 0.2125 | 0.0 | 0.157 |
| LogisticRegression | 1.0 | Certified | 3 | 70 | 2 | 0.0267 | 0.0462 | 0.425 | 1.0 | 0.809 |
| LogisticRegression | 1.0 | Joint empirical | 3 | 70 | 2 | 0.0267 | 0.0462 | 0.425 | 1.0 | 0.809 |
| LogisticRegression | 1.0 | Nominal | 3 | 70 | 4 | 0.0567 | 0.0603 | 0.3238 | 0.0 | 0.157 |
| LogisticRegression | 1.0 | Point stability | 3 | 70 | 3 | 0.0433 | 0.0404 | 0.25 | 0.0 | 0.5 |
| LogisticRegression | 1.0 | Uncertainty | 3 | 70 | 3 | 0.0433 | 0.0404 | 0.2125 | 0.0 | 0.157 |
| RandomForest | 0.25 | Certified | 3 | 70 | 1 | 0.0119 | 0.0206 | 0.175 | 1.0 | 0.0 |
| RandomForest | 0.25 | Joint empirical | 3 | 70 | 1 | 0.0119 | 0.0206 | 0.175 | 1.0 | 0.0 |
| RandomForest | 0.25 | Nominal | 3 | 70 | 3 | 0.0357 | 0.0619 | 0.1917 | 0.178 | 0.0 |
| RandomForest | 0.25 | Point stability | 3 | 70 | 2 | 0.0238 | 0.0412 | 0.2 | 0.822 | 0.0 |
| RandomForest | 0.25 | Uncertainty | 3 | 70 | 3 | 0.0357 | 0.0619 | 0.1917 | 0.178 | 0.0 |
| RandomForest | 0.5 | Certified | 3 | 70 | 2 | 0.0238 | 0.0412 | 0.3 | 1.0 | 0.285 |
| RandomForest | 0.5 | Joint empirical | 3 | 70 | 2 | 0.0238 | 0.0412 | 0.3 | 1.0 | 0.285 |
| RandomForest | 0.5 | Nominal | 3 | 70 | 3 | 0.0357 | 0.0619 | 0.1917 | 0.178 | 0.0 |
| RandomForest | 0.5 | Point stability | 3 | 70 | 3 | 0.0357 | 0.0619 | 0.2333 | 0.548 | 0.0 |
| RandomForest | 0.5 | Uncertainty | 3 | 70 | 3 | 0.0357 | 0.0619 | 0.1917 | 0.178 | 0.0 |
| RandomForest | 1.0 | Certified | 3 | 70 | 2 | 0.0238 | 0.0412 | 0.3 | 1.0 | 0.285 |
| RandomForest | 1.0 | Joint empirical | 3 | 70 | 2 | 0.0238 | 0.0412 | 0.3 | 1.0 | 0.285 |
| RandomForest | 1.0 | Nominal | 3 | 70 | 3 | 0.0357 | 0.0619 | 0.1917 | 0.178 | 0.0 |
| RandomForest | 1.0 | Point stability | 3 | 70 | 3 | 0.0357 | 0.0619 | 0.2333 | 0.548 | 0.0 |
| RandomForest | 1.0 | Uncertainty | 3 | 70 | 3 | 0.0357 | 0.0619 | 0.1917 | 0.178 | 0.0 |

| seed | family | accuracy | roc_auc | brier | rejected |
| --- | --- | --- | --- | --- | --- |
| 42 | LogisticRegression | 0.8618 | 0.8486 | 0.1181 | 25 |
| 42 | RandomForest | 0.8537 | 0.8136 | 0.1392 | 22 |
| 17 | LogisticRegression | 0.8049 | 0.7517 | 0.1544 | 20 |
| 17 | RandomForest | 0.8049 | 0.7539 | 0.1622 | 20 |
| 83 | LogisticRegression | 0.8293 | 0.7415 | 0.1544 | 25 |
| 83 | RandomForest | 0.8049 | 0.739 | 0.1621 | 28 |

Results average run-level quantities across three split seeds. Conditional success/cost metrics exclude abstentions; coverage includes every evaluated rejected applicant. The same dataset and overlapping splits are not independent external replications. These data were used in earlier development, so this is exploratory, not a preregistered confirmation. Undefined cells mean no accepted outputs. See docs/research_extension.md for assumptions, statistical scope, related work and remaining publication requirements.
