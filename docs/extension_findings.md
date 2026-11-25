# What the extension actually established

Six completed experiments use seeds 42, 17 and 83 and both Logistic Regression and Random Forest. Across these experiments, 140 model-rejected applicant-runs were evaluated: 70 for each model family. These are not 140 independent people; applicants and training rows can recur across overlapping splits. Each applicant-run has three budgets and five method decisions.

Four distinct applicant-run recommendations passed an independent exact-binomial certificate, all in seed 83. Two concern Logistic Regression and two Random Forest. Some recommendations are returned at multiple budgets, so counting every budget decision would overstate the number of distinct recommendations. All four achieved 400/400 joint successes in certification, giving a one-sided lower bound of approximately 0.99254 for their simulator joint-success probability. Each applicant had only one distinct preselected certificate candidate, so its alpha allocation was 0.05. The multiplicity correction remains implemented for cases with several distinct budget recommendations.

All certified outputs achieved 100% joint success in the separate 500-trial audit against their fixed six-model committees. Yet their empirical joint success against two additional bootstrap models was 80.9% on average for Logistic Regression and 28.5% for Random Forest at budgets 0.5 and 1.0. This transfer audit lies outside the certificate's stated scope. The result demonstrates why a finite-committee certificate must not be marketed as protection against arbitrary future retraining.

At budget 0.5, mean split-level recommendation coverage was 2.67% for certified Logistic Regression and 2.38% for certified Random Forest. Nominal coverage was 4.33% and 3.57%, respectively. Certified accepted-output mean action costs were 0.425 and 0.300 versus nominal 0.2125 and 0.1917. These conditional comparisons involve different output subsets, very small counts and a finite search pool; they cannot demonstrate population-level superiority.

Joint empirical and Certified made identical decisions in this experiment. Thus the data show no incremental empirical accuracy or coverage advantage of the independent certificate over joint screening. The certificate adds an explicit conditional confidence statement; its practical value and trade-off require larger and more challenging experiments. The probability-standard-deviation ablation mostly matched nominal selection, showing that a low probability dispersion threshold did not ensure joint committee robustness here.

## A defensible paper argument

The strongest observed result is a caution about **what is certified**: recommendations may satisfy a narrow simulator/committee confidence statement and still fail under fresh trained models. A possible empirical paper could systematically measure this gap across budgets, committee compositions, action-execution distributions and multiple external datasets. The current repository supplies a working protocol and pilot findings for that study.

Do not write that this project invented immutable features, robust recourse, uncertainty-aware counterfactuals, exact confidence bounds, conformal prediction or abstention. Do not claim a novel first method, formal protection against all future models, statistical significance, or publication readiness. The proposed title in the methodology is a working title pending a full novelty review.

## Work still needed before submission

1. Full-text comparison and execution of suitable published robust-recourse baselines. The current five methods are controlled internal ablations, not a state-of-the-art benchmark.
2. Additional public datasets with clearly documented licenses, labels, immutable features and domain-specific recourse policies; genuinely observed age is required to study age-related behavior.
3. More seeds and independent holdouts, plus execution-noise, budget and committee-size sensitivity analyses. No further tuning should be presented as independent confirmation on these already-seen data.
4. Fairness, joint feasibility and human action-cost validation. The current model still uses protected attributes and the action simulator is hypothetical.
5. Broader statistical validation of the certificate protocol and clear separation of simulation error, model uncertainty, outcome uncertainty and transfer error.

See `results/extension/summary.md`, the per-decision/certificate CSVs and `docs/research_extension.md` for definitions and source references. Findings are exploratory pilot results and should be presented that way.
