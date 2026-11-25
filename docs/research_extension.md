# Research extension: budgeted recourse with independent certification

Working title: **Selection-Valid Certification of Budgeted Recourse under Incomplete Action Execution**.

This is an implemented research direction, not an established novelty claim or a publication guarantee. The contribution to this project is a fully specified selection/certification/audit protocol and its controlled ablations. The confidence bound and multiplicity correction are standard statistical tools. Related work already covers uncertain, robust and certified recourse; a claim of being the first would be unsupported.

## Research question

When applicants have limited action budgets and may achieve only part of a recommended change, does independent confidence-bound certification provide useful assurance beyond probability dispersion and point stability? What does it cost in coverage, action cost and transfer to models not used in selection?

## What changes from the original framework

The original framework filters a small DiCE pool using training-range distance and point estimates of uncertainty/stability. The extension searches bounded applicant-relative actions, tests the joint success of a fixed model committee under incomplete execution, freezes the recommendation before certification, corrects for the reported budget choices, and abstains if certification fails. Screening trials never serve as certification trials. A further independent simulation audit measures actual simulated success, including transfer to fresh bootstrap models.

The extension is a separate experiment; it does not replace or retrospectively alter the original DiCE results. Its comparison methods share the same search pool, shortlisting, predictor, budgets, committee and applicant cases. Its search algorithm is Sobol sampling plus single-feature grids, not a newly invented optimization method. It gives no global optimality or search-completeness guarantee.

## Budget and execution model

For mutable feature j, let `s_j=max(abs(x_j), IQR_train(j), 1)`. Define cost `C(x,z)=sum_j w_j*abs(z_j-x_j)/s_j`. Candidate actions change each feature by at most `0.5*s_j`, intersected with training bounds and the existing allowed direction. Budgets are 0.25, 0.50 and 1.00. This replaces sensitivity to training minima/maxima in the cost with applicant values and a training-IQR floor, particularly for zero coapplicant income. These are dimensionless hypothetical resource budgets, not monetary or validated effort estimates.

Each achieved feature is `X'_j = x_j + U_j*(z_j-x_j)`, with independent `U_j ~ Uniform(0.8,1.0)`. Unchanged fields stay exactly unchanged. Thus a person may achieve only 80–100% of the requested financial change, and age/other immutable fields remain fixed. The execution distribution is a declared modeling assumption, not learned from human behavior. Correlated execution, complete failure to act, background income shocks and temporal changes are excluded. Real recommendations need external feasibility validation. No synthetic age values are added to the dataset.

## Selection and certificate

The committee contains the anchor predictor and five complete training-bootstrap pipelines of the same model family. It is fixed before candidate screening. For candidate z and execution randomness U define the Bernoulli outcome

`Y(z,U) = 1 if every fixed committee member predicts approval for execute(x,z,U); otherwise 0`.

We target `q(z)=P_U(Y(z,U)=1) >= 0.90`, conditional on this trained committee, applicant and simulator. This is a joint model/implementation success probability. It does not quantify success under a true lending process or all possible future models.

Candidate search uses 256 scrambled Sobol action fractions and 20 nonzero steps on each of three single-feature axes, plus the original point. For each budget, retain up to four cheapest nominal-valid candidates and four with highest minimum committee probability. The union of these shortlists is identical for all ablation methods. This is a finite computational heuristic; narrowing the shortlist can miss reliable candidates.

Using 100 screening trials, freeze one cheapest eligible candidate per budget for each method:

1. Nominal: approval by the anchor, subject to actionability/budget.
2. Uncertainty: additionally, probability standard deviation at most 0.15.
3. Point stability: anchor success in at least 90% of screening trials.
4. Joint empirical: all committee members approve in at least 90% of screening trials.
5. Certified: preselect as in Joint empirical, then apply the independent test below.

Let K be the number of distinct preselected Certified candidates across budgets (at most three). For each, draw 400 fresh IID execution trials independent of all screening trials. If k trials succeed, compute the exact one-sided Clopper–Pearson lower bound

`L=0` for `k=0`; otherwise `L=BetaQuantile(alpha/K; k, n-k+1)`, with `alpha=0.05`.

Return the preselected candidate only if `L>=0.90`. If it fails, abstain; do not inspect another candidate using the same certification samples. Repeating searches until a certificate passes would require a new error-control design.

### Statistical justification and scope

Condition on training data, fixed models, candidate generation and screening outcomes. This fixes K and the selected candidates independently of the certification draws. For each candidate, the exact binomial construction satisfies `P(L_i>q_i) <= alpha/K`. By the union bound, `P(exists i: L_i>q_i) <= alpha`. Therefore, with probability at least 0.95 over certification simulation, every reported certified budget recommendation for this applicant/run has simulator joint-success probability at least 0.90. Candidate samples can share model predictions; no independence among committee members is needed because each execution trial contributes a single joint Bernoulli outcome.

This is simultaneous coverage across preselected budgets **for one applicant/run**. It is not a family-wise guarantee across all applicants, repeated runs, future dashboard queries or user-selected simulator settings. It is not a conformal label guarantee, a posterior probability that the applicant will be approved, an assurance of fair lending, a claim of robustness to arbitrary model changes, or a new statistical theorem. All these distinctions are important in a paper.

## Independent evaluation

Three stratified 60/20/20 split seeds (42, 17, 83) and two fixed model families (Logistic Regression, Random Forest) produce six runs. For comparability, the 20% validation split is reserved but unused; no hyperparameters or thresholds are tuned in this extension. Each model and preprocessing pipeline uses training rows only. All held-out model-rejected applicants are assessed, with no outcome-based case selection. Models can reject different subsets; compare methods within a run/family, not as if cross-family applicant sets were paired.

Two further bootstrap models per run are excluded from screening/certification. Once every decision is frozen, 500 fresh execution samples audit each returned candidate on both the fixed committee and these additional models. Transfer metrics are empirical checks, not covered by the certificate. The same candidate receives the same audit draws across methods. No audit result changes the recommendation.

Results retain every applicant/budget/method decision including abstentions and reasons, all shortlisted candidate values, certificates and split indices. Coverage includes every assessed rejection. Quality and costs condition on accepted outputs, so methods can have different covered subsets. Summary values average run-level quantities across three seeds, with descriptive coverage standard deviations; overlapping splits are not independent external datasets. No significance test or causal improvement claim is made.

**Exploratory status:** the same underlying data were examined in earlier development. Fixed extension settings prevent further evaluation-based tuning, but do not make this a preregistered confirmatory experiment. Independent datasets/holdouts are still needed.

## Closest work and responsible positioning

| Source | Established contribution / overlap | Distinction to investigate, not a claim of precedence |
|---|---|---|
| [Mothilal et al., DiCE, 2020](https://www.microsoft.com/en-us/research/publication/explaining-machine-learning-classifiers-through-diverse-counterfactual-examples/?lang=ja) | Diverse feasible actionable counterfactuals | Independent selection/certification/audit of budget recommendations |
| [Schut et al., AISTATS 2021](https://proceedings.mlr.press/v130/schut21a.html) | Predictive uncertainty for generating/interpreting counterfactuals | Joint committee-execution Bernoulli target, distinct from probability dispersion |
| [Jiang et al., IJCAI 2024 survey](https://ijcai.org/proceedings/2024/894) | Taxonomy of model and execution robustness | Explicit scope and empirical trade-offs under a declared execution model |
| [Marzari et al., Artificial Intelligence, 2025](https://doi.org/10.1016/j.artint.2025.104459) | Probabilistic certification under model changes | Finite fixed committee plus action-execution noise and budget selection; direct comparison remains necessary |
| [Xuan et al., ROSE, Machine Learning 2025](https://link.springer.com/article/10.1007/s10994-025-06821-1) | Sequential recourse under noisy execution | This extension is one-shot, not sequential or causal; ROSE is a stronger related method |
| [Bilkhoo et al., CONFEX, 2025 preprint](https://arxiv.org/abs/2510.19754) | Conformal and optimization guarantees for uncertainty-aware recourse | Monte Carlo simulator success is a different target and weaker scope |
| [Johansson et al., COPA 2026](https://proceedings.mlr.press/v329/johansson26a.html) | Conformity-based selection in feature/SHAP space | Independent action-execution certification rather than conformity selection |
| [Clopper and Pearson, 1934](https://doi.org/10.1093/biomet/26.4.404) | Exact binomial confidence limits | Used as an existing statistical component, not claimed as original |

The targeted search confirms substantial overlap. A defensible initial framing is **a reproducible empirical protocol examining the price of independently certified recourse**, not a first uncertainty-aware counterfactual system. Publication readiness still requires full-text comparisons to the nearest methods, additional datasets, relevant robust-recourse baselines, execution-distribution sensitivity, more split seeds, validated feature costs and careful fairness analysis. A dashboard is an engineering deliverable, not itself evidence of novelty.

## Reproduce

```powershell
python run_research_extension.py
python -m pytest -q
python -m streamlit run app.py
```

Outputs: `results/extension/`, `plots/extension/`, `artifacts/extension/`. Settings: `research_config.yaml`. The original experiment remains in its original locations. SciPy is used for exact beta quantiles and Sobol sampling; versions are listed in the dependency files. See `results/extension/summary.md` for actual results, not assumed improvements.
