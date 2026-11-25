# Uncertainty-Aware Actionable Counterfactual Explanations

**Research extension:** `python run_research_extension.py` adds applicant-relative budgets, incomplete execution, joint committee assessment, independent exact-binomial certification with abstention, fresh-model audits, five ablations and three split seeds across two model families. Read [the research methodology](docs/research_extension.md) and `results/extension/summary.md`. This strengthens the implementation; novelty and publication acceptance are not established. All extension outputs are kept separately from the original experiment.

Complete implementation of the supplied three-page research proposal, Sections V–VI: leakage-free loan prediction, SHAP, constrained DiCE, bootstrap uncertainty, feasible perturbation stability, explicit reliability filtering and baseline evaluation. The paper reports no experimental results; all numbers in `results/` come from this implementation.

## Run

Python 3.11 was used. From this project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python run_experiment.py
python -m pytest -q
python -m streamlit run app.py
```

`requirements.txt` provides compatible ranges; the lock file records the actual executed direct dependency versions. Regeneration overwrites outputs. The default run evaluates up to 30 held-out model-rejected applicants, with 12 requested DiCE candidates each and 100 perturbations per candidate. Actual candidate counts can be lower. Increase budgets in `config.yaml` before a new experiment. Launch from the project directory.

## Data and leakage controls

`train.csv`: 614 rows, 13 columns, 422 Y / 192 N labels. `test.csv`: 367 rows, 12 columns, no target; its predictions are not performance estimates. Loan_ID is excluded as an identifier. There are 11 actual predictors: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area. No age feature exists.

Numeric predictors include binary Credit_History and discrete Loan_Amount_Term, both frozen for recourse; categorical predictors are encoded with unknown-category handling. Missing values are handled by training-only median/mode imputation, numerical scaling and one-hot encoding. Full missing counts, types, unique values, descriptive statistics and duplicates are saved in the data audit. Raw feature units are preserved for recourse; the dataset does not establish currency or income time units, so none are invented.

Stratified splits are approximately 60/20/20. Logistic Regression and Random Forest use fixed parameters; validation ROC-AUC selects the model, with F1 as tie-break. Both candidates' held-out results are reported, but do not influence selection. The selected model stays trained on the training split. The saved split indices establish provenance. Bootstrap members fit their own complete preprocessing pipelines on training resamples. No evaluation/test rows enter fitting, search reference ranges, SHAP background, threshold selection or candidate search.

Missing applicant values are imputed using training statistics before explaining recourse. An imputed value is a model input assumption, not a verified applicant fact; recommendations concerning missing financial values require verification first.

## Actionability assumptions — domain review required

Only ApplicantIncome (increase), CoapplicantIncome (increase), LoanAmount (decrease) can vary. All other columns are immutable, including credit history, education, household composition and property area. Thus categorical transitions are identity only. These conservative restrictions avoid recommending protected-attribute changes, relocation or instant credit-history repair. They do not establish ethical fairness of the predictive model, which still includes protected attributes.

Age, if present in a future labeled dataset, must remain unchanged during recourse: a 54-year-old cannot be advised to become 35. The current files contain no age column, so no age values are fabricated and no rejection is attributed to age. Training an age-aware model requires genuine age observations and retraining/evaluation. The framework freezes every feature outside the explicit mutable allowlist, which also protects an age column if one is introduced later.

Changes must lie within observed training minima/maxima. Directional ranges are intersected with the original applicant value. Costs are 1 for applicant income and loan amount, 1.5 for coapplicant income. These are demonstration assumptions, not empirically validated costs. Increasing income is not immediate or universally feasible; coapplicant participation requires consent. No joint affordability or causal feasibility model is available. Training ranges only establish marginal plausibility. Fractional currency amounts are allowed; historical loan terms and categorical values cannot change. No recommendation guarantees lender approval.

## Definitions

The desired class is Y, encoded 1; the classification threshold is 0.5. The saved model's probability is p(x). With B=15 independent training bootstrap fits, uncertainty is sample standard deviation:

`u(x) = sqrt(sum_b (p_b(x) - mean_b p_b(x))^2 / (B-1))`.

This measures training/model sensitivity, not total uncertainty, a confidence interval, or irreducible outcome noise. The model probability is distinct from this dispersion. Held-out Brier scores and a reliability diagram assess calibration; no post-hoc calibration is applied on evaluation data.

For each candidate, Gaussian noise is added to mutable numerical features with standard deviation 2% of the training 5–95 percentile span. Each noisy value is clipped to the intersection of training bounds and the original applicant's permitted direction. Immutable values stay fixed. `stability = count(model(perturbed_x)==1)/100`. Seeds are fixed per candidate; full perturbations are saved. This is local stability under a specified perturbation distribution, not robustness to future models or distribution shifts. Clipping can concentrate samples at bounds. Sampling error remains with 100 trials.

Proximity is the sum of cost-weighted absolute mutable-feature changes divided by training feature ranges. Sparsity counts changed original features. Reliability is a conjunction: valid desired prediction, actionability, `u<=0.15`, `stability>=0.90`, `proximity<=0.50`, `sparsity<=3`. These prespecified heuristic thresholds require sensitivity analysis and stakeholder validation before operational use. They were not tuned on evaluation or unlabeled test data. The three-feature sparsity cap is deliberately permissive under the conservative mutable set.

The 0.15 uncertainty cap excludes large bootstrap probability variation; the 0.90 stability floor requires at least 90 of 100 local trials to retain approval. The 0.50 proximity cap limits the summed weighted fraction of observed ranges to half a range, and the sparsity cap allows at most the three permitted actions. These interpretable demonstration budgets are not optimality or safety claims, and the range-based budget is sensitive to outliers.

## Fair comparison and interpretation

Constrained random DiCE is the baseline. The proposed method assesses the *same* candidate pool, rejects unreliable candidates and returns up to three ranked by uncertainty (then stability and proximity). Thus model, applicants, search budget and actionability constraints are identical. This isolates post-generation assessment; it is not an unconstrained DiCE comparison or a claim of a new search algorithm. No accepted output is reported explicitly. A finite unsuccessful random search does not prove recourse is impossible.

Baseline coverage is the fraction of evaluated model-rejected applicants with any valid actionable candidate; proposed coverage requires a candidate passing the entire reliability conjunction. Conditional quality averages are calculated per case, then across covered cases. Methods can cover different subsets. Candidate CSVs support paired analyses; quality improvements due to filtering do not establish superior search. Empty proposed quality metrics are undefined, not zeros. Evaluation is limited to a deterministic held-out rejected subset and one split/seed; no significance or generalization claim is warranted.

SHAP permutation explanations operate on the original 11-feature space, explain approval probability and use 40 training background rows / 40 evaluation rows. Category codes serve only as reversible masking labels. The applicant form also computes local transformed-feature SHAP contributions. Attribution is model behavior, not causal evidence. Correlated features, finite masking budgets and small samples limit interpretation.

## Deliverables

- `src/framework.py`: constraints, DiCE, uncertainty, perturbations, reliability.
- `run_experiment.py`, `config.yaml`: reproducible execution and explicit policy.
- `app.py`: dataset, performance, applicant form, local SHAP, recourse, comparison, downloads.
- `artifacts/`: selected pipeline, preprocessing and bootstrap ensemble bundle (trusted local joblib files only).
- `results/`: audits, splits, model metrics/reports, calibration, SHAP values, candidate-level results, full perturbations, per-case coverage, comparison, unlabeled predictions, run metadata and experimental summary.
- `plots/`: class distribution, confusion matrix, ROC, model comparison, calibration, global/summary/local SHAP, original/CF comparison, uncertainty, stability, method comparison and coverage.
- `tests/`: feature immutability, bounds/directions, perturbation feasibility/reproducibility, stability arithmetic, prediction validity, reliability and artifact consistency.
- `docs/paper_extracted.txt`: complete extracted paper text; `docs/verification.md`: execution evidence.

See `results/experimental_summary.md` for actual measured results. This research demonstration does not assess fairness, causal effects, temporal robustness, external validity or deployment suitability.
