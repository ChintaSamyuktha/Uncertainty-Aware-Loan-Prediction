# Execution verification

Verified in the supplied Windows workspace using Python 3.11.

- Read the full research-paper text and inspected rendered pages 1–3.
- Executed dataset audit, training-only preprocessing, both classifiers, validation model selection, 15 bootstrap pipelines, held-out model evaluation, Brier/calibration assessment, SHAP permutation explanation, random DiCE candidate generation, actionability checks, uncertainty estimates, 100 perturbations per candidate, reliability filtering and common-pool baseline comparison.
- The first experiment reached report generation, then failed because pandas Markdown rendering required an absent optional `tabulate` package. Replaced it with a dependency-free Markdown formatter and reran the complete experiment successfully (exit code 0). See `results/execution.log`.
- Saved three joblib artifacts, 13 research PNGs, all CSV/JSON/NPZ results and the experimental summary. Test prediction CSV contains all 367 unlabeled applicants.
- `python -m pytest -q`: **7 passed**, one third-party Streamlit/Altair deprecation warning. Tests cover constraints, feasible/reproducible perturbations, exact stability arithmetic, validity/reliability, saved-artifact consistency, all seven dashboard pages and an interactive applicant prediction with local SHAP and a rejected-applicant recourse flow.
- Started Streamlit outside the socket-restricted sandbox on `http://127.0.0.1:8502`; its `/_stcore/health` endpoint returned `ok`. Initial socket checks inside the sandbox could not connect. This was an environment restriction, resolved for verification with authorized external execution.
- Streamlit was pinned to 1.41.1 and protobuf restored to the pre-existing 4.25.3 version after the newest Streamlit dependency installation conflicted with unrelated installed Google/MediaPipe packages. The actual direct dependency versions are recorded in `requirements-lock.txt`; use a virtual environment for reproduction.
- Inspected SHAP summary and baseline comparison plot images: labels and layouts are readable. Dashboard interactions were tested through Streamlit AppTest; a full browser visual review was not performed.

No experimental step remains unexecuted. Scientific limitations remain: one split/seed, one covered rejected applicant, marginal feasibility assumptions, heuristic thresholds, no causal/fairness or external validation, and no formal uncertainty intervals. See `docs/final_report.md` and README.

## Subsequent research extension

The preceding section describes the original experiment. The separately executed extension adds three split seeds, two fixed model families, action budgets, incomplete execution, fixed-committee joint success, independent exact-binomial certificates, abstention and fresh-model audits. Its narrow conditional certificate is not an interval for real-world approval probability. Original outputs remain unchanged. Full methodology, statistical justification and source comparisons are in `docs/research_extension.md`; pilot findings are in `docs/extension_findings.md`. The dashboard now includes a Research extension page.

The extension ran to completion twice with consistent reported results: six model/split runs, 140 rejected applicant-runs, 2,100 budget/method decisions, four independent certificate records, six saved model bundles and six new research plots. Metadata records source/input hashes and runtime package versions. The final complete test suite reports **14 passed**, with one third-party Altair deprecation warning (`results/extension_tests.txt`). Coverage and transfer plot images were inspected for readability. The dashboard page, including the extension viewer, passes AppTest; a browser visual review of the new page was not performed. No assertion of scientific novelty or readiness for publication follows from successful execution.

