"""Reproduce the complete experiment: python run_experiment.py."""
from pathlib import Path
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay, brier_score_loss
from sklearn.calibration import calibration_curve
from src.framework import complete, dice_generator, generate, assess

ROOT = Path(__file__).resolve().parent


def savefig(name):
    plt.tight_layout()
    plt.savefig(ROOT / 'plots' / f'{name}.png', dpi=160, bbox_inches='tight')
    plt.close('all')


def markdown_table(frame):
    """Small dependency-free Markdown table for the experimental report."""
    rows = frame.round(4).fillna('undefined').astype(str)
    return '| ' + ' | '.join(rows.columns) + ' |\n| ' + ' | '.join(['---']*len(rows.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows.to_numpy())


def main():
    for folder in ['artifacts', 'results', 'plots', 'docs']:
        (ROOT/folder).mkdir(exist_ok=True)
    config = yaml.safe_load((ROOT/'config.yaml').read_text())
    seed = config['seed']
    np.random.seed(seed)
    train, test = [pd.read_csv(ROOT/f'{name}.csv') for name in ['train', 'test']]
    if 'Loan_Status' not in train or set(train.Loan_Status) != {'Y', 'N'}:
        raise ValueError('Expected observed Y/N Loan_Status labels')
    features = train.drop(columns=['Loan_ID', 'Loan_Status'])
    if set(test.columns) - {'Loan_ID', 'Loan_Status'} != set(features.columns):
        raise ValueError('Test schema differs from training schema')
    audit = {name: dict(rows=len(df), columns=list(df), dtypes=df.dtypes.astype(str).to_dict(), missing=df.isna().sum().to_dict(), duplicates=int(df.duplicated().sum()), unique=df.nunique().to_dict(), classes=df.Loan_Status.value_counts().to_dict() if 'Loan_Status' in df else None) for name, df in [('train',train),('test',test)]}
    (ROOT/'results/data_audit.json').write_text(json.dumps(audit, indent=2))
    train.describe(include='all').to_csv(ROOT/'results/data_descriptive.csv')
    y = train.Loan_Status.map({'N':0, 'Y':1})
    fit_idx, held_idx = train_test_split(train.index, test_size=.2, stratify=y, random_state=seed)
    fit_idx, val_idx = train_test_split(fit_idx, test_size=.25, stratify=y.loc[fit_idx], random_state=seed)
    pd.DataFrame({'row':np.r_[fit_idx,val_idx,held_idx], 'split':['train']*len(fit_idx)+['validation']*len(val_idx)+['evaluation']*len(held_idx)}).to_csv(ROOT/'results/splits.csv',index=False)
    numeric = list(features.select_dtypes(include='number').columns)
    categorical = [c for c in features if c not in numeric]
    prep = ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
    estimators = {'LogisticRegression':LogisticRegression(max_iter=2000,random_state=seed), 'RandomForest':RandomForestClassifier(n_estimators=200,min_samples_leaf=3,random_state=seed,n_jobs=-1)}
    models, rows = {}, []
    for name, estimator in estimators.items():
        model = Pipeline([('preprocess',clone(prep)),('classifier',estimator)])
        model.fit(features.loc[fit_idx], y.loc[fit_idx]); models[name] = model
        for split, idx in [('validation',val_idx),('evaluation',held_idx)]:
            pred = model.predict(features.loc[idx]); prob = model.predict_proba(features.loc[idx])[:,1]
            rows.append(dict(model=name,split=split,accuracy=accuracy_score(y.loc[idx],pred),precision=precision_score(y.loc[idx],pred,zero_division=0),recall=recall_score(y.loc[idx],pred,zero_division=0),f1=f1_score(y.loc[idx],pred),roc_auc=roc_auc_score(y.loc[idx],prob),brier=brier_score_loss(y.loc[idx],prob)))
            (ROOT/f'results/{name}_{split}_report.json').write_text(json.dumps(classification_report(y.loc[idx],pred,output_dict=True),indent=2))
    metrics = pd.DataFrame(rows); metrics.to_csv(ROOT/'results/model_metrics.csv',index=False)
    chosen = metrics.query("split == 'validation'").sort_values(['roc_auc','f1'],ascending=False).iloc[0].model
    model = models[chosen]
    reference = complete(features.loc[fit_idx],features.loc[fit_idx])
    ensemble = []
    for b in range(config['ensemble_size']):
        idx = np.random.default_rng(seed+b).choice(fit_idx,len(fit_idx),replace=True)
        member = clone(model); member.set_params(classifier__random_state=seed+b+1)
        member.fit(features.loc[idx],y.loc[idx]); ensemble.append(member)
    bundle = dict(model=model,ensemble=ensemble,reference=reference,config=config,selected=chosen)
    joblib.dump(bundle,ROOT/'artifacts/research.joblib')
    joblib.dump(model,ROOT/'artifacts/model.joblib'); joblib.dump(model['preprocess'],ROOT/'artifacts/preprocessing.joblib')
    pd.DataFrame({'Loan_ID':test.Loan_ID,'approval_probability':model.predict_proba(test[features.columns])[:,1],'prediction':np.where(model.predict(test[features.columns])==1,'Y','N')}).to_csv(ROOT/'results/test_predictions.csv',index=False)
    train.Loan_Status.value_counts().plot.bar(color=['#14b8a6','#f97316'],title='Observed training labels'); savefig('class_distribution')
    metrics.query("split == 'evaluation'").set_index('model')[['accuracy','f1','roc_auc']].plot.bar(ylim=(0,1)); savefig('model_performance')
    ConfusionMatrixDisplay.from_estimator(model,features.loc[held_idx],y.loc[held_idx]); savefig('confusion_matrix')
    RocCurveDisplay.from_estimator(model,features.loc[held_idx],y.loc[held_idx]); savefig('roc_curve')
    prob = model.predict_proba(features.loc[held_idx])[:,1]
    obs, pred = calibration_curve(y.loc[held_idx],prob,n_bins=6)
    plt.plot(pred,obs,'o-',label='Selected model'); plt.plot([0,1],[0,1],'--'); plt.xlabel('Mean predicted probability'); plt.ylabel('Observed approval fraction'); savefig('calibration')
    pd.DataFrame({'mean_probability':pred,'observed_fraction':obs}).to_csv(ROOT/'results/calibration.csv',index=False)
    print('Selected:',chosen,flush=True)
    explain(model,reference,complete(features.loc[held_idx],reference))
    generator = dice_generator(reference,y.loc[fit_idx],model)
    rejected = held_idx[model.predict(features.loc[held_idx])==0][:config['evaluation_cases']]
    records, cases, perturbations = [], [], []
    for number, idx in enumerate(rejected):
        print(f'Counterfactual case {number+1}/{len(rejected)}',flush=True)
        original = complete(features.loc[[idx]],reference).iloc[0]
        candidates,error = generate(generator,original,reference,config,seed+number)
        assessed = []
        for rank, (_,candidate) in enumerate(candidates.iterrows()):
            stats, nearby = assess(original,candidate,model,ensemble,reference,config,seed+1000+number*100+rank)
            record = dict(case_id=int(idx),candidate_id=rank,**stats,**{f'original_{c}':original[c] for c in features},**{f'cf_{c}':candidate[c] for c in features})
            assessed.append(record)
            nearby['case_id']=idx; nearby['candidate_id']=rank; perturbations.append(nearby)
        # Identical candidate pool isolates the effect of uncertainty-aware filtering.
        acceptable = [r for r in assessed if r['reliable']]
        acceptable.sort(key=lambda r:(r['uncertainty'], -r['stability'],r['proximity']))
        selected_ids = {r['candidate_id'] for r in acceptable[:3]}
        for r in assessed:
            records.append(dict(method='DiCE',**r))
            if r['candidate_id'] in selected_ids:
                records.append(dict(method='Proposed',**r))
        cases.append(dict(case_id=int(idx),candidates=len(assessed),baseline_covered=any(r['validity'] and r['actionability'] for r in assessed),proposed_covered=bool(acceptable),error=error))
    cf = pd.DataFrame(records)
    columns=['method','case_id','candidate_id','probability','uncertainty','proximity','sparsity','validity','actionability','stability','reliable']
    if cf.empty: cf=pd.DataFrame(columns=columns)
    cf.to_csv(ROOT/'results/counterfactuals.csv',index=False)
    case_frame=pd.DataFrame(cases,columns=['case_id','candidates','baseline_covered','proposed_covered','error']); case_frame.to_csv(ROOT/'results/cases.csv',index=False)
    if perturbations: pd.concat(perturbations).to_csv(ROOT/'results/perturbations.csv',index=False)
    summaries=[]
    for method,key in [('DiCE','baseline_covered'),('Proposed','proposed_covered')]:
        sub=cf[cf.method==method]
        means=sub.groupby('case_id')[['validity','actionability','proximity','sparsity','uncertainty','stability','reliable']].mean().mean() if len(sub) else pd.Series(dtype=float)
        summaries.append(dict(method=method,evaluated_cases=len(cases),returned_candidates=len(sub),coverage=float(case_frame[key].mean()) if len(cases) else 0,**means.to_dict()))
    summary=pd.DataFrame(summaries); summary.to_csv(ROOT/'results/baseline_comparison.csv',index=False)
    for fields,name in [(['uncertainty'],'uncertainty_comparison'),(['stability'],'stability_comparison'),(['validity','actionability','stability','coverage'],'baseline_comparison'),(['reliable','coverage'],'reliability_coverage')]:
        summary.set_index('method').reindex(columns=fields).plot.bar(title=name.replace('_',' ').title()); savefig(name)
    if len(cf):
        r=cf.iloc[0]; cols=list(config['mutable'])
        pd.DataFrame({'Original':[r[f'original_{c}'] for c in cols],'Counterfactual':[r[f'cf_{c}'] for c in cols]},index=cols).plot.bar(logy=True); savefig('original_vs_counterfactual')
    versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','scikit-learn','shap','dice-ml','streamlit','protobuf','matplotlib','PyYAML','joblib','pytest','pypdf']}
    (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    (ROOT/'results/run_metadata.json').write_text(json.dumps(dict(selected_model=chosen,python=platform.python_version(),versions=versions,split_sizes=[len(fit_idx),len(val_idx),len(held_idx)],config=config),indent=2))
    report = f'# Experimental summary\n\nSelected by validation ROC-AUC (F1 tie-break): {chosen}. No refitting on held-out data.\n\n'+markdown_table(metrics)+'\n\n'+markdown_table(summary)+'\n\nSee README for formulas, assumptions, interpretation and limitations. Empty proposed metrics mean no accepted recommendations, not zero quality. Candidate metrics are averaged within case, then across cases with outputs; coverage uses all evaluated rejected cases.\n'
    (ROOT/'results/experimental_summary.md').write_text(report,encoding='utf-8')
    print(report,flush=True)


def explain(model, reference, evaluation):
    import shap
    # Explain probability in original feature space. Encode only for SHAP masking;
    # decode back before invoking the actual saved sklearn pipeline.
    categories = {c:list(reference[c].unique()) for c in reference.select_dtypes(exclude='number')}
    def encode(frame):
        out=frame.copy()
        for c,values in categories.items(): out[c]=out[c].map({v:i for i,v in enumerate(values)})
        return out.astype(float)
    def predict(array):
        frame=pd.DataFrame(array,columns=reference.columns)
        for c,values in categories.items(): frame[c]=[values[int(np.clip(round(v),0,len(values)-1))] for v in frame[c]]
        return model.predict_proba(frame)[:,1]
    background=encode(reference.sample(min(40,len(reference)),random_state=42))
    sample=encode(evaluation.iloc[:40])
    explainer=shap.Explainer(predict,background,algorithm='permutation',seed=42)
    values=explainer(sample,max_evals=2*len(reference.columns)+1)
    np.savez(ROOT/'results/shap_values.npz',values=values.values,base_values=values.base_values,data=sample.to_numpy(),features=reference.columns.to_numpy(dtype=str))
    importance=pd.Series(abs(values.values).mean(axis=0),index=reference.columns).sort_values(ascending=False)
    importance.to_csv(ROOT/'results/shap_importance.csv',header=['mean_absolute_shap'])
    shap.plots.bar(values,show=False); savefig('shap_importance')
    shap.plots.beeswarm(values,show=False); savefig('shap_summary')
    shap.plots.waterfall(values[0],show=False); savefig('shap_individual')


if __name__=='__main__': main()
