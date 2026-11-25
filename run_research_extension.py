"""Repeated-split research extension; does not overwrite the original experiment."""
from pathlib import Path
import json
import argparse
import time
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.framework import complete
from src.budgeted_recourse import select_and_certify, simulate
from run_experiment import markdown_table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/extension'


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--settings',default='research_config.yaml')
    args=parser.parse_args()
    settings=yaml.safe_load((ROOT/args.settings).read_text())
    policy=yaml.safe_load((ROOT/'config.yaml').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    (ROOT/'plots/extension').mkdir(parents=True,exist_ok=True)
    (ROOT/'artifacts/extension').mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(ROOT/'train.csv'); x=raw.drop(columns=['Loan_ID','Loan_Status']); y=raw.Loan_Status.map({'N':0,'Y':1})
    preprocessing=joblib.load(ROOT/'artifacts/research.joblib')['model']
    model_metrics=[]; decisions=[]; candidate_rows=[]; certificate_rows=[]; split_rows=[]
    started=time.time()
    for seed in settings['seeds']:
        train_idx,test_idx=train_test_split(raw.index,test_size=.2,stratify=y,random_state=seed)
        train_idx,val_idx=train_test_split(train_idx,test_size=.25,stratify=y.loc[train_idx],random_state=seed)
        for split,indices in [('train',train_idx),('validation_unused',val_idx),('evaluation',test_idx)]:
            split_rows.extend(dict(seed=seed,row=int(i),split=split) for i in indices)
        reference=complete(x.loc[train_idx],x.loc[train_idx])
        for family in settings['model_families']:
            print(f'Run seed={seed}, family={family}',flush=True)
            model=clone(preprocessing)
            classifier=LogisticRegression(max_iter=2000,random_state=seed) if family=='LogisticRegression' else RandomForestClassifier(n_estimators=settings['trees'],min_samples_leaf=3,n_jobs=-1,random_state=seed)
            model.set_params(classifier=classifier); model.fit(x.loc[train_idx],y.loc[train_idx])
            members=[]
            for b in range(settings['bootstrap_members']+settings['audit_members']):
                member_seed=seed+100+b
                indices=np.random.default_rng(member_seed).choice(train_idx,len(train_idx),replace=True)
                member=clone(model); member.set_params(classifier__random_state=member_seed)
                member.fit(x.loc[indices],y.loc[indices]); members.append(member)
            committee=[model]+members[:settings['bootstrap_members']]
            audit_models=members[settings['bootstrap_members']:]
            pred=model.predict(x.loc[test_idx]); probability=model.predict_proba(x.loc[test_idx])[:,1]
            model_metrics.append(dict(seed=seed,family=family,accuracy=accuracy_score(y.loc[test_idx],pred),roc_auc=roc_auc_score(y.loc[test_idx],probability),brier=brier_score_loss(y.loc[test_idx],probability),rejected=int((pred==0).sum())))
            bundle=dict(model=model,committee=committee,audit_models=audit_models,reference=reference,policy=policy,settings=settings,seed=seed,family=family)
            joblib.dump(bundle,ROOT/f'artifacts/extension/{family}_{seed}.joblib')
            for number,idx in enumerate(test_idx[pred==0]):
                print(f'  rejected applicant {number+1}/{sum(pred==0)}',flush=True)
                original=complete(x.loc[[idx]],reference).iloc[0]
                run_seed=seed*100000+int(idx)*10
                candidates,screen,choices,cert=select_and_certify(original,model,committee,reference,policy,settings,run_seed)
                for i,row in screen.iterrows():
                    candidate_rows.append(dict(seed=seed,family=family,case_id=int(idx),**row.to_dict(),**{f'original_{c}':original[c] for c in reference},**{f'cf_{c}':candidates.iloc[int(row.candidate_id)][c] for c in reference}))
                for _,row in cert.iterrows(): certificate_rows.append(dict(seed=seed,family=family,case_id=int(idx),**row.to_dict()))
                selected=sorted({v for v in choices.values() if v is not None})
                # Evaluation randomness is independent of screening and certification.
                audit={}
                if selected:
                    deployed=simulate(original,candidates.iloc[selected],committee,policy,settings,settings['audit_trials'],run_seed+1000000)
                    transfer=simulate(original,candidates.iloc[selected],audit_models,policy,settings,settings['audit_trials'],run_seed+1000000)
                    for i,first,second in zip(selected,deployed,transfer):
                        audit[i]=dict(audit_anchor_success=first['anchor_success'],audit_joint_success=first['joint_success'],audit_transfer_joint_success=second['joint_success'])
                for (budget,method),candidate_id in choices.items():
                    record=dict(seed=seed,family=family,case_id=int(idx),budget=budget,method=method,accepted=candidate_id is not None,candidate_id=candidate_id,reason='No nominal candidate within budget')
                    if candidate_id is not None:
                        row=screen.set_index('candidate_id').loc[candidate_id]
                        record.update(reason='Accepted',cost=float(row.cost),probability=float(row.probability),uncertainty=float(row.probability_std),sparsity=int(row.sparsity),screen_joint_success=float(row.joint_success),**audit[candidate_id])
                    elif len(screen[screen.cost<=budget]):
                        record['reason']='Reliability condition failed'
                        if method=='Certified' and len(cert): record['reason']='No independent certificate passed (no retry)'
                    decisions.append(record)
            # Checkpoint all completed runs; interrupted work remains inspectable.
            for name,rows in [('decisions',decisions),('candidates',candidate_rows),('certificates',certificate_rows),('model_metrics',model_metrics),('splits',split_rows)]:
                pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    frame=pd.DataFrame(decisions)
    summary=summarize(frame)
    summary.to_csv(OUT/'summary.csv',index=False)
    paired=[]
    for (family,budget),sub in frame.groupby(['family','budget']):
        pivot=sub.pivot(index=['seed','case_id'],columns='method',values='accepted')
        paired.append(dict(family=family,budget=budget,evaluated=len(pivot),nominal_only=int((pivot.Nominal & ~pivot.Certified).sum()),both=int((pivot.Nominal & pivot.Certified).sum()),certified_only=int((~pivot.Nominal & pivot.Certified).sum()),neither=int((~pivot.Nominal & ~pivot.Certified).sum())))
    pd.DataFrame(paired).to_csv(OUT/'paired_coverage.csv',index=False)
    metadata=dict(settings=settings,policy=policy,elapsed_seconds=time.time()-started,scope='Conditional Monte Carlo certificates per applicant/run across reported budgets; not lender outcome guarantees',original_evaluation_already_seen=True,external_validation=False,
        file_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['train.csv','research_config.yaml','config.yaml','src/budgeted_recourse.py','run_research_extension.py']},
        versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn','joblib']})
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    plot(summary)
    text='# Research extension: measured results\n\n'+markdown_table(summary)+'\n\n'+markdown_table(pd.DataFrame(model_metrics))+'\n\nResults average run-level quantities across three split seeds. Conditional success/cost metrics exclude abstentions; coverage includes every evaluated rejected applicant. The same dataset and overlapping splits are not independent external replications. These data were used in earlier development, so this is exploratory, not a preregistered confirmation. Undefined cells mean no accepted outputs. See docs/research_extension.md for assumptions, statistical scope, related work and remaining publication requirements.\n'
    (OUT/'summary.md').write_text(text,encoding='utf-8')
    print(text,flush=True)


def summarize(frame):
    rows=[]
    for (family,budget,method),sub in frame.groupby(['family','budget','method']):
        runs=[]
        for seed,part in sub.groupby('seed'):
            accepted=part[part.accepted]
            runs.append(dict(seed=seed,coverage=float(part.accepted.mean()),cost=float(accepted.cost.mean()) if len(accepted) else np.nan,joint_success=float(accepted.audit_joint_success.mean()) if len(accepted) else np.nan,transfer_success=float(accepted.audit_transfer_joint_success.mean()) if len(accepted) else np.nan))
        runs=pd.DataFrame(runs)
        rows.append(dict(family=family,budget=budget,method=method,runs=len(runs),evaluated_applicant_runs=len(sub),accepted=int(sub.accepted.sum()),coverage=float(runs.coverage.mean()),coverage_sd=float(runs.coverage.std(ddof=1)),mean_cost=float(runs.cost.mean()),audit_joint_success=float(runs.joint_success.mean()),audit_transfer_success=float(runs.transfer_success.mean())))
    return pd.DataFrame(rows)


def plot(summary):
    for family,sub in summary.groupby('family'):
        for metric,label in [('coverage','Recommendation coverage'),('audit_joint_success','Independent audit success (fixed committee)'),('audit_transfer_success','Independent audit success (unseen bootstrap models)')]:
            fig,ax=plt.subplots(figsize=(8,5))
            for method,part in sub.groupby('method'):
                ax.plot(part.budget,part[metric],marker='o',label=method)
            upper=max(.05,1.3*float(sub[metric].max())) if metric=='coverage' else 1.02
            ax.set(xlabel='Applicant-relative action budget',ylabel=label,ylim=(-.002 if metric=='coverage' else -.02,upper),title=family)
            ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT/f'plots/extension/{family}_{metric}.png',dpi=160); plt.close(fig)


if __name__=='__main__': main()
