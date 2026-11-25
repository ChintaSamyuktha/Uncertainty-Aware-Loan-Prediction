from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.budgeted_recourse import lower_bound
from src.framework import actionable

ROOT=Path(__file__).resolve().parents[1]


def test_extension_certificate_and_audit_integrity():
    out=ROOT/'results/extension'
    decisions=pd.read_csv(out/'decisions.csv')
    cert=pd.read_csv(out/'certificates.csv')
    candidates=pd.read_csv(out/'candidates.csv')
    metrics=pd.read_csv(out/'model_metrics.csv')
    assert len(metrics)==6
    assert decisions.groupby(['seed','family','case_id']).size().eq(15).all()
    assert all(np.isclose(row.lower_bound,lower_bound(row.successes,row.trials,row.alpha_per_candidate)) for row in cert.itertuples())
    passing=cert[cert.certified].set_index(['seed','family','case_id','candidate_id'])
    for row in decisions[decisions.accepted & decisions.method.eq('Certified')].itertuples():
        assert passing.loc[(row.seed,row.family,row.case_id,row.candidate_id),'lower_bound']>=.9
    for keys,sub in decisions[decisions.accepted].groupby(['seed','family','case_id','candidate_id']):
        assert sub.audit_joint_success.nunique()==1
        assert sub.audit_transfer_joint_success.nunique()==1
    for (seed,family),sub in candidates.groupby(['seed','family']):
        bundle=joblib.load(ROOT/f'artifacts/extension/{family}_{seed}.joblib')
        for _,row in sub.iterrows():
            original=pd.Series({c:row[f'original_{c}'] for c in bundle['reference']})
            candidate=pd.Series({c:row[f'cf_{c}'] for c in bundle['reference']})
            assert actionable(original,candidate,bundle['reference'],bundle['policy'])
    splits=pd.read_csv(out/'splits.csv')
    assert splits.groupby(['seed','row']).size().eq(1).all()
    assert len(list((ROOT/'plots/extension').glob('*.png')))==6
