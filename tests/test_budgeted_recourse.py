import numpy as np
import pandas as pd
import pytest
from src.budgeted_recourse import lower_bound, execute_actions, candidate_pool, relative_cost, select_and_certify
from src.framework import actionable


def setup():
    ref=pd.DataFrame({'income':[100.,200.,400.],'loan':[50.,100.,200.],'Age':[35,54,65]})
    original=ref.iloc[1].copy()
    policy={'mutable':{'income':{'direction':'increase','cost':1},'loan':{'direction':'decrease','cost':1}}}
    settings={'sobol_power':3,'axis_steps':4,'max_relative_change':.5,'budgets':[.25,.5,1.],'shortlist_per_budget':4,'screen_trials':20,'certificate_trials':100,'execution_fraction':[.8,1.],'target_success':.9,'alpha':.05,'max_probability_std':.15}
    return ref,original,policy,settings


class Threshold:
    def __init__(self, threshold=220): self.threshold=threshold
    def predict(self,x): return (x.income>=self.threshold).astype(int).to_numpy()
    def predict_proba(self,x):
        p=np.where(self.predict(x),.8,.2)
        return np.column_stack([1-p,p])


def test_exact_confidence_bound():
    assert lower_bound(0,100,.05)==0
    assert lower_bound(100,100,.05)==pytest.approx(.05**.01)
    assert lower_bound(90,100,.01)<lower_bound(90,100,.05)<.9
    with pytest.raises(ValueError): lower_bound(2,1,.05)


def test_budget_and_immutable_age():
    ref,original,policy,settings=setup()
    pool=candidate_pool(original,ref,policy,settings,42)
    assert pool.Age.eq(54).all()
    assert all(actionable(original,row,ref,policy) for _,row in pool.iterrows())
    assert (pool.income<=300).all() and (pool.loan>=50).all()
    assert relative_cost(original,pd.DataFrame([original]),ref,policy)[0]==0
    candidate=original.copy(); candidate.income=300
    assert relative_cost(original,pd.DataFrame([candidate]),ref,policy)[0]==pytest.approx(.5)


def test_incomplete_execution_preserves_untouched_features():
    ref,original,policy,settings=setup(); candidate=original.copy(); candidate.income=300
    out=execute_actions(original,candidate,policy,settings,100,4)
    assert out.Age.eq(54).all() and out.loan.eq(original.loan).all()
    assert out.income.between(280,300).all()
    pd.testing.assert_frame_equal(out,execute_actions(original,candidate,policy,settings,100,4))


def test_independent_certificate_and_abstention():
    ref,original,policy,settings=setup(); model=Threshold()
    candidates,screen,choices,cert=select_and_certify(original,model,[model,model],ref,policy,settings,42)
    assert len(cert)>0
    assert cert.alpha_per_candidate.eq(settings['alpha']/len(cert)).all()
    for (budget,method),idx in choices.items():
        if idx is not None:
            assert screen.loc[idx,'cost']<=budget
            if method=='Certified': assert cert.set_index('candidate_id').loc[idx,'lower_bound']>=.9
    # A committee member that never approves precludes joint certification.
    _,_,choices,cert=select_and_certify(original,model,[model,Threshold(1000)],ref,policy,settings,42)
    assert all(v is None for (b,m),v in choices.items() if m=='Certified')


def test_failed_certificate_abstains_without_retry(monkeypatch):
    import src.budgeted_recourse as module
    ref,original,policy,settings=setup(); model=Threshold(); calls=[]
    def fake_simulation(original,candidates,models,policy,settings,trials,seed):
        calls.append((trials,seed))
        # Screening looks perfect; fresh certification does not support success.
        successes=trials if len(calls)==1 else 0
        return [dict(anchor_success=successes/trials,joint_success=successes/trials,joint_count=successes) for _ in range(len(candidates))]
    monkeypatch.setattr(module,'simulate',fake_simulation)
    _,_,choices,cert=select_and_certify(original,model,[model,model],ref,policy,settings,42)
    assert len(cert)>0 and not cert.certified.any()
    assert all(v is None for (b,m),v in choices.items() if m=='Certified')
    assert any(v is not None for (b,m),v in choices.items() if m=='Nominal')
    assert len(calls)==2 and calls[0][1]!=calls[1][1]
