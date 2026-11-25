import numpy as np
import pandas as pd
import pytest
from src.framework import actionable, perturb, stability, assess

@pytest.fixture
def data():
    reference=pd.DataFrame({'ApplicantIncome':[100.,500.,1000.], 'LoanAmount':[10.,50.,100.], 'Gender':['F','M','F']})
    original=reference.iloc[1].copy()
    config={'mutable':{'ApplicantIncome':{'direction':'increase','cost':1},'LoanAmount':{'direction':'decrease','cost':1}},'perturbations':100,'perturbation_fraction':.02,'reliability':{'max_uncertainty':.15,'min_stability':.9,'max_proximity':.5,'max_sparsity':2}}
    return reference,original,config

def test_immutable_and_direction(data):
    ref,original,cfg=data; candidate=original.copy(); candidate.ApplicantIncome=600
    assert actionable(original,candidate,ref,cfg)
    candidate.Gender='F'; assert not actionable(original,candidate,ref,cfg)
    candidate=original.copy(); candidate.ApplicantIncome=400
    assert not actionable(original,candidate,ref,cfg)
    candidate.ApplicantIncome=1001; assert not actionable(original,candidate,ref,cfg)
    candidate.ApplicantIncome=float('nan'); assert not actionable(original,candidate,ref,cfg)

def test_perturbations_feasible_and_reproducible(data):
    ref,original,cfg=data; candidate=original.copy(); candidate.ApplicantIncome=700
    first=perturb(original,candidate,ref,cfg,8)
    pd.testing.assert_frame_equal(first,perturb(original,candidate,ref,cfg,8))
    assert all(actionable(original,row,ref,cfg) for _,row in first.iterrows())
    assert first.ApplicantIncome.nunique()>1

def test_stability():
    assert stability([1,1,0,1])==.75
    assert stability([0,0])==0
    with pytest.raises(ValueError): stability([])


def test_age_cannot_be_recommended_as_younger(data):
    # Hypothetical feature exercises the immutable allowlist; not training data.
    ref,original,cfg=data
    ref=ref.assign(Age=[35,54,65])
    original=original.copy(); original['Age']=54
    candidate=original.copy(); candidate['ApplicantIncome']=700
    assert actionable(original,candidate,ref,cfg)
    candidate['Age']=35
    assert not actionable(original,candidate,ref,cfg)
    candidate['Age']=54
    nearby=perturb(original,candidate,ref,cfg,42)
    assert nearby.Age.eq(54).all()

class ThresholdModel:
    def predict_proba(self,x):
        p=np.where(x.ApplicantIncome>=600,.8,.2)
        return np.column_stack([1-p,p])
    def predict(self,x): return (self.predict_proba(x)[:,1]>=.5).astype(int)

def test_validity_and_reliability(data):
    ref,original,cfg=data; candidate=original.copy(); candidate.ApplicantIncome=700
    model=ThresholdModel()
    score,_=assess(original,candidate,model,[model,model],ref,cfg)
    assert score['validity'] and score['reliable'] and score['uncertainty']==0
    score,_=assess(original,original,model,[model,model],ref,cfg)
    assert not score['validity'] and not score['reliable']
