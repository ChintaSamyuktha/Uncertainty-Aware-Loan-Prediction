"""Budgeted recourse with independent, selection-valid Monte Carlo certification.

The exact binomial confidence bound is standard statistics, not a new theorem.
The certificate concerns a fixed finite committee and a specified simulator only.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta, qmc
from src.framework import actionable, bounds


def lower_bound(successes, trials, alpha):
    """One-sided exact Clopper-Pearson lower confidence bound."""
    if trials < 1 or not 0 <= successes <= trials or not 0 < alpha < 1:
        raise ValueError('Invalid binomial counts or confidence level')
    return 0.0 if successes == 0 else float(beta.ppf(alpha, successes, trials-successes+1))


def relative_cost(original, candidates, reference, policy):
    """Sum of weighted changes relative to applicant values (training IQR floor)."""
    cost=np.zeros(len(candidates))
    for c, rule in policy['mutable'].items():
        floor=max(float(reference[c].quantile(.75)-reference[c].quantile(.25)),1.0)
        scale=max(abs(float(original[c])),floor)
        cost += rule['cost'] * abs(candidates[c].astype(float).to_numpy()-float(original[c]))/scale
    return cost


def candidate_pool(original, reference, policy, settings, seed):
    columns=list(policy['mutable']); dimensions=len(columns)
    fractions=qmc.Sobol(dimensions,scramble=True,seed=seed).random_base2(settings['sobol_power'])
    axes=[]
    for i in range(dimensions):
        for level in np.linspace(0,1,settings['axis_steps']+1)[1:]:
            action=np.zeros(dimensions); action[i]=level; axes.append(action)
    fractions=np.vstack([np.zeros(dimensions), fractions, axes])
    pool=pd.DataFrame([original.to_dict()]*len(fractions))[reference.columns]
    limits=bounds(reference,policy)
    for j,c in enumerate(columns):
        value=float(original[c]); lo,hi=limits[c]
        floor=max(float(reference[c].quantile(.75)-reference[c].quantile(.25)),1.0)
        distance=settings['max_relative_change']*max(abs(value),floor)
        if policy['mutable'][c]['direction']=='increase':
            endpoint=max(value,min(hi,value+distance))
        else:
            endpoint=min(value,max(lo,value-distance))
        pool[c]=value+(endpoint-value)*fractions[:,j]
    pool=pool.drop_duplicates().reset_index(drop=True)
    feasible=[actionable(original,row,reference,policy) for _,row in pool.iterrows()]
    return pool.loc[feasible].reset_index(drop=True)


def execute_actions(original, candidate, policy, settings, trials, seed):
    """IID incomplete execution; untouched features stay exactly unchanged."""
    rng=np.random.default_rng(seed)
    low,high=settings['execution_fraction']
    if not 0 <= low <= high <= 1 or trials < 1:
        raise ValueError('Execution fraction must lie in [0,1]; trials must be positive')
    frame=pd.DataFrame([candidate.to_dict()]*trials)
    for c in policy['mutable']:
        delta=float(candidate[c])-float(original[c])
        frame[c]=float(original[c])+rng.uniform(low,high,trials)*delta
    return frame


def simulate(original, candidates, models, policy, settings, trials, seed):
    """Batch model calls; rows are IID trials, columns are fixed committee models."""
    if not len(candidates): return []
    frames=[execute_actions(original,row,policy,settings,trials,seed+i) for i,(_,row) in enumerate(candidates.iterrows())]
    data=pd.concat(frames,ignore_index=True)
    predictions=np.column_stack([model.predict(data) for model in models]).reshape(len(candidates),trials,len(models))
    return [dict(anchor_success=float(p[:,0].mean()),joint_success=float((p==1).all(axis=1).mean()),joint_count=int((p==1).all(axis=1).sum())) for p in predictions]


def select_and_certify(original, model, committee, reference, policy, settings, seed):
    """Freeze one candidate per budget BEFORE drawing certificate samples.

    No failed candidate is replaced after seeing certification data. Bonferroni
    correction permits reporting any subset of the preselected budget candidates.
    """
    pool=candidate_pool(original,reference,policy,settings,seed)
    if not len(pool): return pool,pd.DataFrame(),{},pd.DataFrame()
    probabilities=np.column_stack([m.predict_proba(pool)[:,1] for m in committee])
    costs=relative_cost(original,pool,reference,policy)
    nominal=model.predict(pool)==1
    ids=set()
    half=max(1,settings['shortlist_per_budget']//2)
    for budget in settings['budgets']:
        eligible=np.where(nominal & (costs<=budget))[0]
        ids.update(eligible[np.argsort(costs[eligible])[:half]].tolist())
        ids.update(eligible[np.argsort(-probabilities[eligible].min(axis=1))[:half]].tolist())
    ids=sorted(ids)
    candidates=pool.iloc[ids].reset_index(drop=True)
    screened=simulate(original,candidates,committee,policy,settings,settings['screen_trials'],seed+10000)
    statistics=[]
    for i, stats in enumerate(screened):
        statistics.append(dict(candidate_id=i,cost=float(costs[ids[i]]),probability=float(probabilities[ids[i],0]),probability_std=float(probabilities[ids[i]].std(ddof=1)),sparsity=int((candidates.iloc[i]!=original).sum()),**stats))
    screen=pd.DataFrame(statistics,columns=['candidate_id','cost','probability','probability_std','sparsity','anchor_success','joint_success','joint_count'])
    choices={}
    for budget in settings['budgets']:
        eligible=screen[screen.cost<=budget].sort_values(['cost','candidate_id'])
        masks={
            'Nominal':np.ones(len(eligible),dtype=bool),
            'Uncertainty':eligible.probability_std<=settings['max_probability_std'],
            'Point stability':eligible.anchor_success>=settings['target_success'],
            'Joint empirical':eligible.joint_success>=settings['target_success'],
            'Certified':eligible.joint_success>=settings['target_success'],
        }
        for method,mask in masks.items():
            accepted=eligible.loc[mask]
            choices[(float(budget),method)]=None if accepted.empty else int(accepted.iloc[0].candidate_id)
    chosen=sorted({v for (budget,method),v in choices.items() if method=='Certified' and v is not None})
    certificates=[]
    if chosen:
        outcomes=simulate(original,candidates.iloc[chosen],committee,policy,settings,settings['certificate_trials'],seed+100000)
        for candidate_id,stats in zip(chosen,outcomes):
            bound=lower_bound(stats['joint_count'],settings['certificate_trials'],settings['alpha']/len(chosen))
            certificates.append(dict(candidate_id=candidate_id,trials=settings['certificate_trials'],successes=stats['joint_count'],lower_bound=bound,alpha_per_candidate=settings['alpha']/len(chosen),certified=bound>=settings['target_success']))
    cert=pd.DataFrame(certificates,columns=['candidate_id','trials','successes','lower_bound','alpha_per_candidate','certified'])
    for key,candidate_id in list(choices.items()):
        if key[1]=='Certified' and candidate_id is not None and not bool(cert.set_index('candidate_id').loc[candidate_id,'certified']):
            choices[key]=None
    return candidates,screen,choices,cert
