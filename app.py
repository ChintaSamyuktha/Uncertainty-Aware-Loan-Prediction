"""Interactive research dashboard backed by reproducible saved artifacts."""
from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.framework import dice_generator, generate, assess

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Reliable Recourse Lab',page_icon='🔎',layout='wide')
st.markdown('<style>.stApp {background:linear-gradient(120deg,#f8fafc,#eef6f5)} h1,h2,h3 {color:#134e4a}</style>',unsafe_allow_html=True)
st.title('Reliable Recourse Lab')
st.caption('Uncertainty-aware actionable counterfactual explanations · Academic demonstration')
if not (ROOT/'artifacts/research.joblib').exists() or not (ROOT/'results/experimental_summary.md').exists():
    st.error('Research artifacts are missing. Run python run_experiment.py first.'); st.stop()

@st.cache_resource
def load(): return joblib.load(ROOT/'artifacts/research.joblib')
bundle=load(); model=bundle['model']; reference=bundle['reference']; config=bundle['config']
page=st.sidebar.radio('Explore',['Overview','Dataset','Model performance','Applicant lab','SHAP explanations','Counterfactual results','Research extension','Comparison & downloads'])
st.sidebar.info(f"Saved model: {bundle['selected']}\n\nNo real-world approval guarantee.")

def chart(name):
    path=ROOT/'plots'/f'{name}.png'
    if path.exists(): st.image(str(path),use_container_width=True)


def show_actionability():
    st.subheader('What can and cannot change?')
    st.info('Age cannot be changed as an action. For example, a 54-year-old applicant must remain 54 in every recommendation; suggesting age 35 would be rejected. The current datasets do not contain age, so this model cannot use age or explain a rejection as age-related.')
    policies=[]
    for feature in reference.columns:
        policy=config['mutable'].get(feature)
        policies.append({'Feature':feature,'Recommendation rule':('Increase only' if policy['direction']=='increase' else 'Decrease only') if policy else 'Cannot change — kept at the applicant’s original value'})
    st.dataframe(pd.DataFrame(policies),hide_index=True,use_container_width=True)
    st.caption('A feature may influence a prediction while remaining unavailable as an action. If no permitted change passes the reliability checks, the system reports no suitable recommendation.')

if page=='Overview':
    st.header('From a prediction to a measured recommendation')
    st.write('This project combines loan prediction, SHAP model explanations, constrained DiCE candidates, bootstrap probability dispersion and feasible perturbation testing. Recommendations are returned only when they pass the configured reliability rule.')
    cols=st.columns(3)
    for col,label,value in zip(cols,['Labeled records','Unlabeled test records','Bootstrap members'],[614,367,len(bundle['ensemble'])]): col.metric(label,value)
    st.warning('Demonstration policy: income increases and loan reductions are hypothetical actions requiring domain review. SHAP describes model behavior, not causation. This application is not a lending decision system.')
    st.markdown((ROOT/'results/experimental_summary.md').read_text(encoding='utf-8'))
elif page=='Dataset':
    import json
    audit=json.loads((ROOT/'results/data_audit.json').read_text())
    st.header('Dataset audit')
    st.json(audit)
    st.dataframe(pd.read_csv(ROOT/'train.csv'),use_container_width=True); chart('class_distribution')
elif page=='Model performance':
    st.header('Validation selection and held-out evaluation')
    st.dataframe(pd.read_csv(ROOT/'results/model_metrics.csv'),use_container_width=True)
    for name in ['model_performance','confusion_matrix','roc_curve','calibration']: chart(name)
    st.caption('Evaluation data are never used to fit preprocessing, choose the model or tune reliability thresholds. Calibration is assessed, not corrected.')
elif page=='SHAP explanations':
    st.header('What drives this model?')
    st.write('Attributions explain approval probability. Global plots use 40 held-out observations and a 40-row training background. Encoded category values in the summary are labels, not ordinal effects.')
    for name in ['shap_importance','shap_summary','shap_individual']: chart(name)
elif page=='Applicant lab':
    st.header('Explore an applicant')
    show_actionability()
    with st.form('applicant'):
        values={}; columns=st.columns(3)
        for i,c in enumerate(reference):
            with columns[i%3]:
                if pd.api.types.is_numeric_dtype(reference[c]):
                    if c in ['Credit_History','Loan_Amount_Term']:
                        options=sorted(reference[c].unique().tolist()); values[c]=st.selectbox(c,options,index=options.index(float(reference[c].mode().iloc[0])))
                    else: values[c]=st.number_input(c,min_value=float(reference[c].min()),max_value=float(reference[c].max()),value=float(reference[c].median()))
                else: values[c]=st.selectbox(c,sorted(reference[c].unique()))
        submitted=st.form_submit_button('Predict and assess recourse',type='primary')
    if submitted:
        original=pd.Series(values)[reference.columns]; frame=pd.DataFrame([original])
        probability=float(model.predict_proba(frame)[0,1]); prediction=int(model.predict(frame)[0])
        uncertainty=float(np.std([m.predict_proba(frame)[0,1] for m in bundle['ensemble']],ddof=1))
        a,b=st.columns(2); a.metric('Predicted outcome','Approved' if prediction else 'Rejected'); b.metric('Approval probability',f'{probability:.1%}')
        st.metric('Bootstrap probability standard deviation',f'{uncertainty:.3f}')
        with st.spinner('Explaining the prediction and testing feasible alternatives…'):
            import shap
            prep=model['preprocess']; estimator=model['classifier']
            background=prep.transform(reference.sample(40,random_state=42)); encoded=prep.transform(frame)
            explainer=shap.Explainer(lambda x:estimator.predict_proba(x)[:,1],background,algorithm='permutation',seed=42)
            local=explainer(encoded,max_evals=2*encoded.shape[1]+1)
            contributions=pd.Series(local.values[0],index=prep.get_feature_names_out()).sort_values()
            st.subheader('Local SHAP contributions to approval probability')
            st.bar_chart(contributions)
            if prediction:
                st.success('Already in the desired predicted class. No recourse is required.')
            else:
                splits=pd.read_csv(ROOT/'results/splits.csv'); raw=pd.read_csv(ROOT/'train.csv')
                y=raw.loc[splits.loc[splits.split=='train','row'],'Loan_Status'].map({'N':0,'Y':1})
                generator=dice_generator(reference,y,model)
                candidates,error=generate(generator,original,reference,config,config['seed'])
                records=[]
                for _,candidate in candidates.iterrows():
                    stats,_=assess(original,candidate,model,bundle['ensemble'],reference,config)
                    records.append({**candidate.to_dict(),**stats})
                if records:
                    results=pd.DataFrame(records); st.dataframe(results,use_container_width=True)
                    reliable=results[results.reliable]
                    if reliable.empty: st.warning('No candidate passed every reliability requirement.')
                    else:
                        best=reliable.sort_values(['uncertainty','proximity']).iloc[0]
                        st.subheader('Original vs recommended')
                        st.dataframe(pd.DataFrame({'Original':original,'Recommended':best[reference.columns]}))
                    st.download_button('Download assessed candidates',results.to_csv(index=False),'applicant_candidates.csv')
                else: st.warning('No feasible candidate found. '+error)
        st.caption('Uncertainty measures variation across bootstrap models. It is distinct from approval probability and is not a confidence interval or a guarantee.')
elif page=='Counterfactual results':
    st.header('Measured candidate reliability')
    show_actionability()
    st.dataframe(pd.read_csv(ROOT/'results/cases.csv'),use_container_width=True)
    candidates=pd.read_csv(ROOT/'results/counterfactuals.csv')
    st.dataframe(candidates,use_container_width=True)
    for name in ['original_vs_counterfactual','uncertainty_comparison','stability_comparison','reliability_coverage']: chart(name)
    st.json(config)
elif page=='Research extension':
    st.header('Budgeted recourse with independent certification')
    st.write('Can a recommendation survive incomplete execution across several trained models while staying within an applicant-relative action budget? This experiment compares five selection rules over three split seeds and two model families.')
    st.warning('Experimental research direction. Originality and publication acceptance are not established. Certificates concern the fixed models and declared execution simulator, not real-world loan approval.')
    extension=ROOT/'results/extension'
    if not (extension/'summary.md').exists():
        st.info('Run python run_research_extension.py to generate the completed experiment.')
    else:
        summary=pd.read_csv(extension/'summary.csv')
        family=st.selectbox('Model family',sorted(summary.family.unique()))
        st.dataframe(summary[summary.family==family],hide_index=True,use_container_width=True)
        for metric in ['coverage','audit_joint_success','audit_transfer_success']:
            chart(f'extension/{family}_{metric}')
        st.subheader('Why the certificate is different from a stability score')
        st.write('First select a candidate using screening trials. Then freeze it and simulate 400 fresh execution trials. Return it only if the exact lower confidence bound reaches 90% joint approval across the fixed committee. Budget choices share a 5% error allowance for this applicant/run. A failed certificate leads to abstention, without trying another candidate on the same certification samples.')
        st.caption('Execution assumption: each requested financial change is achieved at an independent uniformly sampled 80–100%. Age and other immutable fields cannot change. Transfer to fresh bootstrap models is audited separately and is not certified.')
        decisions=pd.read_csv(extension/'decisions.csv')
        subset=decisions[decisions.family==family]
        seed=st.selectbox('Split seed',sorted(subset.seed.unique()))
        subset=subset[subset.seed==seed]
        case=st.selectbox('Evaluated applicant row',sorted(subset.case_id.unique()))
        subset=subset[subset.case_id==case]
        st.dataframe(subset,hide_index=True,use_container_width=True)
        try:
            certificates=pd.read_csv(extension/'certificates.csv')
        except pd.errors.EmptyDataError:
            certificates=pd.DataFrame()
        if len(certificates):
            certificates=certificates[(certificates.family==family)&(certificates.seed==seed)&(certificates.case_id==case)]
            if len(certificates):
                st.subheader('Independent certificate records')
                st.dataframe(certificates,hide_index=True,use_container_width=True)
            else: st.info('No candidate was preselected for independent certification for this applicant.')
        candidates=pd.read_csv(extension/'candidates.csv')
        if len(candidates):
            candidates=candidates[(candidates.family==family)&(candidates.seed==seed)&(candidates.case_id==case)]
            if len(candidates):
                st.subheader('Searched candidate actions')
                st.dataframe(candidates,use_container_width=True)
            else: st.info('No nominal approval candidate was found within the configured action limits for this applicant.')
        with st.expander('Methodology and publication limitations'):
            st.markdown((ROOT/'docs/research_extension.md').read_text(encoding='utf-8'))
        for path in sorted(extension.glob('*')):
            if path.is_file(): st.download_button('Download '+path.name,path.read_bytes(),file_name=path.name,key='extension_'+path.name)
else:
    st.header('Baseline vs proposed framework')
    st.dataframe(pd.read_csv(ROOT/'results/baseline_comparison.csv'),use_container_width=True); chart('baseline_comparison')
    st.write('Both methods use the same constrained DiCE candidate pool. The proposed method filters and ranks this pool; it does not constitute a new search algorithm. Returned-candidate averages can improve through selection while coverage falls.')
    for path in sorted((ROOT/'results').glob('*')):
        if path.is_file(): st.download_button(path.name,path.read_bytes(),file_name=path.name,key=path.name)
    st.download_button('Download methodology and instructions',(ROOT/'README.md').read_bytes(),file_name='README.md')

