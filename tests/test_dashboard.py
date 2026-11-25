from pathlib import Path
from streamlit.testing.v1 import AppTest
import pandas as pd
import joblib
from src.framework import complete


def test_dashboard_pages_and_prediction():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=120).run()
    assert not app.exception
    for page in ['Dataset','Model performance','SHAP explanations','Counterfactual results','Research extension','Comparison & downloads','Applicant lab']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric)>=3


def test_rejected_applicant_recourse():
    root=Path(__file__).resolve().parents[1]
    reference=joblib.load(root/'artifacts/research.joblib')['reference']
    row=complete(pd.read_csv(root/'train.csv').iloc[[120]][reference.columns],reference).iloc[0]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=120).run()
    app.sidebar.radio[0].set_value('Applicant lab').run()
    for widget in app.number_input:
        widget.set_value(float(row[widget.label]))
    for widget in app.selectbox:
        widget.set_value(row[widget.label])
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value=='Rejected'
    assert len(app.dataframe)>=1
