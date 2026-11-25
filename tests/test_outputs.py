from pathlib import Path
import joblib
import pandas as pd

def test_saved_outputs_and_predictions():
    root=Path(__file__).resolve().parents[1]
    bundle=joblib.load(root/'artifacts/research.joblib')
    raw=pd.read_csv(root/'train.csv'); splits=pd.read_csv(root/'results/splits.csv')
    assert splits.row.nunique()==len(raw)
    assert splits.groupby('row').size().max()==1
    results=pd.read_csv(root/'results/counterfactuals.csv')
    if len(results):
        assert results.query("method == 'Proposed'").reliable.all()
        for _,row in results.iterrows():
            frame=pd.DataFrame([{c:row[f'cf_{c}'] for c in bundle['reference']}])
            assert bool(bundle['model'].predict(frame)[0]==1)==bool(row.validity)
    assert len(pd.read_csv(root/'results/test_predictions.csv'))==367
    assert len(list((root/'plots').glob('*.png')))>=12
