import hashlib
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app import config
from backend.app.services.model_service import get_model_service

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'DATABASE_PATH', tmp_path / 'history.db')
    with TestClient(app) as test_client:
        yield test_client

PATIENT = dict(age=55, trestbps=140, chol=240, thalch=150, oldpeak=1.5, ca=0)

def test_prediction_matches_saved_model_and_history_privacy(client):
    service = get_model_service()
    expected = service.model.predict_proba(pd.DataFrame([PATIENT], columns=service.metadata['features']).astype('float64'))[0,1]
    response = client.post('/api/predict', json=PATIENT)
    assert response.status_code == 200
    assert response.json()['probability'] == pytest.approx(expected)
    assert response.json()['threshold'] == service.metadata['threshold']
    assert client.get('/api/history').json() == []
    assert client.post('/api/predict', json={**PATIENT,'save_history':True}).status_code == 200
    rows = client.get('/api/history').json()
    assert len(rows) == 1
    assert set(rows[0]) == {'id','prediction','probability','model_name','model_version','created_at'}
    assert client.post('/api/explain', json={**PATIENT,'save_history':True}).status_code == 200
    assert len(client.get('/api/history').json()) == 1
    assert client.delete('/api/history').status_code == 200
    assert client.get('/api/history').json() == []

@pytest.mark.parametrize('payload',[{}, {'age':0}, {'age':130}, {'age':55,'chol':0}, {'age':55,'ca':4}, {'age':55,'sex':3}, {'age':55,'patient_name':'private'}, {'age':55,'thalch':100,'thalach':120}])
def test_validation(client,payload):
    assert client.post('/api/predict',json=payload).status_code == 422

def test_alias_and_missing_features(client):
    canonical = client.post('/api/predict',json=PATIENT).json()
    alias = {**PATIENT}; alias['thalach'] = alias.pop('thalch')
    assert client.post('/api/predict',json=alias).json() == canonical
    assert client.post('/api/predict',json={'age':55}).status_code == 200

def test_explanation_is_real_and_additive(client):
    prediction = client.post('/api/predict',json=PATIENT).json()
    response = client.post('/api/explain',json=PATIENT)
    assert response.status_code == 200, response.text
    explanation = response.json()
    assert explanation['probability'] == pytest.approx(prediction['probability'])
    assert len(explanation['contributions']) == 13
    assert explanation['additivity_error'] < 1e-5
    assert explanation['base_probability'] + sum(x['shap_value'] for x in explanation['contributions']) == pytest.approx(prediction['probability'])

@pytest.mark.parametrize('route',['health','models','metrics','comparison','hospitals','features','analytics'])
def test_reports(client,route):
    response = client.get('/api/'+route)
    assert response.status_code == 200, response.text
    assert response.json()

def test_analytics_actual_dataset(client):
    df = pd.read_csv(config.DATA_PATH)
    result = client.get('/api/analytics').json()
    assert result['total_records'] == len(df)
    assert result['disease_cases'] == int(df.target.sum())
    assert sum(result['hospital_distribution'].values()) == len(df)
    assert client.get('/api/features').json()[0]['label'] == 'Maximum Heart Rate'

def test_mlflow_readonly(client):
    path = config.ROOT/'mlflow.db'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    experiments = client.get('/api/experiments').json()
    assert experiments
    total = 0
    for experiment in experiments:
        response = client.get(f"/api/experiments/{experiment['id']}/runs")
        assert response.status_code == 200
        total += len(response.json())
    assert total > 0
    assert client.get('/api/experiments/999999/runs').status_code == 404
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before

def test_local_security_and_allowlisted_plots(client):
    assert client.post('/api/predict',json=PATIENT,headers={'Origin':'https://example.com'}).status_code == 403
    assert client.get('/api/health',headers={'Host':'example.com'}).status_code == 400
    assert client.get('/api/plots/evaluation/roc_curve.png').headers['content-type'] == 'image/png'
    assert client.get('/api/plots/evaluation/test_predictions.csv').status_code == 404
    assert client.get('/api/health').headers['cache-control'] == 'no-store'
    assert client.get('/api/history?limit=999').status_code == 422
