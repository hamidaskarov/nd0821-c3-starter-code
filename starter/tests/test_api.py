import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.append('./')

from main import app

@pytest.fixture(scope="session")
def client():
    client = TestClient(app)
    return client

def test_index_get(client):
    """Test standard get"""
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "Welcome! Please use /predict endpoint for inference"}

def test_404(client):
    """Test response for non existent path"""

    res = client.get("/fake_endpoint")

    assert res.status_code != 200
    assert res.json() == {"detail":"Not Found"}


def test_post_above(client):
    """Test for salary > $50K"""

    res = client.post("/predict", json={
                    "age": 52,
                    "workclass": "Self-emp-not-inc",
                    "fnlgt": 209642,
                    "education": "HS-grad",
                    "education-num": 9,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 45,
                    "native-country": "United-States",
            })

    assert res.status_code == 200
    assert res.json() == {'Prediction result': '>50K'}


def test_post_below(client):
    """Test for salary <= $50K"""
    res = client.post("/predict", json={
                    "age": 37,
                    "workclass": "Private",
                    "fnlgt": 284582,
                    "education": "Masters",
                    "education-num": 14,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Exec-managerial",
                    "relationship": "Wife",
                    "race": "White",
                    "sex": "Female",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
            })
    assert res.status_code == 200
    assert res.json() == {'Prediction result': '<=50K'}
