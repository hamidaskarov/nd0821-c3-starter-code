import sys
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append('./')

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics


@pytest.fixture
def data():
    df = pd.read_csv('./data/cleaned_census.csv')

    train, test = train_test_split(df, test_size=0.15)

    return train, test

@pytest.fixture
def processed_data():
    df = pd.read_csv('./data/cleaned_census.csv')
    train, test = train_test_split(df, test_size=0.15)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        X=train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    X_test, y_test, encoder, lb = process_data(
        test, 
        categorical_features=cat_features,
        encoder=encoder, 
        lb=lb, 
        label="salary", 
        training=False
    )

    return X_train, X_test, y_train, y_test

def test_process_data(data):
    """
    Test if data is in right shape after processing
    """
    train, test = data

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X_train, y_train, cat, lb = process_data(
        X=train,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert len(X_train)==len(y_train)


def test_model(processed_data):
    """
    Test if model prediction shape is correct
    """
    X_train, X_test, y_train, y_test = processed_data 

    model = train_model(X_train, y_train)
    test_pred = model.predict(X_test)

    assert len(test_pred) == len(y_test)


def test_metrics(processed_data):
    """
    Test if metrics is in correct range
    """
    X_train, X_test, y_train, y_test = processed_data 
    model = train_model(X_train, y_train)
    test_preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, test_preds)

    assert precision <= 1.0
    assert recall <= 1.0
    assert fbeta <= 1.0