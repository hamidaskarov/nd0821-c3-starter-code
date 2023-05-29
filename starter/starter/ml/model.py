from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds =  model.predict(X)
    return preds

def data_slicing_metrics(test, feature_to_slice, model,encoder,lb, cat_features, label="salary"):
    """
    Performs data slicing

    Inputs
    """
    unique_items = test[feature_to_slice].unique()
    for item in unique_items:
        x_test = test[test[feature_to_slice]==item]
        y_test = x_test.pop(label)

        X_test, y_test, encoder, lb = process_data(
            test, 
            categorical_features=cat_features,
            encoder=encoder,
            lb=lb,
            label="salary",
            training=False
        )

        y_preds = inference(model,X_test)

        precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

        print(f"Sliced feature: {feature_to_slice}. Value: {item.strip()}. \
             Scores: precision {precision} | recall {recall} | fbeta {fbeta}")
