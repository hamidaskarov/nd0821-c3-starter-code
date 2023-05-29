# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, data_slicing_metrics
import pandas as pd
import pickle as pkl
# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('../data/cleaned_census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, encoder=encoder, lb=lb, label="salary", training=False
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save model and encoders for later use
with open("../model/model.pkl", "wb") as file:
    pkl.dump([encoder, lb, model], file)

# inference and evaluation
test_pred = inference(model,X_test)

precision, recall, fbeta = compute_model_metrics(y_test, test_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-beta: {fbeta}")


for column in cat_features:
    data_slicing_metrics(test,column,model,encoder,lb,cat_features,"salary")