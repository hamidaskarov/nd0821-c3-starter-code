# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
It is RandomForestClassifier model using the default hyperparameters in scikit-learn. It is 
used to predict if salary of someone is greater or lower that $50k.
## Intended Use
Model is intended to use for educational purpose. 

## Training Data
The data that is usd for training this model came from the Census Beurau.
 More information here: https://archive.ics.uci.edu/ml/datasets/census+income
## Evaluation Data
Evaluation data comes from the same dataset, acquired using train_test_split function of sklearn.
## Metrics
Precision: 0.7668944570994685
Recall: 0.6320400500625782
F-beta: 0.692967409948542

## Ethical Considerations
Some bias is present at the unsupervised and supervised level. This implies an unfairness in the underlying data and also unfairness in the model.
## Caveats and Recommendations
Do not use this model for real products withoud further investigation about data, and its fairness.