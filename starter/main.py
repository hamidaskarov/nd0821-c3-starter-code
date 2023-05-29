# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference
import pickle as pkl
import pandas as pd
import os

app = FastAPI()
fpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"model","model.pkl")
with open(fpath, "rb") as f:
    encoder, lb, model = pkl.load(f)

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


class BasicInputData(BaseModel):
    """
    Schema for the input data on the POST method.

    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        """
        FastAPI autogenerates documentation for the data model/API etc.
        schema_extra is just used as an example for documentation purposes.
        Go to http://172.0.0.1:8000/docs to view the docs.
        """
        schema_extra = {
            "example": {
                    "age": 30,
                    "workclass": "State-gov",
                    "fnlgt": 141297,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Married-civ-spouse",
                    "occupation": "Prof-specialty",
                    "relationship": "Husband",
                    "race": "Asian-Pac-Islander",
                    "sex": "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "India",
            }
        }

@app.get('/')
def index():
    return {"message": "Welcome! Please use /predict endpoint for inference"}


@app.post('/predict')
def model_inference(input: BasicInputData):

    x = pd.DataFrame({k:v for k,v in input.dict(by_alias=True).items()},index=[0])


    X_test, _, _, _ = process_data(
        X=x, 
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        label=None,
        training=False
    )

    res = inference(model,X_test)

    return {"Prediction result": "<=50K" if int(res[0]) == 0 else ">50K"}