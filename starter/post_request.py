import requests

url = 'https://render-deployment-example-ffen.onrender.com/predict'

input = {
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
  "native-country": "India"
}

x = requests.post(url, json = input)

print(x.json())