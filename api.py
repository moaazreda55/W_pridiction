from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle 
import pandas as pd

app = FastAPI()

class Features(BaseModel):
    features : List[float]  # List to hold all features (e.g., [Gender, Height, Weight])
      
with open("Weight Predection model_tree.pkl", "rb") as f:
    model = pickle.load(f)    
    
@app.get('/')
def hello():
    return {"message": "Hello from post or postman, FastAPI!"}

@app.post('/predict/')
async def predict(features:Features):
    input_data = pd.DataFrame([features.features])
    prediction = model.predict(input_data)
    return {"prediction":int(prediction[0])}