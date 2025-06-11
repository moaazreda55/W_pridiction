from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import onnxruntime as rt
import numpy as np

app = FastAPI()

class Features(BaseModel):
    features: List[float]  # Example: [Gender, Height, Weight]

# Load ONNX model
session = rt.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

# Get input name
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

@app.get('/')
def hello():
    return {"message": "Hello from FastAPI with ONNX!"}

@app.post('/predict/')
async def predict(features: Features):
    input_data = np.array([features.features], dtype=np.float32)  # Must match training dtype
    prediction = session.run([output_name], {input_name: input_data})
    return {"prediction": int(prediction[0][0])}
