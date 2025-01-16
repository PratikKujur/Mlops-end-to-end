from fastapi import FastAPI
import uvicorn
import pickle
import yaml
from pydantic import BaseModel
import numpy as np
import pickle
from typing import List
from src.ModelPedict import ModelPredict


params_trainer=yaml.safe_load(open("/Users/pratik.kujur/Desktop/Projects/Mlops-end-to-end/params.yaml"))['train']

app = FastAPI(title="Wine Quality Prediction API", description="Predict wine quality and type (red or not)")

model_reg=pickle.load(open(params_trainer['model_reg'],'rb'))
model_clf=pickle.load(open(params_trainer['model_clf'],'rb'))


class WineInput(BaseModel):
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


class PredictionOutput(BaseModel):
    quality: float  
    is_red: int     


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: WineInput):
    
    features = np.array([[
        
        input_data.volatile_acidity,
        input_data.citric_acid,
        input_data.residual_sugar,
        input_data.chlorides,
        input_data.free_sulfur_dioxide,
        input_data.total_sulfur_dioxide,
        input_data.density,
        input_data.pH,
        input_data.sulphates,
        input_data.alcohol
    ]])

    predictions=ModelPredict(params_trainer['model_reg'],params_trainer['model_clf'],features)
    

    return PredictionOutput(quality=predictions[0], is_red=predictions[1])
