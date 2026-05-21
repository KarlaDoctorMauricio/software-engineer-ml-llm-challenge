import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
from challenge.model import DelayModel, TOP_FEATURES
import pandas as pd

app = fastapi.FastAPI()

# Initialize ML model instance
model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

# Request schema
class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class Request(BaseModel):
    flights: List[Flight]

# Validation rules
VALID_TIPOVUELO = {"N", "I"}
VALID_MES = set(range(1, 13))


def validate_flight(flight: Flight) -> None:
    if flight.MES not in VALID_MES:
        raise HTTPException(status_code=400)

    if flight.TIPOVUELO not in VALID_TIPOVUELO:
        raise HTTPException(status_code=400)

@app.post("/predict", status_code=200)
async def post_predict(body: Request) -> dict:
    # Validate all incoming flight records before processing
    for flight in body.flights:
        validate_flight(flight)

    # Convert request payload into pandas DataFrame for preprocessing
    df = pd.DataFrame([f.dict() for f in body.flights])

    # Apply feature engineering using trained preprocessing pipeline
    features = model.preprocess(df)

    # Generate predictions using trained model
    predictions = model.predict(features)

    # Return predictions in expected API format
    return {"predict": predictions}