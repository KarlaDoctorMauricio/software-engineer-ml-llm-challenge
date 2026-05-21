import fastapi
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List

app = fastapi.FastAPI()

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
async def post_predict() -> dict:
    return