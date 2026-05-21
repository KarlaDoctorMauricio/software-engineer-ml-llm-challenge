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
    
@app.post("/predict", status_code=200)
async def post_predict() -> dict:
    return