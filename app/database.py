from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

class ScoreSubmission(BaseModel):
    userId: str
    name: str
    gpax: float
    tgat1: float
    tgat2: float
    tgat3: float

@app.post("/api/save-score")
async def save_score(data: ScoreSubmission):
    print("Received data:", data)
    # TODO: Save to database here (MongoDB, Firebase, etc.)
    return {"message": "Data saved successfully"}
