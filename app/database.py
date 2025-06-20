from fastapi import FastAPI
from pydantic import BaseModel
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os

app = FastAPI()

# ---- SETUP GOOGLE SHEETS ----
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_json_str = os.environ["GOOGLE_CREDS_JSON"]
creds_json = json.loads(creds_json_str)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, SCOPE)  # path to your key file
CLIENT = gspread.authorize(CREDS)

sheet = CLIENT.open_by_key("1IFoQ9PJoralucmufWa11IZ0Njcyq_-Z8NjLmtEySMdY")  # Replace with your actual ID
worksheet = sheet.sheet1    # opens the first sheet

# ---- FASTAPI MODEL ----
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

    # Save to Google Sheet
    worksheet.append_row([
        data.userId,
        data.name,
        data.gpax,
        data.tgat1,
        data.tgat2,
        data.tgat3
    ])

    return {"message": "Data saved to Google Sheets successfully"}
