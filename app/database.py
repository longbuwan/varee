from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os

app = FastAPI()
router = APIRouter()

# ---- SETUP GOOGLE SHEETS ----
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_json_str = os.getenv("GOOGLE_CREDS_JSON")
creds_json = json.loads(creds_json_str)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, SCOPE)
CLIENT = gspread.authorize(creds)

sheet = CLIENT.open_by_key("1IFoQ9PJoralucmufWa11IZ0Njcyq_-Z8NjLmtEySMdY")
worksheet = sheet.sheet1

# ---- FASTAPI MODEL ----
class ScoreSubmission(BaseModel):
    userId: str
    name: str
    gpax: float
    tgat1: float
    tgat2: float
    tgat3: float

@router.post("/api/save-score")
async def save_score(data: ScoreSubmission):
    print("Received data:", data)

    # Save to Google Sheet
    def upsert_user_data(worksheet, data):
        # Get all values in the sheet
        all_values = worksheet.get_all_values()

        # Look for userId in the first column (assumes userId is in column A)
        for i, row in enumerate(all_values):
            if row and row[0] == data.userId:
                # Update the row
                worksheet.update(f'A{i+1}:F{i+1}', [[
                    data.userId,
                    data.name,
                    data.gpax,
                    data.tgat1,
                    data.tgat2,
                    data.tgat3
                ]])
                return  # Exit after updating

        # If not found, append as a new row
        worksheet.append_row([
            data.userId,
            data.name,
            data.gpax,
            data.tgat1,
            data.tgat2,
            data.tgat3
        ])

    # Call the inner function
    upsert_user_data(worksheet, data)

    return {"message": "Data saved to Google Sheets successfully"}

# Register the router
app.include_router(router)
