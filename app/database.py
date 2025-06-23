from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import Optional
import gspread.utils
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
worksheet = sheet.get_worksheet(0)  # First worksheet (index starts at 0)
datasheet = sheet.get_worksheet(1)

# ---- FASTAPI MODEL ----
class ScoreSubmission(BaseModel):
    userId: str
    name: str
    gpax: Optional[float] = None
    tgat1: Optional[float] = None
    tgat2: Optional[float] = None
    tgat3: Optional[float] = None

    tpat1_1: Optional[float] = None
    tpat1_2: Optional[float] = None
    tpat1_3: Optional[float] = None
    tpat2_1: Optional[float] = None
    tpat2_2: Optional[float] = None
    tpat2_3: Optional[float] = None
    tpat3: Optional[float] = None
    tpat4: Optional[float] = None
    tpat5: Optional[float] = None

    alevel1_1: Optional[float] = None
    alevel1_2: Optional[float] = None
    alevel2_1: Optional[float] = None
    alevel2_2: Optional[float] = None
    alevel2_3: Optional[float] = None
    alevel2_4: Optional[float] = None
    alevel3: Optional[float] = None
    alevel4_1: Optional[float] = None
    alevel4_2: Optional[float] = None
    alevel4_3: Optional[float] = None
    alevel4_4: Optional[float] = None
    alevel4_5: Optional[float] = None
    alevel4_6: Optional[float] = None
    alevel4_7: Optional[float] = None
    alevel4_8: Optional[float] = None
    alevel4_9: Optional[float] = None

class UserIdRequest(BaseModel):
    userId: str

# ---- COLLECT UNIQUE UNIVERSITIES ----
all_values = worksheet.get_all_values()

try:
    uni_col_index = all_values[0].index("University")
except ValueError:
    unique_universities = []  # fallback if "University" column is not found
else:
    universities = {
        row[uni_col_index].strip()
        for row in all_values[1:]
        if uni_col_index < len(row) and row[uni_col_index].strip()
    }
    unique_universities = sorted(universities)

# ---- ROUTES ----
@router.post("/api/show-uni")
async def get_score(data: UserIdRequest):
    return {"data": unique_universities}

@router.post("/api/save-score")
async def save_score(data: ScoreSubmission):
    print("Received data:", data)

    def upsert_user_data(worksheet, data):
        all_values = worksheet.get_all_values()

        columns = [
            "userId", "name", "gpax", "tgat1", "tgat2", "tgat3",
            "tpat1_1", "tpat1_2", "tpat1_3", "tpat2_1", "tpat2_2", "tpat2_3",
            "tpat3", "tpat4", "tpat5",
            "alevel1_1", "alevel1_2", "alevel2_1", "alevel2_2", "alevel2_3", "alevel2_4",
            "alevel3", "alevel4_1", "alevel4_2", "alevel4_3", "alevel4_4", "alevel4_5",
            "alevel4_6", "alevel4_7", "alevel4_8", "alevel4_9"
        ]

        # Find existing row by userId
        row_index = None
        for i, row in enumerate(all_values):
            if row and row[0] == data.userId:
                row_index = i + 1  # gspread is 1-indexed
                break

        if row_index:
            old_row = all_values[row_index - 1]
            if len(old_row) < len(columns):
                old_row += [''] * (len(columns) - len(old_row))

            new_row = []
            for idx, col in enumerate(columns):
                if col == "userId":
                    new_row.append(data.userId)
                elif col == "name":
                    new_row.append(data.name)
                else:
                    new_val = getattr(data, col)
                    new_row.append(str(new_val) if new_val is not None else '')

            cell_range = f"A{row_index}:{gspread.utils.rowcol_to_a1(row_index, len(columns))}"
            worksheet.update(cell_range, [new_row])

        else:
            # Append new row
            new_row = [data.userId, data.name]
            for col in columns[2:]:
                val = getattr(data, col)
                new_row.append(str(val) if val is not None else '')
            worksheet.append_row(new_row)

    upsert_user_data(worksheet, data)
    return {"message": "Data saved to Google Sheets successfully"}

app.include_router(router)
