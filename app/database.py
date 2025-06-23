from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import Optional
import gspread.utils
from oauth2client.service_account import ServiceAccountCredentials
import json
import os
import pandas as pd

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

worksheet_df  = pd.DataFrame(worksheet.get_all_records())
datasheet_df  = pd.DataFrame(datasheet.get_all_records())

last_row = worksheet_df.iloc[-1]

def calScore():
    score = 0
    output = []
    for i in range(0,10):
        id_val = last_row["ID"]
        is_duplicate = int(last_row["Is_Duplicated_ID"])
        if is_duplicate == 0:
            matched_df = datasheet_df[datasheet_df["ID"] == id_val]
        else:
            program_shared = last_row["Program_shared"]
            matched_df = datasheet_df[
                (datasheet_df["ID"] == id_val) & 
                (datasheet_df["Program_shared"] == program_shared)
            ]
        match_row = matched_df.iloc[0]
        if last_row["gpax"] < match_row["gpax_req"]:
            output.append("gpax score does not match with the requirement")
            continue 
        else:
            target_columns = [
        "tgat", "tpat3", "a_lv_61", "a_lv_64", "a_lv_65", "a_lv_63", "a_lv_62", "a_lv_82",
        "tpat4", "a_lv_81", "a_lv_86", "tgat2", "a_lv_89", "a_lv_88", "cal_subject_name",
        "a_lv_84", "gpax", "a_lv_87", "a_lv_85", "cal_score_sum", "a_lv_83", "tpat21",
        "a_lv_70", "a_lv_66", "tgat1", "tpat5", "cal_type", "vnet_51", "tgat3", "tpat22",
        "tpat2", "gpa28", "gpa22", "gpa23", "tu062", "ged_score", "tu002", "tu005", "tu006",
        "tu004", "tu071", "tu061", "tu072", "tu003", "tpat1", "tpat23", "gpa24", "gpa26",
        "gpa27", "su003", "su002", "su004", "su001", "priority_score", "gpa25", "gpa21",
        "tpat11", "tpat12", "tpat13"
        ]
            score_req = {col: match_row[col] for col in target_columns if pd.notnull(match_row.get(col))}
            # score_req.get("tpat2")
            if pd.notnull(score_req.get("cal_type")):

                subject_specs = str(match_row.get("cal_subject_name", "")).split()
                # Dictionary to keep track of heading → value
                score_dict = {}
                for spec in subject_specs:
                    if "|" in spec:
                        group = spec.split("|")
                        group_values = {
                            subj: row.get(subj)
                            for subj in group
                            if pd.notnull(row.get(subj))
                        }
                        if group_values:
                            # Get highest in this group
                            top_subject = max(group_values, key=group_values.get)
                            score_dict[top_subject] = group_values[top_subject]
                    else:
                        val = match_row.get(spec)
                        if pd.notnull(val):
                            score_dict[spec] = val
                if score_dict:
                    # Get the heading with the highest score
                    best_subject = max(score_dict, key=score_dict.get)
                    highest_score = score_dict[best_subject]
                    print(f"Highest Score: {highest_score} (from {best_subject})")
                else:
                    print("No valid scores found.")
                    output.append("Require more information")
                for col, datasheet_value in score_req.items():
                    worksheet_value = last_row.get(col)
                    
                    # Multiply only if worksheet value is also valid (not null)
                    if pd.notnull(worksheet_value):
                        product = float(worksheet_value) * (float(datasheet_value))/100
                        product_sum += product
                        print(f"{col}: {worksheet_value} × {datasheet_value} = {product}")
                score = product_sum + highest_score * (best_subject / 100)
                output.append(score)
                continue
            for col, datasheet_value in score_req.items():
                    worksheet_value = last_row.get(col)
                    
                    # Multiply only if worksheet value is also valid (not null)
                    if pd.notnull(worksheet_value):
                        product = float(worksheet_value) * (float(datasheet_value))/100
                        product_sum += product
                        print(f"{col}: {worksheet_value} × {datasheet_value} = {product}")
            output.append(product_sum)
            continue

        


    


# ---- FASTAPI MODEL ----
class ScoreSubmission(BaseModel):
    userId: Optional[str]
    name: Optional[str]
    gpax: Optional[float] = None
    tgat1: Optional[float] = None
    tgat2: Optional[float] = None
    tgat3: Optional[float] = None

    tpat11: Optional[float] = None
    tpat12: Optional[float] = None
    tpat13: Optional[float] = None
    tpat21: Optional[float] = None
    tpat22: Optional[float] = None
    tpat23: Optional[float] = None
    tpat3: Optional[float] = None
    tpat4: Optional[float] = None
    tpat5: Optional[float] = None

    a_lv_61: Optional[float] = None
    a_lv_62: Optional[float] = None
    a_lv_63: Optional[float] = None
    a_lv_64: Optional[float] = None
    a_lv_65: Optional[float] = None
    a_lv_66: Optional[float] = None
    a_lv_70: Optional[float] = None
    a_lv_81: Optional[float] = None
    a_lv_82: Optional[float] = None
    a_lv_83: Optional[float] = None
    a_lv_84: Optional[float] = None
    a_lv_85: Optional[float] = None
    a_lv_86: Optional[float] = None
    a_lv_87: Optional[float] = None
    a_lv_88: Optional[float] = None
    a_lv_89: Optional[float] = None

    gpa21: Optional[float] = None
    gpa22: Optional[float] = None
    gpa23: Optional[float] = None
    gpa24: Optional[float] = None
    gpa26: Optional[float] = None
    gpa27: Optional[float] = None
    gpa28: Optional[float] = None
  

@router.post("/api/find_faculty")
async def find_faculty(data: ScoreSubmission):
    values = datasheet.get_all_values()

    # Skip header row
    rowgits = values[1:]

    target_university = data.name.strip().lower()
    faculty_set = set()

    for row in rows:
        if len(row) >= 3:
            university_name = row[1].strip().lower()
            faculty_name = row[2].strip()

            if university_name == target_university and faculty_name:
                faculty_set.add(faculty_name)

    return {"faculties": sorted(faculty_set)}
@router.post("/api/find_field")
async def find_faculty(data: ScoreSubmission):
    values = datasheet.get_all_values()

    # Skip header row
    rows = values[1:]

    target_university = data.name.strip().lower()
    target_faculty = data.faculty.strip().lower()

    field_set = set()

    for row in rows:
        if len(row) >= 3:
            university_name = row[1].strip().lower()
            faculty_name = row[2].strip()
            field_name = row[3].strip
            if university_name == target_university and faculty_name == target_faculty and field_name:
                field_set.add(field_name)

    return {"faculties": sorted(field_set)}
@router.post("/api/save-score")
async def save_score(data: ScoreSubmission):
    print("Received data:", data)

    def upsert_user_data(worksheet, data):
        all_values = worksheet.get_all_values()

        columns = [
          "gpax","tgat1","tgat2","tgat3","tpat11","tpat12","tpat13","tpat21","tpat22","tpat23","tpat3","tpat4","tpat5","a_lv_61","a_lv_62","a_lv_63","a_lv_64","a_lv_65","a_lv_66","a_lv_70","a_lv_81","a_lv_82","a_lv_83","a_lv_84","a_lv_85","a_lv_86","a_lv_87","a_lv_88","a_lv_89","gpa21","gpa22","gpa23","gpa24","gpa26","gpa27","gpa28",

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
