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
worksheet = sheet.get_worksheet(0)  # User data sheet
datasheet = sheet.get_worksheet(1)  # University data sheet

def get_fresh_data():
    """Get fresh data from sheets to avoid stale data issues"""
    worksheet_df = pd.DataFrame(worksheet.get_all_records())
    datasheet_df = pd.DataFrame(datasheet.get_all_records())
    return worksheet_df, datasheet_df

def calScore(user_id):
    """Calculate score for a specific user"""
    worksheet_df, datasheet_df = get_fresh_data()
    
    # Find the user's row
    user_rows = worksheet_df[worksheet_df["userId"] == user_id]
    if user_rows.empty:
        return ["User not found"]
    
    last_row = user_rows.iloc[-1]
    
    score = 0
    output = []
    product_sum = 0  # Initialize product_sum
    
    for i in range(0, 10):
        id_val = last_row.get("ID")
        if pd.isna(id_val):
            output.append("No ID found for this iteration")
            continue
            
        is_duplicate = int(last_row.get("Is_Duplicated_ID", 0))
        
        if is_duplicate == 0:
            matched_df = datasheet_df[datasheet_df["ID"] == id_val]
        else:
            program_shared = last_row.get("Program_shared")
            matched_df = datasheet_df[
                (datasheet_df["ID"] == id_val) & 
                (datasheet_df["Program_shared"] == program_shared)
            ]
        
        if matched_df.empty:
            output.append("No matching program found")
            continue
            
        match_row = matched_df.iloc[0]
        
        # Check GPAX requirement
        if pd.notna(match_row.get("gpax_req")) and last_row.get("gpax", 0) < match_row["gpax_req"]:
            output.append("gpax score does not match with the requirement")
            continue
        
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
        
        score_req = {col: match_row[col] for col in target_columns if pd.notna(match_row.get(col))}
        product_sum = 0  # Reset for each iteration
        
        if pd.notna(score_req.get("cal_type")):
            subject_specs = str(match_row.get("cal_subject_name", "")).split()
            score_dict = {}
            
            for spec in subject_specs:
                if "|" in spec:
                    group = spec.split("|")
                    group_values = {
                        subj: last_row.get(subj)  # Use last_row instead of undefined 'row'
                        for subj in group
                        if pd.notna(last_row.get(subj))
                    }
                    if group_values:
                        top_subject = max(group_values, key=group_values.get)
                        score_dict[top_subject] = group_values[top_subject]
                else:
                    val = last_row.get(spec)  # Use last_row instead of match_row
                    if pd.notna(val):
                        score_dict[spec] = val
            
            if score_dict:
                best_subject = max(score_dict, key=score_dict.get)
                highest_score = score_dict[best_subject]
                print(f"Highest Score: {highest_score} (from {best_subject})")
                
                # Calculate product sum for other columns
                for col, datasheet_value in score_req.items():
                    if col != "cal_subject_name" and col != "cal_type":
                        worksheet_value = last_row.get(col)
                        if pd.notna(worksheet_value) and pd.notna(datasheet_value):
                            product = float(worksheet_value) * (float(datasheet_value))/100
                            product_sum += product
                            print(f"{col}: {worksheet_value} × {datasheet_value} = {product}")
                
                # Fix the calculation - get the weight for best_subject
                best_subject_weight = score_req.get(best_subject, 0)
                if pd.notna(best_subject_weight):
                    score = product_sum + highest_score * (float(best_subject_weight) / 100)
                else:
                    score = product_sum
                output.append(score)
            else:
                output.append("Require more information")
        else:
            # Regular calculation without special subject handling
            for col, datasheet_value in score_req.items():
                worksheet_value = last_row.get(col)
                if pd.notna(worksheet_value) and pd.notna(datasheet_value):
                    product = float(worksheet_value) * (float(datasheet_value))/100
                    product_sum += product
                    print(f"{col}: {worksheet_value} × {datasheet_value} = {product}")
            output.append(product_sum)
    
    return output

# ---- PYDANTIC MODELS ----
class UniversityRequest(BaseModel):
    name: str
    userId: Optional[str] = None

class FacultyRequest(BaseModel):
    name: str
    faculty: str

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

# ---- UNIVERSITY DATA ENDPOINTS (datasheet) ----
@router.post("/api/find_faculty")
async def find_faculty(data: UniversityRequest):
    """Get faculties for a university from datasheet"""
    try:
        datasheet_df = pd.DataFrame(datasheet.get_all_records())
        
        # Filter by university name
        university_data = datasheet_df[datasheet_df["university_name"] == data.name]
        
        if university_data.empty:
            return {"faculties": []}
        
        # Get unique faculties
        faculties = university_data["faculty_name"].dropna().unique().tolist()
        
        return {"faculties": faculties}
    except Exception as e:
        print(f"Error finding faculties: {e}")
        return {"error": f"Failed to find faculties: {str(e)}"}

@router.post("/api/find_field")
async def find_field(data: FacultyRequest):
    """Get fields for a university and faculty from datasheet"""
    try:
        datasheet_df = pd.DataFrame(datasheet.get_all_records())
        
        # Filter by university name and faculty
        field_data = datasheet_df[
            (datasheet_df["university_name"] == data.name) & 
            (datasheet_df["faculty_name"] == data.faculty)
        ]
        
        if field_data.empty:
            return {"faculties": []}  # Keep original response format
        
        # Get unique fields/programs
        fields = field_data["program_name"].dropna().unique().tolist()
        
        return {"faculties": fields}  # Keep original response format
    except Exception as e:
        print(f"Error finding fields: {e}")
        return {"error": f"Failed to find fields: {str(e)}"}

# ---- USER SCORE ENDPOINT (worksheet) ----
@router.post("/api/save-score")
async def save_score(data: ScoreSubmission):
    """Save user score data to worksheet"""
    print("Received data:", data)

    def upsert_user_data(worksheet, data):
        # Get all current data
        all_values = worksheet.get_all_values()
        
        if not all_values:
            # If sheet is empty, add headers first
            headers = [
                "userId", "name", "gpax", "tgat1", "tgat2", "tgat3", "tpat11", "tpat12", "tpat13",
                "tpat21", "tpat22", "tpat23", "tpat3", "tpat4", "tpat5",
                "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
                "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
                "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89",
                "gpa21", "gpa22", "gpa23", "gpa24", "gpa26", "gpa27", "gpa28"
            ]
            worksheet.append_row(headers)
            all_values = [headers]

        columns = [
            "userId", "name", "gpax", "tgat1", "tgat2", "tgat3", "tpat11", "tpat12", "tpat13",
            "tpat21", "tpat22", "tpat23", "tpat3", "tpat4", "tpat5",
            "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
            "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
            "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89",
            "gpa21", "gpa22", "gpa23", "gpa24", "gpa26", "gpa27", "gpa28"
        ]

        # Find existing row by userId (skip header row)
        row_index = None
        for i, row in enumerate(all_values[1:], start=2):  # Start from row 2 (skip header)
            if row and len(row) > 0 and row[0] == data.userId:
                row_index = i
                break

        # Prepare new row data - IMPORTANT: Convert None to empty string
        new_row = []
        for col in columns:
            val = getattr(data, col, None)
            # Convert None to empty string to overwrite existing data
            new_row.append(str(val) if val is not None else "")

        if row_index:
            # Update existing row - this will overwrite ALL values including blanks
            print(f"Updating existing user at row {row_index}")
            # Get the range for the entire row
            end_col = gspread.utils.rowcol_to_a1(row_index, len(columns))[:-1]  # Remove row number
            cell_range = f"A{row_index}:{end_col}{row_index}"
            worksheet.update(cell_range, [new_row])
        else:
            # Add new row
            print("Adding new user row")
            worksheet.append_row(new_row)

    try:
        upsert_user_data(worksheet, data)
        return {"message": "Data saved to Google Sheets successfully"}
    except Exception as e:
        print(f"Error saving data: {e}")
        return {"error": f"Failed to save data: {str(e)}"}

app.include_router(router)
