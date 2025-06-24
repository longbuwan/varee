from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
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
worksheet = sheet.get_worksheet(0)  # User data sheet (no headers)
datasheet = sheet.get_worksheet(1)  # University data sheet (has headers)

def get_fresh_data():
    """Get fresh data from sheets to avoid stale data issues"""
    # For worksheet (user data) - no headers, so we need to handle differently
    worksheet_values = worksheet.get_all_values()
    
    # For datasheet (university data) - has headers
    datasheet_df = pd.DataFrame(datasheet.get_all_records())
    
    return worksheet_values, datasheet_df

def find_user_data(user_id):
    """Find user data by userId from worksheet"""
    worksheet_values, _ = get_fresh_data()
    
    # Define the expected column structure for user data
    user_columns = [
        "userId", "name", "gpax", "tgat1", "tgat2", "tgat3", "tpat11", "tpat12", "tpat13",
        "tpat21", "tpat22", "tpat23", "tpat3", "tpat4", "tpat5",
        "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
        "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
        "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89",
        "gpa21", "gpa22", "gpa23", "gpa24", "gpa26", "gpa27", "gpa28",
        # Selection columns for 10 universities
        "selection_1_university", "selection_1_faculty", "selection_1_field",
        "selection_2_university", "selection_2_faculty", "selection_2_field",
        "selection_3_university", "selection_3_faculty", "selection_3_field",
        "selection_4_university", "selection_4_faculty", "selection_4_field",
        "selection_5_university", "selection_5_faculty", "selection_5_field",
        "selection_6_university", "selection_6_faculty", "selection_6_field",
        "selection_7_university", "selection_7_faculty", "selection_7_field",
        "selection_8_university", "selection_8_faculty", "selection_8_field",
        "selection_9_university", "selection_9_faculty", "selection_9_field",
        "selection_10_university", "selection_10_faculty", "selection_10_field"
    ]
    
    # Find user row
    for row in worksheet_values:
        if row and len(row) > 0 and row[0] == user_id:
            # Convert row to dictionary
            user_data = {}
            for i, col in enumerate(user_columns):
                if i < len(row):
                    value = row[i]
                    # Convert to appropriate type
                    if col in ["gpax", "tgat1", "tgat2", "tgat3", "tpat11", "tpat12", "tpat13",
                              "tpat21", "tpat22", "tpat23", "tpat3", "tpat4", "tpat5",
                              "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
                              "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
                              "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89",
                              "gpa21", "gpa22", "gpa23", "gpa24", "gpa26", "gpa27", "gpa28"]:
                        try:
                            user_data[col] = float(value) if value else None
                        except ValueError:
                            user_data[col] = None
                    else:
                        user_data[col] = value if value else None
                else:
                    user_data[col] = None
            return user_data
    
    return None

def calScore(user_id):
    """Calculate scores for a specific user's university selections"""
    user_data = find_user_data(user_id)
    if not user_data:
        return {"error": "User not found"}
    
    _, datasheet_df = get_fresh_data()
    results = []
    
    # Check each of the 10 selections
    for i in range(1, 11):
        selection_result = {
            "selection_number": i,
            "university": user_data.get(f"selection_{i}_university"),
            "faculty": user_data.get(f"selection_{i}_faculty"),
            "field": user_data.get(f"selection_{i}_field"),
            "score": None,
            "status": "incomplete",
            "message": ""
        }
        
        university = selection_result["university"]
        faculty = selection_result["faculty"]
        field = selection_result["field"]
        
        if not (university and faculty and field):
            selection_result["message"] = "Selection incomplete"
            results.append(selection_result)
            continue
        
        # Find matching program in datasheet
        matched_programs = datasheet_df[
            (datasheet_df["University"] == university) & 
            (datasheet_df["Faculty"] == faculty) & 
            (datasheet_df["Program"] == field)
        ]
        
        if matched_programs.empty:
            selection_result["message"] = "No matching program found"
            selection_result["status"] = "not_found"
            results.append(selection_result)
            continue
        
        # Get the first matching program
        program = matched_programs.iloc[0]
        
        # Check GPAX requirement
        gpax_req = program.get("gpax_req")
        user_gpax = user_data.get("gpax", 0)
        
        if pd.notna(gpax_req) and user_gpax and user_gpax < gpax_req:
            selection_result["message"] = f"GPAX requirement not met (required: {gpax_req}, have: {user_gpax})"
            selection_result["status"] = "gpax_insufficient"
            results.append(selection_result)
            continue
        
        # Calculate score
        try:
            score = calculate_program_score(user_data, program)
            selection_result["score"] = score
            selection_result["status"] = "calculated"
            selection_result["message"] = f"Score calculated successfully: {score:.2f}"
        except Exception as e:
            selection_result["message"] = f"Error calculating score: {str(e)}"
            selection_result["status"] = "error"
        
        results.append(selection_result)
    
    return {"user_id": user_id, "results": results}

def calculate_program_score(user_data, program):
    """Calculate score for a specific program"""
    score = 0
    calculation_details = []
    
    # Define all possible score columns
    score_columns = [
        "tgat", "tpat3", "a_lv_61", "a_lv_64", "a_lv_65", "a_lv_63", "a_lv_62", "a_lv_82",
        "tpat4", "a_lv_81", "a_lv_86", "tgat2", "a_lv_89", "a_lv_88",
        "a_lv_84", "gpax", "a_lv_87", "a_lv_85", "a_lv_83", "tpat21",
        "a_lv_70", "a_lv_66", "tgat1", "tpat5", "vnet_51", "tgat3", "tpat22",
        "tpat2", "gpa28", "gpa22", "gpa23", "tu062", "ged_score", "tu002", "tu005", "tu006",
        "tu004", "tu071", "tu061", "tu072", "tu003", "tpat1", "tpat23", "gpa24", "gpa26",
        "gpa27", "su003", "su002", "su004", "su001", "priority_score", "gpa25", "gpa21",
        "tpat11", "tpat12", "tpat13"
    ]
    
    # Check if this program has special calculation type
    cal_type = program.get("cal_type")
    cal_subject_name = program.get("cal_subject_name")
    
    if pd.notna(cal_type) and pd.notna(cal_subject_name):
        # Special calculation with subject selection
        subject_specs = str(cal_subject_name).split()
        best_subject_score = 0
        best_subject = None
        
        for spec in subject_specs:
            if "|" in spec:
                # Group of subjects - pick the highest
                subjects = spec.split("|")
                group_scores = {}
                for subj in subjects:
                    user_score = user_data.get(subj)
                    if user_score is not None:
                        group_scores[subj] = float(user_score)
                
                if group_scores:
                    best_subj = max(group_scores, key=group_scores.get)
                    if group_scores[best_subj] > best_subject_score:
                        best_subject_score = group_scores[best_subj]
                        best_subject = best_subj
            else:
                # Single subject
                user_score = user_data.get(spec)
                if user_score is not None and float(user_score) > best_subject_score:
                    best_subject_score = float(user_score)
                    best_subject = spec
        
        # Add best subject score with its weight
        if best_subject:
            weight = program.get(best_subject)
            if pd.notna(weight):
                score += best_subject_score * (float(weight) / 100)
                calculation_details.append(f"{best_subject}: {best_subject_score} × {weight}% = {best_subject_score * (float(weight) / 100)}")
        
        # Add other weighted scores (excluding special subjects and calculation columns)
        for col in score_columns:
            if col not in ["cal_subject_name", "cal_type", "cal_score_sum"] and col != best_subject:
                program_weight = program.get(col)
                user_score = user_data.get(col)
                
                if pd.notna(program_weight) and user_score is not None:
                    weighted_score = float(user_score) * (float(program_weight) / 100)
                    score += weighted_score
                    calculation_details.append(f"{col}: {user_score} × {program_weight}% = {weighted_score}")
    
    else:
        # Regular calculation - sum all weighted scores
        for col in score_columns:
            if col not in ["cal_subject_name", "cal_type", "cal_score_sum"]:
                program_weight = program.get(col)
                user_score = user_data.get(col)
                
                if pd.notna(program_weight) and user_score is not None:
                    weighted_score = float(user_score) * (float(program_weight) / 100)
                    score += weighted_score
                    calculation_details.append(f"{col}: {user_score} × {program_weight}% = {weighted_score}")
    
    return round(score, 2)

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

class UniversitySelection(BaseModel):
    university: str
    faculty: str
    field: str

class MultipleSelectionsSubmission(BaseModel):
    userId: str
    name: str
    selections: List[UniversitySelection]

# ---- UNIVERSITY DATA ENDPOINTS (datasheet) ----
@router.post("/api/find_faculty")
async def find_faculty(data: UniversityRequest):
    """Get faculties for a university from datasheet"""
    try:
        _, datasheet_df = get_fresh_data()
        
        # Filter by university name
        university_data = datasheet_df[datasheet_df["University"] == data.name]
        
        if university_data.empty:
            return {"faculties": []}
        
        # Get unique faculties
        faculties = university_data["Faculty"].dropna().unique().tolist()
        
        return {"faculties": faculties}
    except Exception as e:
        print(f"Error finding faculties: {e}")
        return {"error": f"Failed to find faculties: {str(e)}"}

@router.post("/api/find_field")
async def find_field(data: FacultyRequest):
    """Get fields for a university and faculty from datasheet"""
    try:
        _, datasheet_df = get_fresh_data()
        
        # Filter by university name and faculty
        field_data = datasheet_df[
            (datasheet_df["University"] == data.name) & 
            (datasheet_df["Faculty"] == data.faculty)
        ]
        
        if field_data.empty:
            return {"faculties": []}  # Keep original response format
        
        # Get unique fields/programs
        fields = field_data["Program"].dropna().unique().tolist()
        
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
        
        # Define the complete column structure (no headers in worksheet)
        columns = [
            "userId", "name", "gpax", "tgat1", "tgat2", "tgat3", "tpat11", "tpat12", "tpat13",
            "tpat21", "tpat22", "tpat23", "tpat3", "tpat4", "tpat5",
            "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
            "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
            "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89",
            "gpa21", "gpa22", "gpa23", "gpa24", "gpa26", "gpa27", "gpa28",
            # Selection columns for 10 universities
            "selection_1_university", "selection_1_faculty", "selection_1_field",
            "selection_2_university", "selection_2_faculty", "selection_2_field",
            "selection_3_university", "selection_3_faculty", "selection_3_field",
            "selection_4_university", "selection_4_faculty", "selection_4_field",
            "selection_5_university", "selection_5_faculty", "selection_5_field",
            "selection_6_university", "selection_6_faculty", "selection_6_field",
            "selection_7_university", "selection_7_faculty", "selection_7_field",
            "selection_8_university", "selection_8_faculty", "selection_8_field",
            "selection_9_university", "selection_9_faculty", "selection_9_field",
            "selection_10_university", "selection_10_faculty", "selection_10_field"
        ]

        # Find existing row by userId
        row_index = None
        for i, row in enumerate(all_values):
            if row and len(row) > 0 and row[0] == data.userId:
                row_index = i + 1  # gspread uses 1-based indexing
                break

        # Prepare new row data
        new_row = []
        for col in columns:
            if col.startswith("selection_"):
                # Keep existing selection data if it exists
                if row_index and len(all_values) > row_index - 1:
                    existing_row = all_values[row_index - 1]
                    col_index = columns.index(col)
                    if col_index < len(existing_row):
                        new_row.append(existing_row[col_index])
                    else:
                        new_row.append("")
                else:
                    new_row.append("")
            else:
                val = getattr(data, col, None)
                new_row.append(str(val) if val is not None else "")

        if row_index:
            # Update existing row
            print(f"Updating existing user at row {row_index}")
            # Ensure we have enough columns
            while len(new_row) < len(columns):
                new_row.append("")
            
            # Update the row
            end_col_letter = gspread.utils.rowcol_to_a1(1, len(columns))[:-1]
            cell_range = f"A{row_index}:{end_col_letter}{row_index}"
            worksheet.update(cell_range, [new_row])
        else:
            # Add new row
            print("Adding new user row")
            # Ensure we have the right number of columns
            while len(new_row) < len(columns):
                new_row.append("")
            worksheet.append_row(new_row)

    try:
        upsert_user_data(worksheet, data)
        return {"message": "Data saved to Google Sheets successfully"}
    except Exception as e:
        print(f"Error saving data: {e}")
        return {"error": f"Failed to save data: {str(e)}"}

# ---- MULTIPLE SELECTIONS ENDPOINT ----
@router.post("/api/submit_multiple_selections")
async def submit_multiple_selections(data: MultipleSelectionsSubmission):
    """Save multiple university selections for a user"""
    print("Received multiple selections:", data)

    try:
        # Get existing user data
        all_values = worksheet.get_all_values()
        
        # Find existing row by userId
        row_index = None
        existing_row = None
        for i, row in enumerate(all_values):
            if row and len(row) > 0 and row[0] == data.userId:
                row_index = i + 1  # gspread uses 1-based indexing
                existing_row = row
                break

        # Define the complete column structure
        columns = [
            "userId", "name", "gpax", "tgat1", "tgat2", "tgat3", "tpat11", "tpat12", "tpat13",
            "tpat21", "tpat22", "tpat23", "tpat3", "tpat4", "tpat5",
            "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
            "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
            "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89",
            "gpa21", "gpa22", "gpa23", "gpa24", "gpa26", "gpa27", "gpa28",
            # Selection columns for 10 universities
            "selection_1_university", "selection_1_faculty", "selection_1_field",
            "selection_2_university", "selection_2_faculty", "selection_2_field",
            "selection_3_university", "selection_3_faculty", "selection_3_field",
            "selection_4_university", "selection_4_faculty", "selection_4_field",
            "selection_5_university", "selection_5_faculty", "selection_5_field",
            "selection_6_university", "selection_6_faculty", "selection_6_field",
            "selection_7_university", "selection_7_faculty", "selection_7_field",
            "selection_8_university", "selection_8_faculty", "selection_8_field",
            "selection_9_university", "selection_9_faculty", "selection_9_field",
            "selection_10_university", "selection_10_faculty", "selection_10_field"
        ]

        # Prepare new row data
        new_row = [""] * len(columns)
        
        # Set basic user info
        new_row[0] = data.userId  # userId
        new_row[1] = data.name    # name
        
        # Copy existing score data if available
        if existing_row:
            for i in range(2, 37):  # Score columns (gpax through gpa28)
                if i < len(existing_row):
                    new_row[i] = existing_row[i]

        # Add selections
        for idx, selection in enumerate(data.selections):
            if idx < 10:  # Only handle first 10 selections
                base_idx = 37 + (idx * 3)  # Starting index for selection columns
                new_row[base_idx] = selection.university
                new_row[base_idx + 1] = selection.faculty
                new_row[base_idx + 2] = selection.field

        if row_index:
            # Update existing row
            print(f"Updating existing user selections at row {row_index}")
            end_col_letter = gspread.utils.rowcol_to_a1(1, len(columns))[:-1]
            cell_range = f"A{row_index}:{end_col_letter}{row_index}"
            worksheet.update(cell_range, [new_row])
        else:
            # Add new row
            print("Adding new user with selections")
            worksheet.append_row(new_row)

        return {"message": f"Successfully saved {len(data.selections)} university selections"}

    except Exception as e:
        print(f"Error saving multiple selections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save selections: {str(e)}")

# ---- SCORE CALCULATION ENDPOINT ----
@router.post("/api/calculate_scores")
async def calculate_scores(data: dict):
    """Calculate scores for a user's university selections"""
    user_id = data.get("userId")
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    
    try:
        results = calScore(user_id)
        return results
    except Exception as e:
        print(f"Error calculating scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate scores: {str(e)}")

# ---- GET USER DATA ENDPOINT ----
@router.get("/api/user_data/{user_id}")
async def get_user_data(user_id: str):
    """Get user data including scores and selections"""
    try:
        user_data = find_user_data(user_id)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"user_data": user_data}
    except Exception as e:
        print(f"Error getting user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user data: {str(e)}")

app.include_router(router)
