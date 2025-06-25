from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import gspread.utils
from oauth2client.service_account import ServiceAccountCredentials
import json
import os
import pandas as pd
import asyncio
from datetime import datetime, timedelta
import threading
from functools import lru_cache


app = FastAPI()
router = APIRouter()

# ---- CONFIGURATION ----
class Config:
    CACHE_TTL_SECONDS = 300  # 5 minutes
    SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    SHEET_ID = "1IFoQ9PJoralucmufWa11IZ0Njcyq_-Z8NjLmtEySMdY"

# ---- COLUMN DEFINITIONS ----
USER_COLUMNS = [
    "userId", "name", "gpax", "tgat1", "tgat2", "tgat3", "tpat11", "tpat12", "tpat13",
    "tpat21", "tpat22", "tpat23", "tpat3", "tpat4", "tpat5",
    "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
    "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
    "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89", "gpa21", "gpa22", "gpa23", 
    "gpa24", "gpa25", "gpa26", "gpa27", "gpa28",
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

NUMERIC_COLUMNS = {
    "gpax", "tgat1", "tgat2", "tgat3", "tpat11", "tpat12", "tpat13",
    "tpat21", "tpat22", "tpat23", "tpat3", "tpat4", "tpat5",
    "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
    "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
    "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89", "gpa21", "gpa22", "gpa23", 
    "gpa24", "gpa25", "gpa26", "gpa27", "gpa28"
}

# Updated score columns based on the CSV structure
SCORE_COLUMNS = [
    "tgat", "tpat3", "a_lv_61", "a_lv_64", "a_lv_65", "a_lv_63", "a_lv_62", "a_lv_82",
    "tpat4", "a_lv_81", "a_lv_86", "tgat2", "a_lv_89", "a_lv_88",
    "a_lv_84", "gpax", "a_lv_87", "a_lv_85", "a_lv_83", "tpat21",
    "a_lv_70", "a_lv_66", "tgat1", "tpat5", "vnet_51", "tgat3", "tpat22",
    "tpat2", "gpa28", "gpa22", "gpa23", "tu062", "ged_score", "tu002", "tu005", "tu006",
    "tu004", "tu071", "tu061", "tu072", "tu003", "tpat1", "tpat23", "gpa24", "gpa26",
    "gpa27", "su003", "su002", "su004", "su001", "priority_score", "gpa25", "gpa21",
    "tpat11", "tpat12", "tpat13"
]

# ---- DATA CACHE CLASS ----
class DataCache:
    def __init__(self, ttl_seconds=Config.CACHE_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self.last_update = None
        self.user_data_df = None
        self.university_data_df = None
        self.user_data_dict = {}
        self.university_faculties = {}
        self.faculty_fields = {}
        self._lock = threading.Lock()
        
    def is_cache_valid(self):
        if self.last_update is None:
            return False
        return datetime.now() - self.last_update < timedelta(seconds=self.ttl_seconds)
    
    def invalidate_cache(self):
        with self._lock:
            self.last_update = None
            self.user_data_df = None
            self.university_data_df = None
            self.user_data_dict.clear()
            self.university_faculties.clear()
            self.faculty_fields.clear()

# Global cache instance
data_cache = DataCache()

# ---- GOOGLE SHEETS SETUP ----
def setup_google_sheets():
    """Initialize Google Sheets connection"""
    try:
        creds_json_str = os.getenv("GOOGLE_CREDS_JSON")
        if not creds_json_str:
            raise ValueError("GOOGLE_CREDS_JSON environment variable not set")
            
        creds_json = json.loads(creds_json_str)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, Config.SCOPE)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_key(Config.SHEET_ID)
        worksheet = sheet.get_worksheet(0)  # User data sheet
        datasheet = sheet.get_worksheet(1)  # University data sheet
        
        return worksheet, datasheet
    except Exception as e:
        print(f"Error setting up Google Sheets: {e}")
        raise

worksheet, datasheet = setup_google_sheets()

# ---- SCORE CALCULATION FUNCTIONS ----
def validate_user_scores(user_data: Dict, required_columns: List[str]) -> Dict[str, Any]:
    """Validate that user has required scores for calculation"""
    missing_scores = []
    available_scores = {}
    
    for col in required_columns:
        user_score = user_data.get(col)
        if user_score is None or pd.isna(user_score):
            missing_scores.append(col)
        else:
            available_scores[col] = float(user_score)
    
    return {
        "is_valid": len(missing_scores) == 0,
        "missing_scores": missing_scores,
        "available_scores": available_scores,
        "missing_count": len(missing_scores),
        "total_required": len(required_columns)
    }

def get_required_score_columns(program: pd.Series) -> List[str]:
    """Get list of score columns required for this program"""
    required_columns = []
    
    # Check special calculation columns
    cal_type = program.get("cal_type")
    cal_subject_name = program.get("cal_subject_name")
    
    if pd.notna(cal_type) and pd.notna(cal_subject_name):
        # Add special subject columns
        subject_specs = str(cal_subject_name).split()
        for spec in subject_specs:
            if "|" in spec:
                required_columns.extend(spec.split("|"))
            else:
                required_columns.append(spec)
    
    # Add regular score columns that have weights
    for col in SCORE_COLUMNS:
        if col not in ["cal_subject_name", "cal_type", "cal_score_sum"]:
            program_weight = program.get(col)
            if pd.notna(program_weight) and float(program_weight) > 0:
                required_columns.append(col)
    
    return list(set(required_columns))  # Remove duplicates

def calculate_program_score(user_data: Dict, program: pd.Series) -> Dict[str, Any]:
    """Calculate score for a program with detailed validation"""
    try:
        # Get required columns for this program
        required_columns = get_required_score_columns(program)
        
        # Validate user scores
        validation = validate_user_scores(user_data, required_columns)
        
        if not validation["is_valid"]:
            return {
                "success": False,
                "error": "missing_scores",
                "score": None,
                "missing_scores": validation["missing_scores"],
                "missing_count": validation["missing_count"],
                "total_required": validation["total_required"],
                "message": f"Missing {validation['missing_count']} required scores: {', '.join(validation['missing_scores'])}"
            }
        
        score = 0
        score_breakdown = []
        
        # Check if this program has special calculation type
        cal_type = program.get("cal_type")
        cal_subject_name = program.get("cal_subject_name")
        cal_score_sum = program.get("cal_score_sum")
        
        if pd.notna(cal_type) and pd.notna(cal_subject_name):
            # Special calculation with subject selection
            subject_specs = str(cal_subject_name).split()
            best_subject_score = 0
            best_subject = None
            
            # Find the highest score among available subjects
            for spec in subject_specs:
                if "|" in spec:
                    # Group of subjects - pick the highest
                    subjects = spec.split("|")
                    for subj in subjects:
                        user_score = user_data.get(subj)
                        if user_score is not None and user_score > best_subject_score:
                            best_subject_score = user_score
                            best_subject = subj
                else:
                    # Single subject
                    user_score = user_data.get(spec)
                    if user_score is not None and user_score > best_subject_score:
                        best_subject_score = user_score
                        best_subject = spec
            
            # Add best subject score with its weight
            if best_subject and pd.notna(cal_score_sum):
                weight = float(cal_score_sum) / 100
                contribution = best_subject_score * weight
                score += contribution
                score_breakdown.append({
                    "subject": best_subject,
                    "user_score": best_subject_score,
                    "weight": weight,
                    "contribution": contribution
                })
            
            # Add other weighted scores (excluding special subjects)
            for col in SCORE_COLUMNS:
                if (col not in ["cal_subject_name", "cal_type", "cal_score_sum"] and 
                    col != best_subject):
                    
                    program_weight = program.get(col)
                    user_score = user_data.get(col)
                    
                    if pd.notna(program_weight) and user_score is not None:
                        weight = float(program_weight) / 100
                        contribution = user_score * weight
                        score += contribution
                        score_breakdown.append({
                            "subject": col,
                            "user_score": user_score,
                            "weight": weight,
                            "contribution": contribution
                        })
        else:
            # Regular calculation
            for col in SCORE_COLUMNS:
                if col not in ["cal_subject_name", "cal_type", "cal_score_sum"]:
                    program_weight = program.get(col)
                    user_score = user_data.get(col)
                    
                    if pd.notna(program_weight) and user_score is not None:
                        weight = float(program_weight) / 100
                        contribution = user_score * weight
                        score += contribution
                        score_breakdown.append({
                            "subject": col,
                            "user_score": user_score,
                            "weight": weight,
                            "contribution": contribution
                        })
        
        return {
            "success": True,
            "score": round(score, 2),
            "score_breakdown": score_breakdown,
            "program_info": {
                "id": program.get("ID"),
                "name": program.get("Program"),
                "university": program.get("University"),
                "faculty": program.get("Faculty")
            },
            "message": f"Score calculated successfully: {round(score, 2)}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": "calculation_error",
            "score": None,
            "message": f"Error calculating score: {str(e)}"
        }

# ---- DATA LOADING FUNCTIONS ----
def load_and_cache_data():
    """Load data from Google Sheets and cache it"""
    with data_cache._lock:
        if data_cache.is_cache_valid():
            return
        
        try:
            print("Loading data from Google Sheets...")
            
            # Load user data
            worksheet_values = worksheet.get_all_values()
            
            if worksheet_values:
                # Ensure all rows have the same length
                max_cols = len(USER_COLUMNS)
                normalized_rows = []
                for row in worksheet_values:
                    if len(row) < max_cols:
                        row.extend([''] * (max_cols - len(row)))
                    normalized_rows.append(row[:max_cols])
                
                user_df = pd.DataFrame(normalized_rows, columns=USER_COLUMNS)
                
                # Convert numeric columns
                for col in NUMERIC_COLUMNS:
                    if col in user_df.columns:
                        user_df[col] = pd.to_numeric(user_df[col], errors='coerce')
                
                data_cache.user_data_df = user_df
                
                # Create user lookup dictionary
                data_cache.user_data_dict = {}
                for _, row in user_df.iterrows():
                    if row['userId']:
                        data_cache.user_data_dict[row['userId']] = row.to_dict()
            
            # Load university data
            university_records = datasheet.get_all_records()
            data_cache.university_data_df = pd.DataFrame(university_records)
            
            # Pre-compute mappings
            if not data_cache.university_data_df.empty:
                # Cache faculties by university
                faculty_groups = data_cache.university_data_df.groupby('University')['Faculty'].apply(
                    lambda x: x.dropna().unique().tolist()
                ).to_dict()
                data_cache.university_faculties = faculty_groups
                
                # Cache fields by university+faculty combination
                for (university, faculty), group in data_cache.university_data_df.groupby(['University', 'Faculty']):
                    key = f"{university}|{faculty}"
                    fields = []
                    
                    for _, row in group.iterrows():
                        program = row.get('Program', '')
                        program_shared = row.get('Program_shared', '')
                        
                        if pd.isna(program) or str(program).strip() == '':
                            continue
                            
                        if pd.notna(program_shared) and str(program_shared).strip() != '':
                            field_display = f"{program}:{program_shared}"
                        else:
                            field_display = str(program)
                        
                        if field_display not in fields:
                            fields.append(field_display)
                    
                    data_cache.faculty_fields[key] = fields
            
            data_cache.last_update = datetime.now()
            print("Data loaded successfully")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

# ---- HELPER FUNCTIONS ----
@lru_cache(maxsize=128)
def get_cached_faculties(university: str) -> List[str]:
    """Get faculties for a university with caching"""
    load_and_cache_data()
    return data_cache.university_faculties.get(university, [])

@lru_cache(maxsize=256)
def get_cached_fields(university: str, faculty: str) -> List[str]:
    """Get fields for university+faculty with caching"""
    load_and_cache_data()
    key = f"{university}|{faculty}"
    return data_cache.faculty_fields.get(key, [])

def get_user_data_fast(user_id: str) -> Optional[Dict]:
    """Fast user data lookup using cached dictionary"""
    load_and_cache_data()
    return data_cache.user_data_dict.get(user_id)

def find_program_fast(university: str, faculty: str, field: str) -> Optional[pd.Series]:
    """Fast program lookup using indexed DataFrame"""
    load_and_cache_data()
    
    if data_cache.university_data_df is None or data_cache.university_data_df.empty:
        return None
    
    # Extract the program name from field
    program_name = field.split(':')[0] if ':' in field else field
    
    # Use boolean indexing for faster lookup
    mask = (
        (data_cache.university_data_df["University"] == university) & 
        (data_cache.university_data_df["Faculty"] == faculty) & 
        (data_cache.university_data_df["Program"] == program_name)
    )
    
    matched_programs = data_cache.university_data_df[mask]
    
    if matched_programs.empty:
        return None
    
    return matched_programs.iloc[0]

def calculate_user_scores(user_id: str) -> Dict[str, Any]:
    """Calculate scores for all user selections"""
    user_data = get_user_data_fast(user_id)
    if not user_data:
        return {"error": "User not found", "user_id": user_id}
    
    results = []
    
    # Check each of the 10 selections
    for i in range(1, 11):
        university = user_data.get(f"selection_{i}_university")
        faculty = user_data.get(f"selection_{i}_faculty")
        field = user_data.get(f"selection_{i}_field")
        
        selection_result = {
            "selection_number": i,
            "university": university,
            "faculty": faculty,
            "field": field,
            "status": "incomplete",
            "score": None,
            "message": ""
        }
        
        if not (university and faculty and field):
            selection_result["message"] = "Selection incomplete - missing university, faculty, or field"
            results.append(selection_result)
            continue
        
        # Find program
        program = find_program_fast(university, faculty, field)
        
        if program is None:
            selection_result["status"] = "not_found"
            selection_result["message"] = "Program not found in database"
            results.append(selection_result)
            continue
        
        # Check GPAX requirement
        gpax_req = program.get("gpax_req")
        user_gpax = user_data.get("gpax", 0)
        
        if pd.notna(gpax_req) and user_gpax and user_gpax < gpax_req:
            selection_result["status"] = "gpax_insufficient"
            selection_result["message"] = f"GPAX requirement not met (required: {gpax_req}, current: {user_gpax})"
            selection_result["gpax_required"] = float(gpax_req)
            selection_result["gpax_current"] = float(user_gpax)
            results.append(selection_result)
            continue
        
        # Calculate score
        score_result = calculate_program_score(user_data, program)
        
        if score_result["success"]:
            selection_result["status"] = "calculated"
            selection_result["score"] = score_result["score"]
            selection_result["message"] = score_result["message"]
            selection_result["score_breakdown"] = score_result["score_breakdown"]
        else:
            selection_result["status"] = "error"
            selection_result["message"] = score_result["message"]
            if score_result["error"] == "missing_scores":
                selection_result["missing_scores"] = score_result["missing_scores"]
                selection_result["missing_count"] = score_result["missing_count"]
        
        results.append(selection_result)
    
    # Calculate summary
    calculated_scores = [r["score"] for r in results if r["score"] is not None]
    
    return {
        "user_id": user_id,
        "user_name": user_data.get("name"),
        "user_gpax": user_data.get("gpax"),
        "results": results,
        "summary": {
            "total_selections": len([r for r in results if r["university"]]),
            "calculated_scores": len(calculated_scores),
            "missing_scores": len([r for r in results if r["status"] == "error" and "missing_scores" in r]),
            "highest_score": max(calculated_scores) if calculated_scores else 0,
            "lowest_score": min(calculated_scores) if calculated_scores else 0,
            "average_score": sum(calculated_scores) / len(calculated_scores) if calculated_scores else 0
        }
    }

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
    gpa25: Optional[float] = None
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

# ---- DATA SAVING FUNCTIONS ----
def upsert_user_data_optimized(worksheet, data):
    """Optimized user data upsert with minimal sheet operations"""
    # Get current user data from cache first
    user_data = get_user_data_fast(data.userId)
    
    # Prepare new row data
    new_row = [""] * len(USER_COLUMNS)
    
    # Set provided data
    for i, col in enumerate(USER_COLUMNS):
        if col.startswith("selection_"):
            # Keep existing selection data
            if user_data and col in user_data:
                new_row[i] = str(user_data[col]) if user_data[col] is not None else ""
        else:
            val = getattr(data, col, None)
            new_row[i] = str(val) if val is not None else ""
    
    # Single sheet operation
    if user_data:
        # Update existing row - find row number
        all_values = worksheet.get_all_values()
        row_index = None
        for i, row in enumerate(all_values):
            if row and len(row) > 0 and row[0] == data.userId:
                row_index = i + 1
                break
        
        if row_index:
            end_col_letter = gspread.utils.rowcol_to_a1(1, len(USER_COLUMNS))[:-1]
            cell_range = f"A{row_index}:{end_col_letter}{row_index}"
            worksheet.update(cell_range, [new_row])
    else:
        # Add new row
        worksheet.append_row(new_row)
    
    # Invalidate cache to force refresh
    data_cache.invalidate_cache()

def save_multiple_selections(worksheet, user_id: str, name: str, selections: List[Dict]):
    """Save multiple university selections"""
    # Get existing user data
    user_data = get_user_data_fast(user_id)
    
    # Prepare new row data
    new_row = [""] * len(USER_COLUMNS)
    
    # Set basic user info
    new_row[0] = user_id
    new_row[1] = name
    
    # Copy existing score data if available
    if user_data:
        for i, col in enumerate(USER_COLUMNS[2:], 2):
            if col in NUMERIC_COLUMNS and col in user_data and user_data[col] is not None:
                new_row[i] = str(user_data[col])

    # Add selections (find the index where selections start)
    selection_start_idx = USER_COLUMNS.index("selection_1_university")
    
    for idx, selection in enumerate(selections):
        if idx < 10:  # Max 10 selections
            base_idx = selection_start_idx + (idx * 3)
            new_row[base_idx] = selection.get("university", "")
            new_row[base_idx + 1] = selection.get("faculty", "")
            new_row[base_idx + 2] = selection.get("field", "")

    # Single sheet operation
    if user_data:
        # Update existing row
        all_values = worksheet.get_all_values()
        row_index = None
        for i, row in enumerate(all_values):
            if row and len(row) > 0 and row[0] == user_id:
                row_index = i + 1
                break
        
        if row_index:
            end_col_letter = gspread.utils.rowcol_to_a1(1, len(USER_COLUMNS))[:-1]
            cell_range = f"A{row_index}:{end_col_letter}{row_index}"
            worksheet.update(cell_range, [new_row])
    else:
        worksheet.append_row(new_row)

    # Invalidate cache
    data_cache.invalidate_cache()

# ---- API ENDPOINTS ----
@router.post("/api/find_faculty")
async def find_faculty(data: UniversityRequest):
    """Get faculties for a university"""
    try:
        faculties = get_cached_faculties(data.name)
        return {"faculties": faculties}
    except Exception as e:
        print(f"Error finding faculties: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find faculties: {str(e)}")

@router.post("/api/find_field")
async def find_field(data: FacultyRequest):
    """Get fields for a university and faculty"""
    try:
        fields = get_cached_fields(data.name, data.faculty)
        return {"faculties": fields}  # Keep original response format for compatibility
    except Exception as e:
        print(f"Error finding fields: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find fields: {str(e)}")

@router.post("/api/calculate_scores")
async def calculate_scores(data: dict):
    """Calculate scores for all user selections"""
    user_id = data.get("userId")
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    
    try:
        results = calculate_user_scores(user_id)
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate scores: {str(e)}")

@router.get("/api/user_data/{user_id}")
async def get_user_data(user_id: str):
    """Get user data"""
    try:
        user_data = get_user_data_fast(user_id)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"user_data": user_data}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user data: {str(e)}")

@router.get("/api/debug/user/{user_id}")
async def debug_user_data(user_id: str):
    """Debug endpoint to check user data"""
    try:
        user_data = get_user_data_fast(user_id)
        if not user_data:
            return {"error": "User not found", "user_id": user_id}
        
        # Check selections
        selections = []
        for i in range(1, 11):
            university = user_data.get(f"selection_{i}_university")
            faculty = user_data.get(f"selection_{i}_faculty")
            field = user_data.get(f"selection_{i}_field")
            
            if university or faculty or field:
                selections.append({
                    "selection": i,
                    "university": university,
                    "faculty": faculty,
                    "field": field,
                    "complete": bool(university and faculty and field)
                })
        
        # Check available scores
        available_scores = {}
        missing_scores = []
        for col in NUMERIC_COLUMNS:
            value = user_data.get(col)
            if value is not None and not pd.isna(value):
                available_scores[col] = value
            else:
                missing_scores.append(col)
        
        return {
            "user_id": user_id,
            "user_name": user_data.get("name"),
            "user_gpax": user_data.get("gpax"),
            "selections": selections,
            "total_selections": len(selections),
            "available_scores": available_scores,
            "missing_scores": missing_scores,
            "score_completeness": f"{len(available_scores)}/{len(NUMERIC_COLUMNS)}",
            "cache_status": {
                "cache_valid": data_cache.is_cache_valid(),
                "last_update": data_cache.last_update.isoformat() if data_cache.last_update else None,
                "user_data_loaded": data_cache.user_data_df is not None,
                "university_data_loaded": data_cache.university_data_df is not None
            }
        }
    except Exception as e:
        return {"error": str(e), "user_id": user_id}

@router.post("/api/save-score")
async def save_score(data: ScoreSubmission):
    """Save user score data"""
    try:
        if not data.userId:
            raise HTTPException(status_code=400, detail="userId is required")
        
        upsert_user_data_optimized(worksheet, data)
        return {"message": "Data saved to Google Sheets successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

@router.post("/api/submit_multiple_selections")
async def submit_multiple_selections(data: MultipleSelectionsSubmission):
    """Save multiple university selections"""
    try:
        if not data.userId:
            raise HTTPException(status_code=400, detail="userId is required")
        
        if not data.name:
            raise HTTPException(status_code=400, detail="name is required")
        
        if not data.selections:
            raise HTTPException(status_code=400, detail="selections are required")
        
        # Convert selections to dict format
        selections_dict = []
        for selection in data.selections:
            selections_dict.append({
                "university": selection.university,
                "faculty": selection.faculty,
                "field": selection.field
            })
        
        save_multiple_selections(worksheet, data.userId, data.name, selections_dict)
        
        return {"message": f"Successfully saved {len(data.selections)} university selections"}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error saving multiple selections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save selections: {str(e)}")

@router.post("/api/calculate_program_score")
async def calculate_program_score_endpoint(data: dict):
    """Calculate score for a specific program"""
    user_id = data.get("userId")
    university = data.get("university")
    faculty = data.get("faculty")
    field = data.get("field")
    
    if not all([user_id, university, faculty, field]):
        raise HTTPException(
            status_code=400, 
            detail="userId, university, faculty, and field are required"
        )
    
    try:
        user_data = get_user_data_fast(user_id)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        program = find_program_fast(university, faculty, field)
        if program is None:
            raise HTTPException(status_code=404, detail="Program not found")
        
        # Check GPAX requirement
        gpax_req = program.get("gpax_req")
        user_gpax = user_data.get("gpax", 0)
        
        if pd.notna(gpax_req) and user_gpax and user_gpax < gpax_req:
            return {
                "status": "gpax_insufficient",
                "message": f"GPAX too low (need {gpax_req}, have {user_gpax})",
                "score": None,
                "gpax_required": float(gpax_req),
                "gpax_current": float(user_gpax)
            }
        
        # Calculate score
        score_result = calculate_program_score(user_data, program)
        
        if score_result["success"]:
            return {
                "status": "success",
                "score": score_result["score"],
                "program_info": score_result["program_info"],
                "score_breakdown": score_result["score_breakdown"],
                "university": university,
                "faculty": faculty,
                "field": field,
                "message": score_result["message"]
            }
        else:
            return {
                "status": "error",
                "score": None,
                "message": score_result["message"],
                "missing_scores": score_result.get("missing_scores", [])
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating program score: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate score: {str(e)}")

@router.get("/api/user_scores/{user_id}")
async def get_user_scores(user_id: str):
    """Get calculated scores for a user"""
    try:
        results = calculate_user_scores(user_id)
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        return results
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting user scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scores: {str(e)}")

@router.post("/api/refresh_cache")
async def refresh_cache():
    """Manually refresh the data cache"""
    try:
        data_cache.invalidate_cache()
        load_and_cache_data()
        return {"message": "Cache refreshed successfully"}
    except Exception as e:
        print(f"Error refreshing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")

# Include router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
