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

# ---- CACHING AND DATA MANAGEMENT ----
class DataCache:
    def __init__(self, ttl_seconds=300):  # 5 minutes cache
        self.ttl_seconds = ttl_seconds
        self.last_update = None
        self.user_data_df = None
        self.university_data_df = None
        self.user_data_dict = {}  # For O(1) user lookups
        self.university_faculties = {}  # Cached faculty data
        self.faculty_fields = {}  # Cached field data
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


def calculate_program_score_improved(user_data: Dict, program: pd.Series) -> Dict:
    """
    Improved score calculation with proper cal_subject_name handling
    and detailed university information
    """
    score = 0
    calculation_details = []
    best_subject_info = None
    
    # Check if this program has special calculation type with cal_subject_name
    cal_type = program.get("cal_type")
    cal_subject_name = program.get("cal_subject_name")
    cal_score_sum = program.get("cal_score_sum")
    
    if pd.notna(cal_type) and pd.notna(cal_subject_name):
        # Special calculation: pick ONE highest score from cal_subject_name subjects
        subject_specs = str(cal_subject_name).split()
        best_subject_score = 0
        best_subject = None
        available_subjects = {}
        
        # Collect all available subject scores
        for spec in subject_specs:
            if "|" in spec:
                # Group of subjects - evaluate each in the group
                subjects = spec.split("|")
                for subj in subjects:
                    user_score = user_data.get(subj)
                    if user_score is not None:
                        available_subjects[subj] = user_score
            else:
                # Single subject
                user_score = user_data.get(spec)
                if user_score is not None:
                    available_subjects[spec] = user_score
        
        # Find the highest score among available subjects
        if available_subjects:
            best_subject = max(available_subjects, key=available_subjects.get)
            best_subject_score = available_subjects[best_subject]
            
            # Apply the weight from cal_score_sum
            if pd.notna(cal_score_sum):
                weighted_score = best_subject_score * (float(cal_score_sum) / 100)
                score += weighted_score
                
                best_subject_info = {
                    "subject": best_subject,
                    "raw_score": best_subject_score,
                    "weight_percentage": float(cal_score_sum),
                    "weighted_score": weighted_score,
                    "available_subjects": available_subjects
                }
                
                calculation_details.append({
                    "type": "cal_subject_selection",
                    "subject": best_subject,
                    "score": best_subject_score,
                    "weight": float(cal_score_sum),
                    "contribution": weighted_score
                })
        
        # Add regular weighted scores for other columns (excluding the selected subject)
        for col in SCORE_COLUMNS:
            if (col not in ["cal_subject_name", "cal_type", "cal_score_sum"] and 
                col != best_subject and col not in subject_specs):
                
                program_weight = program.get(col)
                user_score = user_data.get(col)
                
                if pd.notna(program_weight) and user_score is not None:
                    contribution = user_score * (float(program_weight) / 100)
                    score += contribution
                    
                    calculation_details.append({
                        "type": "regular",
                        "subject": col,
                        "score": user_score,
                        "weight": float(program_weight),
                        "contribution": contribution
                    })
    else:
        # Regular calculation - no special subject selection
        for col in SCORE_COLUMNS:
            if col not in ["cal_subject_name", "cal_type", "cal_score_sum"]:
                program_weight = program.get(col)
                user_score = user_data.get(col)
                
                if pd.notna(program_weight) and user_score is not None:
                    contribution = user_score * (float(program_weight) / 100)
                    score += contribution
                    
                    calculation_details.append({
                        "type": "regular",
                        "subject": col,
                        "score": user_score,
                        "weight": float(program_weight),
                        "contribution": contribution
                    })
    
    # Prepare detailed university information
    university_details = {
        "university_name": program.get("University"),
        "faculty_name": program.get("Faculty"),
        "program_name": program.get("Program"),
        "program_shared": program.get("Program_shared"),
        "program_id": program.get("ID"),
        "gpax_requirement": program.get("gpax_req"),
        "admission_type": program.get("cal_type") if pd.notna(program.get("cal_type")) else "standard",
        "special_calculation": {
            "has_subject_selection": pd.notna(cal_subject_name),
            "subject_selection_info": best_subject_info
        } if pd.notna(cal_subject_name) else None
    }
    
    return {
        "score": round(score, 2),
        "university_details": university_details,
        "calculation_details": calculation_details,
        "total_contributions": len(calculation_details)
    }


def calScore_with_university_details(user_id: str) -> Dict:
    """
    Enhanced score calculation with comprehensive university details
    """
    user_data = get_user_data_fast(user_id)
    if not user_data:
        return {"error": "User not found", "user_id": user_id}
    
    results = []
    user_info = {
        "user_id": user_id,
        "user_name": user_data.get("name"),
        "user_gpax": user_data.get("gpax")
    }
    
    # Check each of the 10 selections
    for i in range(1, 11):
        university = user_data.get(f"selection_{i}_university")
        faculty = user_data.get(f"selection_{i}_faculty")
        field = user_data.get(f"selection_{i}_field")
        
        selection_result = {
            "selection_number": i,
            "selection_data": {
                "university": university,
                "faculty": faculty,
                "field": field
            },
            "status": "incomplete",
            "score": None,
            "university_details": None,
            "calculation_details": None,
            "message": ""
        }
        
        if not (university and faculty and field):
            selection_result["message"] = "Selection incomplete - missing university, faculty, or field"
            results.append(selection_result)
            continue
        
        # Fast program lookup
        program = find_program_fast(university, faculty, field)
        
        if program is None:
            selection_result["status"] = "not_found"
            selection_result["message"] = f"Program not found: {field} at {faculty}, {university}"
            results.append(selection_result)
            continue
        
        # Check GPAX requirement
        gpax_req = program.get("gpax_req")
        user_gpax = user_data.get("gpax", 0)
        
        if pd.notna(gpax_req) and user_gpax and user_gpax < gpax_req:
            selection_result["status"] = "gpax_insufficient"
            selection_result["message"] = f"GPAX requirement not met (required: {gpax_req}, current: {user_gpax})"
            selection_result["university_details"] = {
                "university_name": program.get("University"),
                "faculty_name": program.get("Faculty"),
                "program_name": program.get("Program"),
                "gpax_requirement": float(gpax_req),
                "user_gpax": float(user_gpax)
            }
            results.append(selection_result)
            continue
        
        # Calculate score with detailed information
        try:
            score_data = calculate_program_score_improved(user_data, program)
            
            selection_result.update({
                "status": "calculated",
                "score": score_data["score"],
                "university_details": score_data["university_details"],
                "calculation_details": score_data["calculation_details"],
                "message": f"Score calculated successfully: {score_data['score']:.2f} points"
            })
            
        except Exception as e:
            selection_result["status"] = "error"
            selection_result["message"] = f"Error calculating score: {str(e)}"
        
        results.append(selection_result)
    
    # Calculate summary statistics
    calculated_results = [r for r in results if r["status"] == "calculated"]
    valid_scores = [r["score"] for r in calculated_results if r["score"] is not None]
    
    summary = {
        "total_selections": len([r for r in results if r["selection_data"]["university"]]),
        "calculated_scores": len(calculated_results),
        "failed_calculations": len([r for r in results if r["status"] == "error"]),
        "gpax_insufficient": len([r for r in results if r["status"] == "gpax_insufficient"]),
        "programs_not_found": len([r for r in results if r["status"] == "not_found"]),
        "highest_score": max(valid_scores) if valid_scores else 0,
        "lowest_score": min(valid_scores) if valid_scores else 0,
        "average_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0,
        "recommended_programs": sorted(calculated_results, key=lambda x: x["score"], reverse=True)[:3]
    }
    
    return {
        "user_info": user_info,
        "results": results,
        "summary": summary,
        "calculation_timestamp": datetime.now().isoformat()
    }

# ---- SETUP GOOGLE SHEETS ----
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_json_str = os.getenv("GOOGLE_CREDS_JSON")
creds_json = json.loads(creds_json_str)
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, SCOPE)
CLIENT = gspread.authorize(creds)

# Initialize sheets once
sheet = CLIENT.open_by_key("1IFoQ9PJoralucmufWa11IZ0Njcyq_-Z8NjLmtEySMdY")
worksheet = sheet.get_worksheet(0)  # User data sheet (no headers)
datasheet = sheet.get_worksheet(1)  # University data sheet (has headers)

# Column definitions (define once)
USER_COLUMNS = [
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

NUMERIC_COLUMNS = {
    "gpax", "tgat1", "tgat2", "tgat3", "tpat11", "tpat12", "tpat13",
    "tpat21", "tpat22", "tpat23", "tpat3", "tpat4", "tpat5",
    "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
    "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
    "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89",
    "gpa21", "gpa22", "gpa23", "gpa24", "gpa26", "gpa27", "gpa28"
}

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

def load_and_cache_data():
    """Load data from Google Sheets and cache it with optimized structure"""
    with data_cache._lock:
        if data_cache.is_cache_valid():
            return
        
        try:
            # Load user data
            worksheet_values = worksheet.get_all_values()
            
            # Convert to DataFrame with proper column names
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
                
                # Create user lookup dictionary for O(1) access
                data_cache.user_data_dict = {}
                for _, row in user_df.iterrows():
                    if row['userId']:
                        data_cache.user_data_dict[row['userId']] = row.to_dict()
            
            # Load university data
            university_records = datasheet.get_all_records()
            data_cache.university_data_df = pd.DataFrame(university_records)
            
            # Pre-compute faculty and field mappings for faster lookups
            data_cache.university_faculties.clear()
            data_cache.faculty_fields.clear()
            
            if not data_cache.university_data_df.empty:
                # Cache faculties by university
                faculty_groups = data_cache.university_data_df.groupby('University')['Faculty'].apply(
                    lambda x: x.dropna().unique().tolist()
                ).to_dict()
                data_cache.university_faculties = faculty_groups
                
                # Cache fields by university+faculty combination with Program:Program_shared format
                for (university, faculty), group in data_cache.university_data_df.groupby(['University', 'Faculty']):
                    key = f"{university}|{faculty}"
                    fields = []
                    
                    for _, row in group.iterrows():
                        program = row.get('Program', '')
                        program_shared = row.get('Program_shared', '')
                        
                        # Skip empty programs
                        if pd.isna(program) or str(program).strip() == '':
                            continue
                            
                        # Format: Program:Program_shared
                        if pd.notna(program_shared) and str(program_shared).strip() != '':
                            field_display = f"{program}:{program_shared}"
                        else:
                            field_display = str(program)
                        
                        if field_display not in fields:
                            fields.append(field_display)
                    
                    data_cache.faculty_fields[key] = fields
            
            data_cache.last_update = datetime.now()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

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
    
    # Extract the program name from field (remove :Program_shared part if present)
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
def calculate_program_score_simple(user_data: Dict, program: pd.Series) -> Dict:
    """Simple score calculation with clean result"""
    score = 0
    
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
        if best_subject:
            weight = program.get("cal_score_sum", 0)
            if pd.notna(weight):
                score += best_subject_score * (float(weight) / 100)
        
        # Add other weighted scores (excluding special subjects)
        for col in SCORE_COLUMNS:
            if col not in ["cal_subject_name", "cal_type", "cal_score_sum"] and col != best_subject:
                program_weight = program.get(col)
                user_score = user_data.get(col)
                
                if pd.notna(program_weight) and user_score is not None:
                    score += user_score * (float(program_weight) / 100)
    else:
        # Regular calculation
        for col in SCORE_COLUMNS:
            if col not in ["cal_subject_name", "cal_type", "cal_score_sum"]:
                program_weight = program.get(col)
                user_score = user_data.get(col)
                
                if pd.notna(program_weight) and user_score is not None:
                    score += user_score * (float(program_weight) / 100)
    
    return {
        "score": round(score, 2),
        "program_id": program.get("ID"),
        "program_name": program.get("Program"),
        "university": program.get("University"),
        "faculty": program.get("Faculty")
    }

def calScore_simple(user_id: str) -> Dict:
    """Simple score calculation returning clean results"""
    user_data = get_user_data_fast(user_id)
    if not user_data:
        return {"error": "User not found", "user_id": user_id}
    
    results = []
    
    # Check each of the 10 selections
    for i in range(1, 11):
        university = user_data.get(f"selection_{i}_university")
        faculty = user_data.get(f"selection_{i}_faculty")
        field = user_data.get(f"selection_{i}_field")
        
        if not (university and faculty and field):
            results.append({
                "selection": i,
                "university": university,
                "faculty": faculty,
                "field": field,
                "status": "incomplete",
                "score": None,
                "message": "Selection incomplete"
            })
            continue
        
        # Fast program lookup
        program = find_program_fast(university, faculty, field)
        
        if program is None:
            results.append({
                "selection": i,
                "university": university,
                "faculty": faculty,
                "field": field,
                "status": "not_found",
                "score": None,
                "message": "Program not found"
            })
            continue
        
        # Check GPAX requirement
        gpax_req = program.get("gpax_req")
        user_gpax = user_data.get("gpax", 0)
        
        if pd.notna(gpax_req) and user_gpax and user_gpax < gpax_req:
            results.append({
                "selection": i,
                "university": university,
                "faculty": faculty,
                "field": field,
                "status": "gpax_insufficient",
                "score": None,
                "message": f"GPAX too low (need {gpax_req}, have {user_gpax})",
                "gpax_required": float(gpax_req),
                "gpax_current": float(user_gpax)
            })
            continue
        
        # Calculate score
        try:
            score_data = calculate_program_score_simple(user_data, program)
            results.append({
                "selection": i,
                "university": university,
                "faculty": faculty,
                "field": field,
                "status": "calculated",
                "score": score_data["score"],
                "program_id": score_data["program_id"],
                "program_name": score_data["program_name"],
                "message": f"Score: {score_data['score']}"
            })
        except Exception as e:
            results.append({
                "selection": i,
                "university": university,
                "faculty": faculty,
                "field": field,
                "status": "error",
                "score": None,
                "message": f"Calculation error: {str(e)}"
            })
    
    return {
        "user_id": user_id,
        "user_name": user_data.get("name"),
        "results": results,
        "summary": {
            "total_selections": len([r for r in results if r["university"]]),
            "calculated_scores": len([r for r in results if r["status"] == "calculated"]),
            "highest_score": max([r["score"] for r in results if r["score"] is not None], default=0)
        }
    }

# Update the calculate_scores endpoint
@router.post("/api/calculate_scores_detailed")
async def calculate_scores_detailed(data: dict):
    """
    Calculate scores with comprehensive university details and calculation breakdown
    """
    user_id = data.get("userId")
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    
    try:
        results = calScore_with_university_details(user_id)
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        return results
    except Exception as e:
        print(f"Error calculating detailed scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate scores: {str(e)}")


# Additional endpoint for single program calculation with details
@router.post("/api/calculate_single_program_detailed")
async def calculate_single_program_detailed(data: dict):
    """
    Calculate score for a single program with detailed breakdown
    """
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
                "message": f"GPAX requirement not met (required: {gpax_req}, current: {user_gpax})",
                "university_details": {
                    "university_name": program.get("University"),
                    "faculty_name": program.get("Faculty"),
                    "program_name": program.get("Program"),
                    "gpax_requirement": float(gpax_req),
                    "user_gpax": float(user_gpax)
                }
            }
        
        # Calculate score with detailed breakdown
        score_data = calculate_program_score_improved(user_data, program)
        
        return {
            "status": "success",
            "user_info": {
                "user_id": user_id,
                "user_name": user_data.get("name"),
                "user_gpax": user_data.get("gpax")
            },
            "score": score_data["score"],
            "university_details": score_data["university_details"],
            "calculation_details": score_data["calculation_details"],
            "calculation_summary": {
                "total_components": score_data["total_contributions"],
                "final_score": score_data["score"]
            }
        }
        
    except Exception as e:
        print(f"Error calculating single program score: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate score: {str(e)}")

# Add endpoint for getting user score summary
@router.get("/api/user_scores/{user_id}")
async def get_user_scores(user_id: str):
    """Get calculated scores for a user"""
    try:
        results = calScore_simple(user_id)
        return results
    except Exception as e:
        print(f"Error getting user scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get scores: {str(e)}")

# Add endpoint to calculate score for specific program
@router.post("/api/calculate_program_score")
async def calculate_program_score(data: dict):
    """Calculate score for a specific program"""
    user_id = data.get("userId")
    university = data.get("university")
    faculty = data.get("faculty")
    field = data.get("field")
    
    if not all([user_id, university, faculty, field]):
        raise HTTPException(status_code=400, detail="userId, university, faculty, and field are required")
    
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
        score_data = calculate_program_score_simple(user_data, program)
        
        return {
            "status": "success",
            "score": score_data["score"],
            "program_id": score_data["program_id"],
            "program_name": score_data["program_name"],
            "university": university,
            "faculty": faculty,
            "field": field
        }
        
    except Exception as e:
        print(f"Error calculating program score: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate score: {str(e)}")
        
def calScore_optimized(user_id: str) -> Dict:
    """Optimized score calculation using cached data"""
    user_data = get_user_data_fast(user_id)
    if not user_data:
        return {"error": "User not found"}
    
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
        
        # Fast program lookup
        program = find_program_fast(university, faculty, field)
        
        if program is None:
            selection_result["message"] = "No matching program found"
            selection_result["status"] = "not_found"
            results.append(selection_result)
            continue
        
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
            score = calculate_program_score_optimized(user_data, program)
            selection_result["score"] = score
            selection_result["status"] = "calculated"
            selection_result["message"] = f"Score calculated successfully: {score:.2f}"
        except Exception as e:
            selection_result["message"] = f"Error calculating score: {str(e)}"
            selection_result["status"] = "error"
        
        results.append(selection_result)
    
    return {"user_id": user_id, "results": results}

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
        # Update existing row - find row number from cache or search
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

# ---- PYDANTIC MODELS ---- (unchanged)
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

# ---- OPTIMIZED ENDPOINTS ----
@router.post("/api/find_faculty")
async def find_faculty(data: UniversityRequest):
    """Get faculties for a university (optimized with caching)"""
    try:
        faculties = get_cached_faculties(data.name)
        return {"faculties": faculties}
    except Exception as e:
        print(f"Error finding faculties: {e}")
        return {"error": f"Failed to find faculties: {str(e)}"}

@router.post("/api/find_field")
async def find_field(data: FacultyRequest):
    """Get fields for a university and faculty (optimized with caching)"""
    try:
        fields = get_cached_fields(data.name, data.faculty)
        return {"faculties": fields}  # Keep original response format
    except Exception as e:
        print(f"Error finding fields: {e}")
        return {"error": f"Failed to find fields: {str(e)}"}

@router.post("/api/save-score")
async def save_score(data: ScoreSubmission):
    """Save user score data (optimized)"""
    try:
        upsert_user_data_optimized(worksheet, data)
        return {"message": "Data saved to Google Sheets successfully"}
    except Exception as e:
        print(f"Error saving data: {e}")
        return {"error": f"Failed to save data: {str(e)}"}

@router.post("/api/submit_multiple_selections")
async def submit_multiple_selections(data: MultipleSelectionsSubmission):
    """Save multiple university selections (optimized)"""
    try:
        # Get existing user data
        user_data = get_user_data_fast(data.userId)
        
        # Prepare new row data
        new_row = [""] * len(USER_COLUMNS)
        
        # Set basic user info
        new_row[0] = data.userId
        new_row[1] = data.name
        
        # Copy existing score data if available
        if user_data:
            for i, col in enumerate(USER_COLUMNS[2:37], 2):  # Score columns
                if col in user_data and user_data[col] is not None:
                    new_row[i] = str(user_data[col])

        # Add selections
        for idx, selection in enumerate(data.selections):
            if idx < 10:
                base_idx = 37 + (idx * 3)
                new_row[base_idx] = selection.university
                new_row[base_idx + 1] = selection.faculty
                new_row[base_idx + 2] = selection.field

        # Single sheet operation
        if user_data:
            # Update existing row
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
            worksheet.append_row(new_row)

        # Invalidate cache
        data_cache.invalidate_cache()
        
        return {"message": f"Successfully saved {len(data.selections)} university selections"}

    except Exception as e:
        print(f"Error saving multiple selections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save selections: {str(e)}")

@router.post("/api/calculate_scores")
async def calculate_scores(data: dict):
    """Calculate scores (optimized)"""
    user_id = data.get("userId")
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    
    try:
        results = calScore_optimized(user_id)
        return results
    except Exception as e:
        print(f"Error calculating scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate scores: {str(e)}")

@router.get("/api/user_data/{user_id}")
async def get_user_data(user_id: str):
    """Get user data (optimized)"""
    try:
        user_data = get_user_data_fast(user_id)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"user_data": user_data}
    except Exception as e:
        print(f"Error getting user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user data: {str(e)}")

# ---- CACHE MANAGEMENT ENDPOINTS ----
@router.post("/api/refresh_cache")
async def refresh_cache():
    """Manually refresh the data cache"""
    try:
        data_cache.invalidate_cache()
        load_and_cache_data()
        return {"message": "Cache refreshed successfully"}
    except Exception as e:
        print(f"Error refreshing cache: {e}")
        return {"error": f"Failed to refresh cache: {str(e)}"}

app.include_router(router)
