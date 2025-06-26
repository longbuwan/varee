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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
router = APIRouter()

# ---- CONFIGURATION ----
class Config:
    CACHE_TTL_SECONDS = 300  # 5 minutes
    SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    SHEET_ID = "1IFoQ9PJoralucmufWa11IZ0Njcyq_-Z8NjLmtEySMdY"

# ---- COLUMN DEFINITIONS ----
USER_COLUMNS = [
    "userId", "name", "gpax", "tgat", "tgat1", "tgat2", "tgat3", 
    "tpat1", "tpat11", "tpat12", "tpat13",
    "tpat2", "tpat21", "tpat22", "tpat23", 
    "tpat3", "tpat4", "tpat5",
    "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
    "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
    "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89", 
    "gpa21", "gpa22", "gpa23", "gpa24", "gpa25", "gpa26", "gpa27", "gpa28",
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

# Numeric columns (all score-related fields)
NUMERIC_COLUMNS = [
    "gpax", "tgat", "tgat1", "tgat2", "tgat3", 
    "tpat1", "tpat11", "tpat12", "tpat13",
    "tpat2", "tpat21", "tpat22", "tpat23", 
    "tpat3", "tpat4", "tpat5",
    "a_lv_61", "a_lv_62", "a_lv_63", "a_lv_64", "a_lv_65", "a_lv_66",
    "a_lv_70", "a_lv_81", "a_lv_82", "a_lv_83", "a_lv_84", "a_lv_85",
    "a_lv_86", "a_lv_87", "a_lv_88", "a_lv_89", 
    "gpa21", "gpa22", "gpa23", "gpa24", "gpa25", "gpa26", "gpa27", "gpa28"
]

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

# Numeric columns for university data that need conversion
UNIVERSITY_NUMERIC_COLUMNS = [
    "gpax_req", "projected_min_score_68_from_67", "คะแนนต่ำสุด_67", 
    "คะแนนต่ำสุด ประมวลผลครั้งที่ 1_68", "cal_score_sum"
] + SCORE_COLUMNS

# ---- PYDANTIC MODELS ----
class ScoreSubmission(BaseModel):
    userId: Optional[str] = None
    name: Optional[str] = None
    gpax: Optional[float] = None
    tgat: Optional[float] = None
    tgat1: Optional[float] = None
    tgat2: Optional[float] = None
    tgat3: Optional[float] = None
    tpat1: Optional[float] = None
    tpat11: Optional[float] = None
    tpat12: Optional[float] = None
    tpat13: Optional[float] = None
    tpat2: Optional[float] = None
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

    class Config:
        extra = "forbid"  # Prevent extra fields

class UniversityRequest(BaseModel):
    name: str
    userId: Optional[str] = None

class FacultyRequest(BaseModel):
    name: str
    faculty: str

class UniversitySelection(BaseModel):
    university: str
    faculty: str
    field: str

class MultipleSelectionsSubmission(BaseModel):
    userId: str
    name: str
    selections: List[UniversitySelection]

# ---- HELPER FUNCTIONS ----
def safe_float_conversion(value, default=None):
    """Safely convert a value to float, returning default if conversion fails"""
    if value is None or pd.isna(value):
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_numeric_comparison(val1, val2, operator='<'):
    """Safely compare two values after converting to float"""
    try:
        num1 = safe_float_conversion(val1)
        num2 = safe_float_conversion(val2)
        
        if num1 is None or num2 is None:
            return False
            
        if operator == '<':
            return num1 < num2
        elif operator == '>':
            return num1 > num2
        elif operator == '<=':
            return num1 <= num2
        elif operator == '>=':
            return num1 >= num2
        elif operator == '==':
            return num1 == num2
        else:
            return False
    except:
        return False

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
        logger.error(f"Error setting up Google Sheets: {e}")
        raise

# Initialize sheets (with proper error handling)
try:
    worksheet, datasheet = setup_google_sheets()
    logger.info("Google Sheets initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize Google Sheets: {e}")
    worksheet, datasheet = None, None

# ---- DATA VALIDATION FUNCTIONS ----
def validate_score_submission(data: ScoreSubmission) -> Dict[str, Any]:
    """Validate score submission data"""
    errors = []
    warnings = []
    
    # Check required fields
    if not data.userId:
        errors.append("userId is required")
    if not data.name:
        errors.append("name is required")
    
    # Validate score ranges
    score_validations = [
        ("gpax", data.gpax, 0, 100, "GPAX should be between 0-100"),
        ("tgat1", data.tgat1, 0, 100, "TGAT1 should be between 0-100"),
        ("tgat2", data.tgat2, 0, 100, "TGAT2 should be between 0-100"),
        ("tgat3", data.tgat3, 0, 100, "TGAT3 should be between 0-100"),
    ]
    
    for field_name, value, min_val, max_val, message in score_validations:
        if value is not None and (value < min_val or value > max_val):
            errors.append(f"{message}, got {value}")
    
    # Check for missing critical scores
    critical_fields = ["gpax", "tgat1", "tgat2", "tgat3"]
    missing_critical = [field for field in critical_fields if getattr(data, field) is None]
    if missing_critical:
        warnings.append(f"Missing critical scores: {', '.join(missing_critical)}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "has_warnings": len(warnings) > 0
    }

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
            program_weight = safe_float_conversion(program.get(col))
            if program_weight is not None and program_weight > 0:
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
        cal_score_sum = safe_float_conversion(program.get("cal_score_sum"))
        
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
                        user_score = safe_float_conversion(user_data.get(subj))
                        if user_score is not None and user_score > best_subject_score:
                            best_subject_score = user_score
                            best_subject = subj
                else:
                    # Single subject
                    user_score = safe_float_conversion(user_data.get(spec))
                    if user_score is not None and user_score > best_subject_score:
                        best_subject_score = user_score
                        best_subject = spec
            
            # Add best subject score with its weight
            if best_subject and cal_score_sum is not None:
                weight = cal_score_sum / 100
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
                    
                    program_weight = safe_float_conversion(program.get(col))
                    user_score = safe_float_conversion(user_data.get(col))
                    
                    if program_weight is not None and user_score is not None and program_weight > 0:
                        weight = program_weight / 100
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
                    program_weight = safe_float_conversion(program.get(col))
                    user_score = safe_float_conversion(user_data.get(col))
                    
                    if program_weight is not None and user_score is not None and program_weight > 0:
                        weight = program_weight / 100
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
        logger.error(f"Error calculating score: {e}")
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
        
        if worksheet is None or datasheet is None:
            raise Exception("Google Sheets not properly initialized")
        
        try:
            logger.info("Loading data from Google Sheets...")
            
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
                
                logger.info(f"Loaded {len(user_df)} user records")
            
            # Load university data
            university_records = datasheet.get_all_records()
            data_cache.university_data_df = pd.DataFrame(university_records)
            
            # Convert numeric columns for university data
            if not data_cache.university_data_df.empty:
                for col in UNIVERSITY_NUMERIC_COLUMNS:
                    if col in data_cache.university_data_df.columns:
                        data_cache.university_data_df[col] = pd.to_numeric(
                            data_cache.university_data_df[col], errors='coerce'
                        )
                
                logger.info(f"Loaded {len(data_cache.university_data_df)} university programs")
            
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
            logger.info("Data loaded and cached successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
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

# ---- DATA SAVING FUNCTIONS ----
def upsert_user_data_optimized(worksheet, data: ScoreSubmission):
    """Improved user data upsert with better duplicate prevention and logging"""
    try:
        logger.info(f"Upserting data for user {data.userId}")
        
        # First, get all current data to find existing row
        all_values = worksheet.get_all_values()
        
        # Find existing row by userId (case-insensitive and strip whitespace)
        existing_row_index = None
        target_user_id = str(data.userId).strip().lower()
        
        for i, row in enumerate(all_values):
            if row and len(row) > 0:
                current_user_id = str(row[0]).strip().lower()
                if current_user_id == target_user_id:
                    existing_row_index = i + 1  # Google Sheets is 1-indexed
                    break
        
        # Get current user data from cache
        user_data = get_user_data_fast(data.userId)
        
        # Prepare new row data
        new_row = [""] * len(USER_COLUMNS)
        
        # Set provided data
        for i, col in enumerate(USER_COLUMNS):
            if col.startswith("selection_"):
                # Keep existing selection data if not being updated
                if user_data and col in user_data and user_data[col] is not None:
                    new_row[i] = str(user_data[col])
            else:
                val = getattr(data, col, None)
                if val is not None:
                    new_row[i] = str(val)
                elif user_data and col in user_data and user_data[col] is not None:
                    # Keep existing data for fields not being updated
                    new_row[i] = str(user_data[col])
        
        # Log the data being saved
        non_empty_data = {col: new_row[i] for i, col in enumerate(USER_COLUMNS) if new_row[i] and new_row[i] != ""}
        logger.info(f"Saving data for user {data.userId}: {len(non_empty_data)} fields")
        
        # Single sheet operation
        if existing_row_index:
            # Update existing row
            end_col_letter = gspread.utils.rowcol_to_a1(1, len(USER_COLUMNS))[:-1]
            cell_range = f"A{existing_row_index}:{end_col_letter}{existing_row_index}"
            worksheet.update(cell_range, [new_row])
            logger.info(f"Updated existing row {existing_row_index} for user {data.userId}")
        else:
            # Add new row
            worksheet.append_row(new_row)
            logger.info(f"Added new row for user {data.userId}")
        
        # Invalidate cache to force refresh
        data_cache.invalidate_cache()
        
    except Exception as e:
        logger.error(f"Error in upsert_user_data_optimized: {e}")
        raise

def save_multiple_selections(worksheet, user_id: str, name: str, selections: List[Dict]):
    """Improved multiple selections save with better duplicate handling"""
    try:
        logger.info(f"Saving {len(selections)} selections for user {user_id}")
        
        # Get all current data to find existing row
        all_values = worksheet.get_all_values()
        
        # Find existing row by userId
        existing_row_index = None
        target_user_id = str(user_id).strip().lower()
        
        for i, row in enumerate(all_values):
            if row and len(row) > 0:
                current_user_id = str(row[0]).strip().lower()
                if current_user_id == target_user_id:
                    existing_row_index = i + 1
                    break
        
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

        # Clear all existing selections first
        selection_start_idx = USER_COLUMNS.index("selection_1_university")
        for i in range(selection_start_idx, len(USER_COLUMNS)):
            new_row[i] = ""

        # Add new selections
        for idx, selection in enumerate(selections):
            if idx < 10:  # Max 10 selections
                base_idx = selection_start_idx + (idx * 3)
                if base_idx + 2 < len(new_row):
                    new_row[base_idx] = selection.get("university", "")
                    new_row[base_idx + 1] = selection.get("faculty", "")
                    new_row[base_idx + 2] = selection.get("field", "")

        # Single sheet operation
        if existing_row_index:
            # Update existing row
            end_col_letter = gspread.utils.rowcol_to_a1(1, len(USER_COLUMNS))[:-1]
            cell_range = f"A{existing_row_index}:{end_col_letter}{existing_row_index}"
            worksheet.update(cell_range, [new_row])
            logger.info(f"Updated selections for existing user {user_id} at row {existing_row_index}")
        else:
            # Add new row
            worksheet.append_row(new_row)
            logger.info(f"Added new user {user_id} with selections")

        # Invalidate cache
        data_cache.invalidate_cache()
        
    except Exception as e:
        logger.error(f"Error in save_multiple_selections: {e}")
        raise

# ---- API ENDPOINTS ----
@router.post("/api/save-score")
async def save_score(data: ScoreSubmission):
    """Save user score data with enhanced validation and logging"""
    try:
        # Validate the submission
        validation = validate_score_submission(data)
        
        if not validation["is_valid"]:
            logger.warning(f"Invalid score submission for user {data.userId}: {validation['errors']}")
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": "Invalid data submission",
                    "errors": validation["errors"],
                    "warnings": validation.get("warnings", [])
                }
            )
        
        if worksheet is None:
            raise HTTPException(status_code=500, detail="Google Sheets not initialized")
        
        # Log warnings if any
        if validation["has_warnings"]:
            logger.warning(f"Warnings for user {data.userId}: {validation['warnings']}")
        
        # Save the data
        upsert_user_data_optimized(worksheet, data)
        
        # Count non-null fields for response
        data_dict = data.dict()
        non_null_fields = sum(1 for v in data_dict.values() if v is not None)
        
        logger.info(f"Successfully saved {non_null_fields} fields for user {data.userId}")
        
        response_data = {
            "message": "Data saved to Google Sheets successfully",
            "user_id": data.userId,
            "fields_saved": non_null_fields,
            "warnings": validation.get("warnings", [])
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving data for user {data.userId}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

@router.post("/api/find_faculty")
async def find_faculty(data: UniversityRequest):
    """Get faculties for a university"""
    try:
        faculties = get_cached_faculties(data.name)
        return {"faculties": faculties}
    except Exception as e:
        logger.error(f"Error finding faculties: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find faculties: {str(e)}")

@router.post("/api/find_field")
async def find_field(data: FacultyRequest):
    """Get fields for a university and faculty"""
    try:
        fields = get_cached_fields(data.name, data.faculty)
        return {"faculties": fields}  # Keep original response format for compatibility
    except Exception as e:
        logger.error(f"Error finding fields: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find fields: {str(e)}")

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
        
        if worksheet is None:
            raise HTTPException(status_code=500, detail="Google Sheets not initialized")
        
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
        logger.error(f"Error saving multiple selections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save selections: {str(e)}")

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
        logger.error(f"Error getting user data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user data: {str(e)}")

@router.get("/api/health")
async def health_check():
    """Health check endpoint with detailed status"""
    try:
        load_and_cache_data()
        
        user_count = len(data_cache.user_data_dict) if data_cache.user_data_dict else 0
        university_count = len(data_cache.university_data_df) if data_cache.university_data_df is not None else 0
        
        return {
            "status": "healthy",
            "cache_valid": data_cache.is_cache_valid(),
            "last_update": data_cache.last_update.isoformat() if data_cache.last_update else None,
            "user_data_loaded": data_cache.user_data_df is not None,
            "university_data_loaded": data_cache.university_data_df is not None,
            "google_sheets_initialized": worksheet is not None and datasheet is not None,
            "user_count": user_count,
            "university_program_count": university_count,
            "cached_faculties": len(data_cache.university_faculties),
            "cached_fields": len(data_cache.faculty_fields)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/api/refresh_cache")
async def refresh_cache():
    """Manually refresh the data cache"""
    try:
        data_cache.invalidate_cache()
        load_and_cache_data()
        return {"message": "Cache refreshed successfully"}
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")

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
        logger.error(f"Debug endpoint error: {e}")
        return {"error": str(e), "user_id": user_id}

# Include router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
