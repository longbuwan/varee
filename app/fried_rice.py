@router.post("/api/new_program")
async def calculate_scores(data: dict, uni: dict):
    """Calculate scores for all user selections"""
    user_id = data.get("userId")
    if not user_id:
        raise HTTPException(status_code=400, detail="userId is required")
    
    try:
        user_data = get_user_data_fast(user_id)
        if not user_data:
            return {"error": "User not found", "user_id": user_id}
        
        results = []
        
        # Check each of the 10 selections with correct column names
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
                "score_d": None,
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
            
            # Check GPAX requirement with safe conversion
            gpax_req = safe_float_conversion(program.get("gpax_req"))
            user_gpax = safe_float_conversion(user_data.get("gpax"))
            
            # Get projected minimum score with safe conversion
            score_p = safe_float_conversion(program.get("projected_min_score_68_from_67"))
            if score_p is None:
                score_p = safe_float_conversion(program.get("คะแนนต่ำสุด_67"))
                if score_p is None:
                    score_p = safe_float_conversion(program.get("คะแนนต่ำสุด ประมวลผลครั้งที่ 1_68"), 0)
            
            if gpax_req is not None and user_gpax is not None and user_gpax < gpax_req:
                selection_result["status"] = "gpax_insufficient"
                selection_result["message"] = f"GPAX requirement not met (required: {gpax_req}, current: {user_gpax})"
                selection_result["gpax_required"] = gpax_req
                selection_result["gpax_current"] = user_gpax
                results.append(selection_result)
                continue
            
            # Calculate score
            score_result = calculate_program_score(user_data, program)
            
            if score_result["success"]:
                selection_result["status"] = "calculated"
                selection_result["score"] = score_result["score"]
                selection_result["message"] = score_result["message"]
                selection_result["score_breakdown"] = score_result["score_breakdown"]
                selection_result["score_d"] = score_result["score"] - score_p if score_p is not None else None
            else:
                selection_result["status"] = "error"
                selection_result["message"] = score_result["message"]
                if score_result["error"] == "missing_scores":
                    selection_result["missing_scores"] = score_result["missing_scores"]
                    selection_result["missing_count"] = score_result["missing_count"]
                    selection_result["total_required"] = score_result["total_required"]

            results.append(selection_result)
        
        # Calculate summary
        calculated_scores = [r["score"] for r in results if r["score"] is not None]
        
        universitiy_c = uni.get("University")
        faculties = get_cached_faculties(universitiy_c)
        feilds = []
        programs = []
        scores = []
        for i in faculties:
            field_c = get_cached_fields(universitiy_c, i)
            feilds.append(field_c)
            for i in feilds:
                programs = find_program_fast(universitiy_c, faculties, feilds)
        for i in programs:
            
            # Check GPAX requirement with safe conversion
            gpax_req = safe_float_conversion(program.get("gpax_req"))
            user_gpax = safe_float_conversion(user_data.get("gpax"))
            
            # Get projected minimum score with safe conversion
            score_p = safe_float_conversion(program.get("projected_min_score_68_from_67"))
            if score_p is None:
                score_p = safe_float_conversion(program.get("คะแนนต่ำสุด_67"))
                if score_p is None:
                    score_p = safe_float_conversion(program.get("คะแนนต่ำสุด ประมวลผลครั้งที่ 1_68"), 0)
            
            if gpax_req is not None and user_gpax is not None and user_gpax < gpax_req:
                selection_result["status"] = "gpax_insufficient"
                selection_result["message"] = f"GPAX requirement not met (required: {gpax_req}, current: {user_gpax})"
                selection_result["gpax_required"] = gpax_req
                selection_result["gpax_current"] = user_gpax
                results.append(selection_result)
                continue
            
            # Calculate score
            score_result = calculate_program_score(user_data, program)
            
            
            if score_result["success"]:
                selection_result["status"] = "calculated"
                selection_result["score"] = score_result["score"]
                selection_result["message"] = score_result["message"]
                selection_result["score_breakdown"] = score_result["score_breakdown"]
                selection_result["score_d"] = score_result["score"] - score_p if score_p is not None else None
            else:
                selection_result["status"] = "error"
                selection_result["message"] = score_result["message"]
                if score_result["error"] == "missing_scores":
                    selection_result["missing_scores"] = score_result["missing_scores"]
                    selection_result["missing_count"] = score_result["missing_count"]
                    selection_result["total_required"] = score_result["total_required"]
            if selection_result["score_d"] > 0:
                selection_result["should"] = "true"
            else:
                selection_result["should"] = "false"
            results.append(selection_result)

























        response_data = {
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
        
















        if "error" in response_data:
            raise HTTPException(status_code=404, detail=response_data["error"])
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating scores: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate scores: {str(e)}")
