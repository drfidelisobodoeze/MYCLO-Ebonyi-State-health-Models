# ============================================================
# CLINICAL RULES FOR LASSA FEVER (FINAL)
# ============================================================
def lassa_clinical_rules(input_data):
    """
    Overrides ML prediction based on clinical rules:
    1. Lab positive -> Confirmed Case
    2. Fever > 38C -> Suspected Case
    3. Lab negative AND all other categorical features are 'No' -> Not a Case
    """
    temp = input_data.get("Current_body_temperature_in___C", 37)
    lab_result = input_data.get("Latest_sample_final_laboratory_result", "Negative").upper()
    
    # List of other categorical features (excluding lab result)
    categorical_features = [
        "Fever",
        "Abdominal_pain",
        "Bleeding_or_bruising",
        "Vomiting",
        "Sore_throat",
        "Diarrhea",
        "General_weakness",
        "Chest_pain"
    ]
    
    # Check if all other categorical features are 'No'
    all_no = all(input_data.get(f, "No") == "No" for f in categorical_features)

    # Apply rules in order of priority
    if lab_result == "POSITIVE":
        return "Confirmed Case"
    elif lab_result == "NEGATIVE" and all_no:
        return "Not a Case"
    elif temp > 38:
        return "Suspected Case"
    else:
        return None  # follow ML prediction
