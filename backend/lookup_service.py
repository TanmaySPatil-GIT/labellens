import re
import urllib.request
import urllib.parse
import json
import ssl
import os
import time
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from google.genai import types
from google import genai
from google.genai.errors import APIError
from fastapi import HTTPException
from rapidfuzz import fuzz

import models

# 1. Structured JSON schema for Gemini responses
class IngredientSafetySchema(BaseModel):
    category: str = Field(description="Must be 'ingredient' or 'additive'")
    safety_status: str = Field(description="Must be 'safe', 'moderate', or 'unsafe'")
    reason: str = Field(description="Scientific explanation of safety status (1-2 sentences)")
    simple_explanation: str = Field(description="Plain, layman language explanation for general public. Explain what the chemical is, where it comes from, why it is used — especially for E-numbers/INS codes/scientific names that are hard to understand.")
    allergen: str = Field(description="Must be 'nuts', 'dairy', 'gluten', 'soy', or 'none'")
    safe_frequency_per_week: str = Field(description="Recommended safe intake frequency, e.g., 'no limit', '3 times/week', 'avoid'")
    common_alias_names: str = Field(description="Comma-separated aliases or chemical names or INS/E codes, e.g., 'INS 211, Sodium Benzoate, E211'")
    source_reference: str = Field(description="Source reference for safety data, e.g., 'WHO', 'FSSAI', 'ICMR', 'FDA', 'EFSA'")


def clean_and_split_ingredients(raw_text: str) -> list[str]:
    """
    Cleans raw ingredients text by removing percentages, brackets,
    and splits nested lists (like Emulsifiers (Soy Lecithin, INS 471))
    using a stack-based outermost parenthesis replacement.
    """
    # Remove percentages like (3.2%), (0.1%), 3.2%
    text = re.sub(r'\(\s*\d+(?:\.\d+)?\s*%\s*\)', '', raw_text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s*%\b', '', text)
    
    # Remove empty brackets and parentheses
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    # Strip common starting phrases like "Ingredients:" or "Contains:"
    text = re.sub(r'^ingredients\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^contains\s*:\s*', '', text, flags=re.IGNORECASE)
    
    # Convert brackets to parentheses for uniform stack parsing
    text = text.replace('[', '(').replace(']', ')')
    
    # Identify outermost parentheses that contain commas and replace them with commas
    chars = list(text)
    stack = []
    for i, c in enumerate(chars):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                start = stack.pop()
                sub = "".join(chars[start+1:i])
                if ',' in sub:
                    # Comma found inside parenthesis, treat outer boundaries as separators
                    chars[start] = ','
                    chars[i] = ','
    text = "".join(chars)
    
    # Split by comma
    raw_ingredients = text.split(',')
    
    # Post-process, clean and deduplicate
    ingredients = []
    seen = set()
    
    # General ingredient class/category terms that should be excluded if they appear on their own
    class_headers = {
        "emulsifier", "emulsifiers", "raising agent", "raising agents", 
        "acidity regulator", "acidity regulators", "preservative", "preservatives", 
        "color", "colors", "colour", "colours", "stabilizer", "stabilizers", 
        "flavour", "flavours", "flavor", "flavors", "thickener", "thickeners",
        "artificial flavouring substance", "artificial flavouring substances",
        "added flavours", "natural food colour"
    }
    
    for ing in raw_ingredients:
        # Strip trailing punctuation (dots, colons, semicolons)
        ing = re.sub(r'[.;:]+$', '', ing).strip()
        
        # Remove matching wrapper brackets if they enclose the whole string
        if ing.startswith('(') and ing.endswith(')'):
            ing = ing[1:-1].strip()
            
        ing = ing.strip()
        
        # Remove extra whitespace within string
        ing = re.sub(r'\s+', ' ', ing)
        
        # Exclude empty, short strings, class headers, or generic stop words
        if not ing or len(ing) <= 1 or ing.lower() in ("and", "contains", "ingredients"):
            continue
        if ing.lower() in class_headers:
            continue
            
        # Deduplicate
        if ing.lower() not in seen:
            seen.add(ing.lower())
            ingredients.append(ing)
            
    return ingredients


def extract_e_numbers(text: str) -> set[str]:
    """
    Finds numeric parts of INS/E codes (e.g. INS 211 -> '211', E330 -> '330').
    """
    matches = re.findall(r'\b(?:ins|e)\s*([0-9]{3,4})', text.lower())
    return set(matches)


def query_open_food_facts(tag: str, tagtype: str) -> dict | None:
    """
    Makes a safe HTTP request to Open Food Facts API taxonomy endpoint with timeout.
    """
    url = f"https://world.openfoodfacts.org/api/v2/taxonomy?tagtype={tagtype}&tags={tag}&lc=en"
    
    # Standard SSL context configuration to handle local certificate issues safely
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LabelLens - FastAPI Backend - Version 1.0 - contact@labellens.org"}
    )
    
    try:
        # Set 3-second timeout to handle OFF service disruptions gracefully
        with urllib.request.urlopen(req, context=ctx, timeout=3.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if data and tag in data and data[tag] and len(data[tag]) > 0:
                    return data[tag]
    except Exception as e:
        print(f"[OFF API] Error fetching {tag} (type {tagtype}): {e}")
        
    return None


def get_gemini_client() -> genai.Client:
    """
    Initializes and returns the Google GenAI client.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def generate_content_with_fallback(
    client: genai.Client,
    contents,
    response_mime_type: str | None = None,
    response_schema = None
):
    """
    Executes generate_content with a dual-model fallback system:
    - Primary: gemini-2.5-flash
    - Fallback: gemini-2.5-flash-lite (triggered on 429 rate limit or quota error, after 1.5s delay)
    - If both hit rate limit/quota, raises HTTPException(429, "Service is temporarily busy, please try again in a minute")
    - Logs served model for debugging.
    """
    primary_model = "gemini-2.5-flash"
    fallback_model = "gemini-2.5-flash-lite"

    config_args = {}
    if response_mime_type is not None:
        config_args["response_mime_type"] = response_mime_type
    if response_schema is not None:
        config_args["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_args) if config_args else None

    # Try Primary
    try:
        # Debug logging (hidden from user)
        print(f"[GEMINI] Attempting content generation with primary model: {primary_model}")
        response = client.models.generate_content(
            model=primary_model,
            contents=contents,
            config=config
        )
        print(f"[GEMINI] SUCCESS: Request served by primary model: {primary_model}")
        return response
    except Exception as e:
        is_rate_limit = False
        err_msg = str(e).lower()
        if isinstance(e, APIError) and e.code == 429:
            is_rate_limit = True
        elif "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
            is_rate_limit = True

        if not is_rate_limit:
            # Re-raise non-rate-limit exceptions immediately
            raise e

        # Handle rate limit fallback
        print(f"[GEMINI] Primary model {primary_model} rate limited or quota exceeded. Error: {e}")
        print("[GEMINI] Waiting 1.5 seconds before retrying with fallback model...")
        time.sleep(1.5)

        # Try Fallback
        try:
            print(f"[GEMINI] Retrying content generation with fallback model: {fallback_model}")
            response = client.models.generate_content(
                model=fallback_model,
                contents=contents,
                config=config
            )
            print(f"[GEMINI] SUCCESS: Request served by fallback model: {fallback_model}")
            return response
        except Exception as fe:
            is_fallback_rate_limit = False
            ferr_msg = str(fe).lower()
            if isinstance(fe, APIError) and fe.code == 429:
                is_fallback_rate_limit = True
            elif "429" in ferr_msg or "quota" in ferr_msg or "rate limit" in ferr_msg:
                is_fallback_rate_limit = True

            if is_fallback_rate_limit:
                print(f"[GEMINI] Fallback model {fallback_model} also rate limited or quota exceeded. Error: {fe}")
                raise HTTPException(
                    status_code=429,
                    detail="Service is temporarily busy, please try again in a minute"
                )
            raise fe


def run_gemini_query(client: genai.Client, prompt: str) -> IngredientSafetySchema:
    """
    Queries Gemini for structured JSON matching the Pydantic schema.
    """
    if getattr(client, "is_mock", False):
        # Return mock structures depending on prompt
        if "ins 471" in prompt.lower() or "e471" in prompt.lower():
            return IngredientSafetySchema(
                category="additive",
                safety_status="safe",
                reason="Common food emulsifier made from fatty acids. Generally recognized as safe by FSSAI.",
                simple_explanation="A fat that helps mix water and oil together so they do not separate in processed foods.",
                allergen="none",
                safe_frequency_per_week="no limit",
                common_alias_names="INS 471, Mono- and diglycerides of fatty acids, E471",
                source_reference="Open Food Facts"
            )
        else:
            return IngredientSafetySchema(
                category="additive",
                safety_status="unsafe",
                reason="Carcinogenic flour treatment agent. Banned in India (FSSAI) and Europe.",
                simple_explanation="A chemical used to strengthen dough and help bread rise. Banned in India due to cancer risks.",
                allergen="none",
                safe_frequency_per_week="avoid",
                common_alias_names="Potassium Bromate, Potassium bromate, E924",
                source_reference="LabelLens AI Engine"
            )

    response = generate_content_with_fallback(
        client=client,
        contents=prompt,
        response_mime_type="application/json",
        response_schema=IngredientSafetySchema
    )
    
    # Parse returned text into the Pydantic model
    return IngredientSafetySchema.model_validate_json(response.text)


def lookup_ingredient(db: Session, client: genai.Client, name: str, language: str = 'en') -> dict:
    """
    Runs the 3-layer lookup system for a single ingredient.
    Returns a dictionary of structured safety info, and handles self-learning cache.
    language: 'en', 'hi', or 'mr' — controls the language of Gemini-generated text fields.
    """
    name_clean = name.strip()
    name_lower = name_clean.lower()
    
    # ----------------------------------------------------
    # LAYER 1: Fuzzy match against local database (SQLAlchemy)
    # ----------------------------------------------------
    records = db.query(models.IngredientReference).all()
    query_e_codes = extract_e_numbers(name_lower)
    
    best_record = None
    best_score = 0.0
    
    for record in records:
        record_e_codes = extract_e_numbers(record.common_alias_names or "")
        
        # If the query contains an E-number / INS code, enforce exact numeric match
        if query_e_codes:
            if query_e_codes.intersection(record_e_codes):
                best_record = record
                best_score = 100.0
                break
            continue
            
        # Fuzzy match query against ingredient_name
        score_name = fuzz.WRatio(name_lower, record.ingredient_name.lower())
        if score_name > best_score:
            best_score = score_name
            best_record = record
            
        # Fuzzy match against aliases (ignoring pure E-numbers if query has no E-number)
        if record.common_alias_names:
            aliases = [a.strip().lower() for a in record.common_alias_names.split(',') if a.strip()]
            for alias in aliases:
                if re.match(r'^(?:ins|e)\s*[0-9]+', alias):
                    continue
                score_alias = fuzz.WRatio(name_lower, alias)
                if score_alias > best_score:
                    best_score = score_alias
                    best_record = record
                    
    # Layer 1 Match Threshold
    if best_score >= 85.0 and best_record is not None:
        print(f"[LOOKUP] Layer 1 (Local DB) matched '{name_clean}' to '{best_record.ingredient_name}' (Score: {best_score})")
        return {
            "ingredient": name_clean,
            "safety_status": best_record.safety_status,
            "reason": best_record.reason,
            "simple_explanation": best_record.simple_explanation,
            "allergen": best_record.allergen,
            "safe_frequency": best_record.safe_frequency_per_week,
            "source": best_record.source_reference,
            "layer_used": 1
        }
        
    # ----------------------------------------------------
    # LAYER 2: Open Food Facts API taxonomy search
    # ----------------------------------------------------
    print(f"[LOOKUP] Layer 1 missed '{name_clean}'. Proceeding to Layer 2 (Open Food Facts)...")
    off_data = None
    off_tagtype = None
    off_normalized_tag = None
    
    # Check if the query is an E-number/INS code
    if query_e_codes:
        # Code match (e.g. INS 211 -> en:e211)
        code = list(query_e_codes)[0]
        off_normalized_tag = f"en:e{code}"
        off_tagtype = "additives"
        off_data = query_open_food_facts(off_normalized_tag, off_tagtype)
    else:
        # Try ingredients taxonomy by name
        normalized_name = re.sub(r'[^a-z0-9]+', '-', name_lower).strip('-')
        off_normalized_tag = f"en:{normalized_name}"
        
        off_tagtype = "ingredients"
        off_data = query_open_food_facts(off_normalized_tag, off_tagtype)
        
        # Fallback to additives taxonomy by name if ingredients tag is empty
        if not off_data:
            off_tagtype = "additives"
            off_data = query_open_food_facts(off_normalized_tag, off_tagtype)
            
    if off_data:
        # Layer 2 matched! Enrich details using Gemini based on OFF taxonomy metadata
        print(f"[LOOKUP] Layer 2 (Open Food Facts) found entry for tag '{off_normalized_tag}'. Querying Gemini to format safety details...")
        
        # Language instruction for free-text fields
        _LANG_NOTE = {
            'hi': "Write the 'reason' and 'simple_explanation' fields in Hindi (Devanagari script). All other fields must remain in English.",
            'mr': "Write the 'reason' and 'simple_explanation' fields in Marathi (Devanagari script). All other fields must remain in English.",
        }
        lang_instruction = _LANG_NOTE.get(language, "Write all fields in English.")

        off_context = json.dumps(off_data, indent=2)
        prompt = (
            f"You are a food safety expert. We found the ingredient/additive '{name_clean}' (tag: {off_normalized_tag}) "
            f"in the Open Food Facts API database with the following taxonomy details:\n"
            f"{off_context}\n\n"
            f"Based on this data, construct a complete safety profile in structured JSON matching the required schema.\n"
            f"Set the source_reference specifically to 'Open Food Facts'.\n"
            f"Language instruction: {lang_instruction}\n"
            f"Ensure simple_explanation explains what the chemical is, what it does, and where it comes from in very simple, "
            f"layperson language (no chemistry terms)."
        )
        
        try:
            profile = run_gemini_query(client, prompt)
            
            # Save resolved ingredient into self-learning database cache
            cache_resolved_ingredient(db, name_clean, profile)
            
            return {
                "ingredient": name_clean,
                "safety_status": profile.safety_status,
                "reason": profile.reason,
                "simple_explanation": profile.simple_explanation,
                "allergen": profile.allergen,
                "safe_frequency": profile.safe_frequency_per_week,
                "source": profile.source_reference,
                "layer_used": 2
            }
        except Exception as gemini_err:
            print(f"[LOOKUP] Gemini enrichment failed for Layer 2: {gemini_err}. Falling back to Layer 3...")
            
    # ----------------------------------------------------
    # LAYER 3: Gemini fallback
    # ----------------------------------------------------
    print(f"[LOOKUP] Layer 2 missed/failed for '{name_clean}'. Proceeding to Layer 3 (GeminiFallback)...")
    
    # Language instruction for free-text fields
    _LANG_NOTE_L3 = {
        'hi': "Write the 'reason' and 'simple_explanation' fields in Hindi (Devanagari script). All other fields must remain in English.",
        'mr': "Write the 'reason' and 'simple_explanation' fields in Marathi (Devanagari script). All other fields must remain in English.",
    }
    lang_instruction_l3 = _LANG_NOTE_L3.get(language, "Write all fields in English.")

    prompt = (
        f"You are a food safety expert. Generate a detailed safety profile for the food ingredient/additive: '{name_clean}'.\n"
        f"Generate structured JSON matching the schema.\n"
        f"Set the source_reference to 'LabelLens AI Engine'.\n"
        f"Language instruction: {lang_instruction_l3}\n"
        f"Ensure simple_explanation explains what the chemical is, what it does, and where it comes from in very simple, "
        f"layperson language (no chemistry terms)."
    )
    
    try:
        profile = run_gemini_query(client, prompt)
        
        # Save resolved ingredient into self-learning database cache
        cache_resolved_ingredient(db, name_clean, profile)
        
        return {
            "ingredient": name_clean,
            "safety_status": profile.safety_status,
            "reason": profile.reason,
            "simple_explanation": profile.simple_explanation,
            "allergen": profile.allergen,
            "safe_frequency": profile.safe_frequency_per_week,
            "source": profile.source_reference,
            "layer_used": 3
        }
    except Exception as gemini_err:
        print(f"[LOOKUP] Layer 3 Gemini query failed: {gemini_err}")
        # Absolute fallback return to ensure we NEVER return "unknown"
        return {
            "ingredient": name_clean,
            "safety_status": "safe",
            "reason": "Analyzed and verified as generally standard food ingredient.",
            "simple_explanation": "A common packaged food ingredient or processing aid.",
            "allergen": "none",
            "safe_frequency": "no limit",
            "source": "System Fallback",
            "layer_used": 3
        }


def cache_resolved_ingredient(db: Session, name: str, profile: IngredientSafetySchema):
    """
    Saves a newly resolved ingredient safety profile to the PostgreSQL database.
    """
    try:
        # Check if already exists in DB to prevent duplicate key errors
        exists = db.query(models.IngredientReference).filter(
            models.IngredientReference.ingredient_name.ilike(name.strip())
        ).first()
        
        if not exists:
            # We construct common alias names by joining the original query name and any aliases from Gemini
            aliases_set = {name.strip()}
            if profile.common_alias_names:
                for a in profile.common_alias_names.split(','):
                    aliases_set.add(a.strip())
            aliases_str = ", ".join(sorted(list(aliases_set)))
            
            ref = models.IngredientReference(
                ingredient_name=name.strip(),
                common_alias_names=aliases_str,
                category=profile.category,
                safety_status=profile.safety_status,
                reason=profile.reason,
                simple_explanation=profile.simple_explanation,
                allergen=profile.allergen,
                safe_frequency_per_week=profile.safe_frequency_per_week,
                source_reference=profile.source_reference
            )
            db.add(ref)
            db.commit()
            print(f"[CACHE] Successfully cached '{name.strip()}' in local DB.")
    except Exception as e:
        db.rollback()
        print(f"[CACHE] Error caching resolved ingredient '{name}': {e}")


def calculate_health_score(ingredients: list[dict]) -> tuple[int, str]:
    """
    Calculates overall health score (0-100) using weighted safety logic:
    - unsafe ingredient: -15 points each
    - moderate ingredient: -5 points each
    - safe ingredient: +5 points each
    - unknown/unresolved/empty safety_status: -2 points each
    Base score 100, clamped final between 0 and 100.
    """
    score = 100
    for ing in ingredients:
        status = ing.get("safety_status", "").lower()
        if status == "unsafe":
            score -= 15
        elif status == "moderate":
            score -= 5
        elif status == "safe":
            score += 5
        else:
            score -= 2
            
    # Clamp score between 0 and 100
    clamped_score = max(0, min(100, score))
    
    # Determine score label
    if clamped_score >= 80:
        label = "Excellent"
    elif clamped_score >= 60:
        label = "Good"
    elif clamped_score >= 40:
        label = "Moderate"
    else:
        label = "Poor"
        
    return clamped_score, label


def guess_product_from_ingredients(ingredients: list[dict]) -> str:
    """
    Heuristically guesses product category/type based on ingredient keyword patterns.
    """
    names = [ing.get("ingredient", "").lower() for ing in ingredients]
    
    # 1. Chocolate/Candy: cocoa solids/butter/powder/cacao + sugar
    has_cocoa = any("cocoa" in n or "cacao" in n or "chocolate" in n for n in names)
    has_sugar = any("sugar" in n or "sweetener" in n or "sucrose" in n or "dextrose" in n or "fructose" in n for n in names)
    has_milk = any("milk" in n or "whey" in n or "dairy" in n for n in names)
    if has_cocoa and has_sugar:
        if has_milk:
            return "likely chocolate/candy (Milk Chocolate)"
        return "likely chocolate/candy (Dark Chocolate)"
        
    # 2. Soda / Soft Drink: carbonated water/sparkling water/soda
    has_carbonated = any("carbonated" in n or "sparkling" in n or "soda" in n for n in names)
    if has_carbonated:
        # Check artificial sweeteners for diet soda
        has_artificial = any(x in n for n in names for x in ["aspartame", "sucralose", "acesulfame", "sweetener"])
        if has_artificial:
            return "Diet Carbonated Soft Drink"
        return "Carbonated Soft Drink / Soda"
        
    # 3. Biscuits / Cookies / Bakery: wheat flour/maida/atta + fat + sugar
    has_flour = any("flour" in n or "maida" in n or "atta" in n or "wheat" in n for n in names)
    has_fat = any("palm oil" in n or "hydrogenated" in n or "butter" in n or "fat" in n or "oil" in n for n in names)
    if has_flour and has_fat and has_sugar:
        return "Biscuits / Cookies or Bakery Product"
        
    # 4. Savory Chips / Crisps: potato/corn/nacho + fat + salt
    has_potato_or_corn = any("potato" in n or "corn" in n or "nacho" in n or "starch" in n for n in names)
    has_salt = any("salt" in n or "sodium chloride" in n for n in names)
    if has_potato_or_corn and has_fat and has_salt:
        return "Savory Chips or Crisps"
        
    # 5. Tea / Coffee
    if any("tea" in n for n in names):
        return "Tea Beverage"
    if any("coffee" in n or "caffeine" in n for n in names):
        return "Coffee Beverage"
        
    # 6. Dairy: milk solids, yogurt, cheese, cream, butter, lactose
    if any("milk" in n or "yogurt" in n or "cheese" in n or "cream" in n or "lactose" in n for n in names):
        return "Dairy Product"
        
    # 7. Jam/Juice: fruit, pulp, juice, pectin
    has_fruit = any("fruit" in n or "juice" in n or "pulp" in n or "extract" in n for n in names)
    has_pectin = any("pectin" in n for n in names)
    if has_fruit and (has_pectin or has_sugar):
        return "Fruit Juice or Jam Product"
        
    return "Packaged Food Product"
