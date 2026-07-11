import os
import io
import json
import time
import uvicorn
import cv2
import numpy as np
import urllib.request
import urllib.parse
import ssl
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
from google import genai
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import jwt, JWTError

import csv
from contextlib import asynccontextmanager
from database import check_db_connection, engine, Base, SessionLocal, get_db
import models
from lookup_service import clean_and_split_ingredients, lookup_ingredient, calculate_health_score, guess_product_from_ingredients, extract_e_numbers

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.

    Startup sequence:
      1. Verify PostgreSQL connectivity (3 attempts, 2 s apart). Hard-fails if all
         attempts fail — no SQLite fallback, no silent substitution.
      2. Create ORM tables if they do not yet exist.
      3. Seed 'ingredients_reference' from CSV if the table is empty.
    """
    # ------------------------------------------------------------------ #
    # 1. PostgreSQL connection check — retry 3 times, then abort.         #
    # ------------------------------------------------------------------ #
    _MAX_RETRIES = 3
    _RETRY_DELAY_SEC = 2

    for attempt in range(1, _MAX_RETRIES + 1):
        if check_db_connection():
            print(f"[DATABASE] PostgreSQL connection verified (attempt {attempt}/{_MAX_RETRIES}).")
            break
        if attempt < _MAX_RETRIES:
            print(
                f"[DATABASE] Connection attempt {attempt}/{_MAX_RETRIES} failed. "
                f"Retrying in {_RETRY_DELAY_SEC}s..."
            )
            time.sleep(_RETRY_DELAY_SEC)
    else:
        # All retries exhausted — refuse to start.
        raise RuntimeError(
            "\n"
            "================================================================\n"
            " DATABASE CONNECTION FAILED - server is refusing to start.     \n"
            "                                                                \n"
            " * Check that PostgreSQL / Neon is reachable.                  \n"
            " * Verify DATABASE_URL in backend/.env is correct.             \n"
            " * LabelLens requires PostgreSQL and will never fall back to    \n"
            "   SQLite or any other engine.                                  \n"
            "================================================================\n"
        )

    # ------------------------------------------------------------------ #
    # 2. Create ORM tables if they do not exist.                          #
    # ------------------------------------------------------------------ #
    try:
        Base.metadata.create_all(bind=engine)
        print("[DATABASE] ORM tables verified / created.")
    except Exception as create_err:
        raise RuntimeError(f"[DATABASE] Failed to create ORM tables: {create_err}") from create_err

    # ------------------------------------------------------------------ #
    # 3. Seed 'ingredients_reference' from CSV if empty.                 #
    # ------------------------------------------------------------------ #
    db = SessionLocal()
    try:
        count = db.query(models.IngredientReference).count()
        if count == 0:
            csv_path = os.path.join(os.path.dirname(__file__), "data", "ingredient_safety_db.csv")
            if os.path.exists(csv_path):
                print(f"[DATABASE] Table is empty — seeding from {csv_path} ...")
                with open(csv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ref = models.IngredientReference(
                            ingredient_name=row["ingredient_name"].strip(),
                            common_alias_names=row["common_alias_names"].strip() if row.get("common_alias_names") else None,
                            category=row["category"].strip() if row.get("category") else None,
                            safety_status=row["safety_status"].strip(),
                            reason=row["reason"].strip() if row.get("reason") else None,
                            simple_explanation=row["simple_explanation"].strip() if row.get("simple_explanation") else None,
                            allergen=row["allergen"].strip() if row.get("allergen") else None,
                            safe_frequency_per_week=row["safe_frequency_per_week"].strip() if row.get("safe_frequency_per_week") else None,
                            source_reference=row["source_reference"].strip() if row.get("source_reference") else None,
                        )
                        db.add(ref)
                    db.commit()
                print("[DATABASE] Seeding completed successfully.")
            else:
                print(f"[DATABASE] Seeding skipped — CSV not found at {csv_path}")
        else:
            print(f"[DATABASE] 'ingredients_reference' already populated ({count} rows). Skipping seed.")
    except Exception as seed_err:
        print(f"[DATABASE] Error during seeding: {seed_err}")
        db.rollback()
    finally:
        db.close()

    yield  # Server is running

app = FastAPI(title="LabelLens API", version="0.1.0", lifespan=lifespan)

# ──────────────────────────────────────────────────────────────────────
# Phase 9: User Authentication, Hashing, and JWT Tokens Setup
# ──────────────────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable is not set")
    
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 day

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> models.User:
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

def get_current_user_optional(credentials: HTTPAuthorizationCredentials | None = Depends(security_optional), db: Session = Depends(get_db)) -> models.User | None:
    if not credentials:
        return None
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        user = db.query(models.User).filter(models.User.email == email).first()
        return user
    except JWTError:
        return None




# Configure CORS origins
# Allow localhost origins (default Vite port is 5173)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Allow custom production origins if defined in .env
env_origins = os.getenv("CORS_ORIGINS")
if env_origins:
    origins.extend([o.strip() for o in env_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint that tests connectivity to the database.
    Returns status and whether database is successfully connected.
    """
    db_connected = check_db_connection()
    return {
        "status": "ok",
        "db_connected": db_connected
    }

def detect_blur(image_bytes: bytes, threshold: float = 30.0) -> tuple[bool, float]:
    """
    Calculates the Laplacian variance of the image to determine if it is blurry.
    Only rejects if the image resolution is high enough (>= 400px width/height) and variance < threshold.
    Returns (is_blurry, variance).
    """
    # Convert bytes to numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance (measure of focus/high frequency changes)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Get image dimensions
    height, width = gray.shape[:2]
    print(f"[BLUR] Image uploaded: {width}x{height}, Laplacian variance: {variance:.2f} (threshold: {threshold})")
    
    # Skip strict blur rejection if resolution is too small
    if width < 400 or height < 400:
        return False, variance
        
    return variance < threshold, variance

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...), user: models.User = Depends(get_current_user)):
    """
    Validates the uploaded file size and content type,
    performs blur check via OpenCV, and uses Gemini Vision API
    to extract ingredients list.
    """
    # 1. Validate file format
    allowed_types = ["image/jpeg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file.content_type}'. Only JPEG and PNG images are allowed."
        )
    
    # 2. Validate file size (limit to 5MB)
    max_size = 5 * 1024 * 1024  # 5MB
    image_bytes = await file.read()
    file_size = len(image_bytes)
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds 5MB limit. Uploaded file was {file_size / (1024 * 1024):.2f}MB."
        )
    
    # 3. Blur detection via OpenCV
    try:
        is_blurry, variance = detect_blur(image_bytes, threshold=30.0)
        if is_blurry:
            raise HTTPException(
                status_code=400,
                detail=f"The uploaded image is too blurry (variance: {variance:.1f}). Please upload a clearer, focused photo."
            )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Failed to parse upload as a valid image."
        )
    except HTTPException:
        # Re-raise HTTPExceptions from blur check
        raise
    except Exception as err:
        # Log unexpected errors during OpenCV validation, but continue as fallback
        print(f"OpenCV processing error: {err}")
    
    # 4. Extract ingredients using Google Gemini Vision API
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print(
            "\n*** WARNING ****************************************************\n"
            "MOCK MODE - no real Gemini API key found (GEMINI_API_KEY unset)\n"
            "Returning a hardcoded ingredient list for local development.   \n"
            "Set GEMINI_API_KEY in backend/.env before deploying to prod.   \n"
            "****************************************************************\n"
        )
        return {
            "success": True,
            "raw_text": "Ingredients: Sugar, Refined Palm Oil, Milk Solids, Cocoa butter, INS 471, Potassium Bromate."
        }
        
    try:
        # Convert bytes to PIL Image for the GenAI SDK
        image = Image.open(io.BytesIO(image_bytes))
        
        # Initialize Google GenAI client
        client = genai.Client(api_key=gemini_api_key)
        
        # Call the model via fallback wrapper
        prompt = (
            "Extract the raw ingredient list as plain text from the image. "
            "Focus specifically on the list of ingredients. Do not summarize or format into a report. "
            "Return only the raw list of ingredients as plain text."
        )
        
        from lookup_service import generate_content_with_fallback
        response = generate_content_with_fallback(
            client=client,
            contents=[prompt, image]
        )
        
        if not response.text:
            raise ValueError("Empty response text from Gemini API")
            
        return {
            "success": True,
            "raw_text": response.text
        }
        
    except Exception as api_err:
        print(f"Gemini API invocation error: {api_err}")
        raise HTTPException(
            status_code=500,
            detail="AI analysis failed. Please try again in a moment."
        )

from pydantic import BaseModel

class ParseIngredientsRequest(BaseModel):
    raw_text: str
    language: str = 'en'  # 'en', 'hi', 'mr'

@app.post("/api/parse-ingredients")
async def parse_ingredients(payload: ParseIngredientsRequest, db = Depends(get_db), user: models.User = Depends(get_current_user)):
    """
    Endpoint that splits raw text into individual ingredients, matches them
    using the 3-layer lookup system, calculates a weighted health score,
    heuristically guesses the product type, and returns a structured safety list.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print(
            "\n*** WARNING ****************************************************\n"
            "MOCK MODE - no real Gemini API key found (GEMINI_API_KEY unset)\n"
            "Ingredient lookups will use mock Gemini responses.            \n"
            "Set GEMINI_API_KEY in backend/.env before deploying to prod.  \n"
            "****************************************************************\n"
        )
        class MockClient:
            is_mock = True
        client = MockClient()
    else:
        # Initialize Gemini client
        client = genai.Client(api_key=gemini_api_key)
        
    try:
        # 1. Clean and split raw ingredients text
        raw_list = clean_and_split_ingredients(payload.raw_text)
        
        # 2. Resolve each ingredient
        results = []
        for name in raw_list:
            resolved = lookup_ingredient(db, client, name, language=payload.language)
            results.append({
                "ingredient": resolved["ingredient"],
                "safety_status": resolved["safety_status"],
                "reason": resolved["reason"],
                "simple_explanation": resolved["simple_explanation"],
                "allergen": resolved["allergen"],
                "safe_frequency": resolved["safe_frequency"],
                "source": resolved["source"]
            })
            
        # 3. Calculate overall health score and classification
        overall_score, score_label = calculate_health_score(results)
        
        # 4. Classify product guess
        product_guess = guess_product_from_ingredients(results)
        
        # 5. Save to history (user is always authenticated now)
        try:
            hist = models.ScanHistory(
                user_id=user.id,
                product_name=product_guess,
                product_guess=product_guess,
                overall_score=overall_score,
                score_label=score_label,
                ingredients_data=results
            )
            db.add(hist)
            db.commit()
            print(f"[HISTORY] Successfully saved scan for user {user.email}")
        except Exception as hist_err:
            db.rollback()
            print(f"[HISTORY] Error saving scan: {hist_err}")
                
        return {
            "ingredients": results,
            "overall_score": overall_score,
            "score_label": score_label,
            "product_guess": product_guess
        }
    except Exception as err:
        print(f"Error parsing ingredients: {err}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse and lookup ingredients: {str(err)}"
        )

class SuggestAlternativeRequest(BaseModel):
    product_guess: str
    ingredients: list[str]
    language: str = 'en'  # 'en', 'hi', 'mr'

_ALTERNATIVE_FALLBACK = {
    "alternative_name": "Fresh / home-made options",
    "reasons": [
        "No specific alternative could be identified for this product.",
        "Home-made food avoids hidden additives and preservatives.",
        "Fresh ingredients give you full control over what you eat.",
    ],
}

def _call_gemini_for_alternative(client, product_guess: str, ingredients: list[str], language: str = 'en') -> dict:
    """
    Calls the Gemini API to suggest a healthier alternative product available in India.
    Returns a dict with keys 'alternative_name' and 'reasons' (list of strings).
    Raises ValueError if the response cannot be parsed as valid JSON with required keys.
    """
    _LANG_INSTRUCTION = {
        'hi': "Respond in Hindi (Devanagari script). The 'reasons' array values must be in Hindi.",
        'mr': "Respond in Marathi (Devanagari script). The 'reasons' array values must be in Marathi.",
    }
    lang_note = _LANG_INSTRUCTION.get(language, "Respond in English.")

    ingredient_summary = ", ".join(ingredients[:20])  # cap at 20 to stay within token limits
    prompt = (
        f"You are a food safety advisor for Indian consumers.\n"
        f"A user just scanned a packaged food product categorised as: '{product_guess}'.\n"
        f"The product contains these ingredients: {ingredient_summary}.\n\n"
        f"Suggest ONE real, commonly available healthier alternative product or category "
        f"that a person in India can easily buy or prepare instead.\n\n"
        f"Language instruction: {lang_note}\n\n"
        f"Respond ONLY with valid JSON — no markdown, no explanation, no extra text:\n"
        f'{{"alternative_name": "<product name>", "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"]}}\n\n'
        f"Rules:\n"
        f"- 'alternative_name' must be a real, specific product or food category available in India (keep in English or common name).\n"
        f"- 'reasons' must be a JSON array of exactly 2–3 concise strings (each under 20 words) in the requested language.\n"
        f"- Do NOT wrap the JSON in code fences or add any commentary.\n"
        f"- If no specific alternative exists, use: "
        f'{{"alternative_name": "Fresh / home-made options", '
        f'"reasons": ["Avoids hidden additives", "Full ingredient control", "Generally lower in preservatives"]}}'
    )

    from lookup_service import generate_content_with_fallback
    response = generate_content_with_fallback(
        client=client,
        contents=[prompt],
    )

    raw = (response.text or "").strip()

    # Strip accidental markdown fences (```json ... ```)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)  # raises json.JSONDecodeError if malformed

    if "alternative_name" not in parsed or "reasons" not in parsed:
        raise ValueError(f"Missing required keys in Gemini response: {parsed}")
    if not isinstance(parsed["reasons"], list) or len(parsed["reasons"]) == 0:
        raise ValueError(f"'reasons' must be a non-empty list: {parsed}")

    return {
        "alternative_name": str(parsed["alternative_name"]).strip(),
        "reasons": [str(r).strip() for r in parsed["reasons"][:3]],
    }


@app.post("/api/suggest-alternative")
async def suggest_alternative(payload: SuggestAlternativeRequest, user: models.User = Depends(get_current_user)):
    """
    Calls Gemini to suggest a healthier packaged food alternative available in India.

    - Retries once if the first response is not valid JSON.
    - Falls back to a generic message if both attempts fail or GEMINI_API_KEY is absent.
    - Never raises an HTTP 5xx — always returns a usable payload.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print(
            "\n*** WARNING ****************************************************\n"
            "MOCK MODE - no real Gemini API key found (GEMINI_API_KEY unset)\n"
            "/api/suggest-alternative returning generic fallback.          \n"
            "Set GEMINI_API_KEY in backend/.env before deploying to prod.  \n"
            "****************************************************************\n"
        )
        return _ALTERNATIVE_FALLBACK

    client = genai.Client(api_key=gemini_api_key)

    last_error: Exception | None = None
    for attempt in range(1, 3):  # attempt 1, then retry attempt 2
        try:
            result = _call_gemini_for_alternative(client, payload.product_guess, payload.ingredients, language=payload.language)
            print(f"[SUGGEST-ALTERNATIVE] Success on attempt {attempt}.")
            return result
        except (json.JSONDecodeError, ValueError) as parse_err:
            last_error = parse_err
            print(f"[SUGGEST-ALTERNATIVE] Attempt {attempt} returned invalid JSON: {parse_err}. "
                  + ("Retrying..." if attempt < 2 else "Falling back to generic message."))
        except Exception as api_err:
            last_error = api_err
            print(f"[SUGGEST-ALTERNATIVE] Gemini API error on attempt {attempt}: {api_err}. "
                  + ("Retrying..." if attempt < 2 else "Falling back to generic message."))

    # Both attempts failed — return graceful fallback, never crash
    print(f"[SUGGEST-ALTERNATIVE] All retries exhausted. Last error: {last_error}")
    return _ALTERNATIVE_FALLBACK


# ──────────────────────────────────────────────────────────────────────
# Phase 8: Brand/Product Search & Category Leaderboard Fallbacks & Routes
# ──────────────────────────────────────────────────────────────────────

_CATEGORY_MOCK_DATA = {
    'biscuits': [{'name': 'Britannia Marie Gold', 'brand': 'Britannia', 'image_thumb_url': '', 'code': '8901063015395', 'ingredients_text': 'Wheat Flour, Sugar, Refined Palm Oil, Invert Sugar Syrup, Milk Solids, Iodised Salt, Raising Agents (INS 500ii, INS 503ii), Emulsifiers (Soy Lecithin, INS 471), Citric Acid'}, {'name': "McVitie's Digestive", 'brand': "McVitie's", 'image_thumb_url': '', 'code': '5000168001007', 'ingredients_text': 'Wheat Flour, Whole Wheat Flour, Vegetable Oil, Sugar, Malt Extract, Invert Sugar Syrup, Raising Agents, Salt, Malic Acid'}, {'name': 'Oreo Chocolate Creme Biscuits', 'brand': 'Oreo', 'image_thumb_url': '', 'code': '7622210449283', 'ingredients_text': 'Refined Wheat Flour, Sugar, Refined Palm Oil, Cocoa Powder, High Fructose Corn Syrup, Raising Agents, Salt, Soy Lecithin, Artificial Flavouring Substance'}, {'name': 'Sunfeast Dark Fantasy Choco Fills', 'brand': 'Sunfeast', 'image_thumb_url': '', 'code': '8901725181225', 'ingredients_text': 'Refined Wheat Flour, Choco Filling (Sugar, Refined Palm Oil, Cocoa Solids, Soy Lecithin), Refined Palm Oil, Sugar, Cocoa Solids, Invert Sugar Syrup, Liquid Glucose, Raising Agents, Salt, Soy Lecithin, Artificial Vanilla Flavour'}],
    'cookies': [{'name': 'Good Day Cashew Cookies', 'brand': 'Britannia', 'image_thumb_url': '', 'code': '8901063016255', 'ingredients_text': 'Wheat Flour, Sugar, Palm Oil, Cashew Bits, Invert Sugar Syrup, Butter, Iodised Salt, Emulsifiers, Raising Agents'}, {'name': 'Unibic Chocofills Cookies', 'brand': 'Unibic', 'image_thumb_url': '', 'code': '8906002213764', 'ingredients_text': 'Wheat Flour, Sugar, Hydrogenated Vegetable Oil, Cocoa Powder, Milk Solids, Emulsifiers, Raising Agents'}, {'name': 'Parle-G Gold Biscuits', 'brand': 'Parle', 'image_thumb_url': '', 'code': '8901719102434', 'ingredients_text': 'Wheat Flour, Sugar, Refined Palm Oil, Invert Sugar Syrup, Milk Solids, Iodised Salt, Raising Agents, Emulsifiers'}],
    'potato-chips': [{'name': "Lay's Classic Salted", 'brand': "Lay's", 'image_thumb_url': '', 'code': '8901491101838', 'ingredients_text': 'Potato, Refined Palm Oil, Salt'}, {'name': 'Pringles Sour Cream & Onion', 'brand': 'Pringles', 'image_thumb_url': '', 'code': '3890000159985', 'ingredients_text': 'Dried Potatoes, Vegetable Oil, Corn Flour, Wheat Starch, Maltodextrin, Emulsifier (INS 471), Salt, Whey, Dextrose, Monosodium Glutamate, Coconut Oil, Yeast Extract, Citric Acid'}, {'name': 'Too Yumm Karare Veggie Stix', 'brand': 'Too Yumm', 'image_thumb_url': '', 'code': '8906090570275', 'ingredients_text': 'Rice Flour, Corn Flour, Palm Oil, Spices, Salt, Acidity Regulator (INS 330), Stabilizer (INS 412)'}],
    'chips': [{'name': "Lay's Classic Salted", 'brand': "Lay's", 'image_thumb_url': '', 'code': '8901491101838', 'ingredients_text': 'Potato, Refined Palm Oil, Salt'}, {'name': 'Balaji Wafers Simply Salted', 'brand': 'Balaji', 'image_thumb_url': '', 'code': '8906007410052', 'ingredients_text': 'Potatoes, Edible Vegetable Oil, Iodised Salt'}, {'name': "Haldiram's Alooo Bhujia", 'brand': 'Haldiram', 'image_thumb_url': '', 'code': '8904063200111', 'ingredients_text': 'Potatoes, Tepary Beans Flour, Edible Vegetable Oil, Chickpea Flour, Salt, Coriander Powder, Cumin Powder, Ginger Powder'}],
    'chocolates': [{'name': 'Amul Dark Chocolate', 'brand': 'Amul', 'image_thumb_url': '', 'code': '8901262070400', 'ingredients_text': 'Sugar, Cocoa Solids, Cocoa Butter, Permitted Emulsifiers (INS 322, INS 476), Ethyl Vanillin'}, {'name': 'Cadbury Dairy Milk', 'brand': 'Cadbury', 'image_thumb_url': '', 'code': '7622201140168', 'ingredients_text': 'Sugar, Milk Solids, Cocoa Butter, Cocoa Solids, Emulsifiers (INS 442, INS 476), Ethyl Vanillin'}, {'name': 'Nestle KitKat Share Bag', 'brand': 'Nestle', 'image_thumb_url': '', 'code': '8901058863611', 'ingredients_text': 'Milk Chocolate (Sugar, Milk Solids, Cocoa Butter, Cocoa Solids, Emulsifiers), Wafer (Refined Wheat Flour, Sugar, Hydrogenated Vegetable Fats)'}],
    'ketchups': [{'name': 'Heinz Tomato Ketchup', 'brand': 'Heinz', 'image_thumb_url': '', 'code': '0013000006403', 'ingredients_text': 'Tomato Concentrate from Red Ripe Tomatoes, Distilled Vinegar, High Fructose Corn Syrup, Salt, Spice, Onion Powder, Natural Flavoring'}, {'name': 'Maggi Tomato Ketchup', 'brand': 'Maggi', 'image_thumb_url': '', 'code': '8901058002287', 'ingredients_text': 'Tomato Paste, Sugar, Water, Salt, Acidity Regulator (INS 260), Thickener (INS 415), Preservative (INS 211), Onion, Garlic, Spices'}, {'name': 'Kissan Fresh Tomato Ketchup', 'brand': 'Kissan', 'image_thumb_url': '', 'code': '8901030341859', 'ingredients_text': 'Tomato Paste, Water, Sugar, Liquid Glucose, Salt, Acidity Regulator (INS 260), Thickeners (INS 1422, INS 415), Preservative (INS 211), Onion Powder, Garlic Powder, Spices'}],
    'ketchup': [{'name': 'Heinz Tomato Ketchup', 'brand': 'Heinz', 'image_thumb_url': '', 'code': '0013000006403', 'ingredients_text': 'Tomato Concentrate, Distilled Vinegar, High Fructose Corn Syrup, Salt, Spice'}, {'name': 'Maggi Tomato Ketchup', 'brand': 'Maggi', 'image_thumb_url': '', 'code': '8901058002287', 'ingredients_text': 'Tomato Paste, Sugar, Water, Salt, Acidity Regulator (INS 260)'}],
    'fruit-juices': [{'name': 'Real Fruit Power Orange Juice', 'brand': 'Dabur', 'image_thumb_url': '', 'code': '8901207000394', 'ingredients_text': 'Water, Orange Juice Concentrate, Sugar, Acidity Regulator (INS 330), Stabilizer (INS 440), Vitamin C'}, {'name': 'Tropicana 100% Orange Juice', 'brand': 'Tropicana', 'image_thumb_url': '', 'code': '8902083001086', 'ingredients_text': 'Orange Juice Concentrate, Water'}, {'name': 'Raw Pressery Valencia Orange Juice', 'brand': 'Raw Pressery', 'image_thumb_url': '', 'code': '8906067030146', 'ingredients_text': 'Valencia Orange Juice'}],
    'juice': [{'name': 'Real Orange Juice', 'brand': 'Dabur', 'image_thumb_url': '', 'code': '8901207000394', 'ingredients_text': 'Water, Orange Juice Concentrate, Sugar, Acidity Regulator (INS 330)'}, {'name': 'Minute Maid Pulpy Orange', 'brand': 'Minute Maid', 'image_thumb_url': '', 'code': '8901764061151', 'ingredients_text': 'Water, Sugar, Orange Juice Concentrate, Orange Pulp, Acidity Regulator (INS 330)'}],
    'instant-noodles': [{'name': 'Maggi 2-Minute Masala Noodles', 'brand': 'Maggi', 'image_thumb_url': '', 'code': '8901058895476', 'ingredients_text': 'Wheat Flour, Palm Oil, Salt, Wheat Gluten, Potassium Chloride, Gelling Agent (INS 508), Raising Agent (INS 500ii), Humectant (INS 451i), Coriander, Chilli, Turmeric, Cumin, Aniseed, Fenugreek, Ginger, Black Pepper, Clove, Nutmeg, Cardamom, Monosodium Glutamate, Citric Acid'}, {'name': 'Yippee Magic Masala Noodles', 'brand': 'Sunfeast', 'image_thumb_url': '', 'code': '8901725198124', 'ingredients_text': 'Wheat Flour, Refined Palm Oil, Iodised Salt, Wheat Gluten, Calcium Carbonate, Gelling Agent (INS 508), Humectant (INS 451i), Spices, Sugar, Garlic Powder, Onion Powder, Monosodium Glutamate, Yeast Extract'}, {'name': "Ching's Secret Schezwan Instant Noodles", 'brand': "Ching's", 'image_thumb_url': '', 'code': '8901595861423', 'ingredients_text': 'Wheat Flour, Refined Palm Oil, Salt, Gelling Agent (INS 508), Wheat Gluten, Sodium Polyphosphate, Soy Sauce, Schezwan Seasoning'}],
    'noodles': [{'name': 'Maggi 2-Minute Masala Noodles', 'brand': 'Maggi', 'image_thumb_url': '', 'code': '8901058895476', 'ingredients_text': 'Wheat Flour, Palm Oil, Salt, Wheat Gluten, Spices, Monosodium Glutamate'}, {'name': 'Nissin Top Ramen Curry Noodles', 'brand': 'Top Ramen', 'image_thumb_url': '', 'code': '8901063142145', 'ingredients_text': 'Wheat Flour, Edible Vegetable Oil, Iodised Salt, Wheat Gluten, Spices, Thickener, Acidity Regulator'}],
    'pastas': [{'name': 'Bambino Macaroni Pasta', 'brand': 'Bambino', 'image_thumb_url': '', 'code': '8901248100075', 'ingredients_text': 'Semolina, Durum Wheat Semolina'}, {'name': 'YiPPee Tricolor Pasta Masala', 'brand': 'Sunfeast', 'image_thumb_url': '', 'code': '8901725103463', 'ingredients_text': 'Semolina, Wheat Gluten, Spices, Sugar, Iodised Salt, Dehydrated Vegetables, Hydrolyzed Vegetable Protein, Citric Acid'}, {'name': 'Maggi Pazzta Cheese Macaroni', 'brand': 'Maggi', 'image_thumb_url': '', 'code': '8901058890044', 'ingredients_text': 'Semolina (Durum Wheat), Milk Solids, Cheese Powder, Refined Wheat Flour, Palm Oil, Salt, Sugar, Dehydrated Sweet Corn'}],
    'pasta': [{'name': 'Bambino Macaroni Pasta', 'brand': 'Bambino', 'image_thumb_url': '', 'code': '8901248100075', 'ingredients_text': 'Semolina, Durum Wheat Semolina'}, {'name': 'Maggi Pazzta Tomato Twist', 'brand': 'Maggi', 'image_thumb_url': '', 'code': '8901058890051', 'ingredients_text': 'Semolina (Durum Wheat), Tomato Paste, Sugar, Palm Oil, Iodised Salt, Dehydrated Vegetables, Spices'}],
    'breakfast-cereals': [{'name': "Kellogg's Corn Flakes", 'brand': "Kellogg's", 'image_thumb_url': '', 'code': '8901000000013', 'ingredients_text': 'Corn, Sugar, Barley Malt Extract, Iodised Salt, Vitamin C, Iron'}, {'name': "Kellogg's Chocos", 'brand': "Kellogg's", 'image_thumb_url': '', 'code': '8901000000014', 'ingredients_text': 'Wheat Flour, Sugar, Cocoa Powder, Wheat Starch, Refined Palm Oil'}, {'name': 'Saffola Masala Oats', 'brand': 'Saffola', 'image_thumb_url': '', 'code': '8901000000015', 'ingredients_text': 'Rolled Oats, Spices & Condiments, Salt, Dehydrated Vegetables, Hydrolyzed Vegetable Protein'}],
    'dairy': [{'name': 'Amul Taaza Toned Milk', 'brand': 'Amul', 'image_thumb_url': '', 'code': '8901000000014', 'ingredients_text': 'Toned Milk, Vitamin A, Vitamin D'}, {'name': 'Mother Dairy Full Cream Milk', 'brand': 'Mother', 'image_thumb_url': '', 'code': '8901000000015', 'ingredients_text': 'Full Cream Milk'}, {'name': 'Nestle a+ Slim Dahi', 'brand': 'Nestle', 'image_thumb_url': '', 'code': '8901000000016', 'ingredients_text': 'Double Toned Milk, Active Culture'}],
    'milks': [{'name': 'Amul Taaza Toned Milk', 'brand': 'Amul', 'image_thumb_url': '', 'code': '8901000000015', 'ingredients_text': 'Toned Milk, Vitamin A, Vitamin D'}, {'name': 'Mother Dairy Full Cream Milk', 'brand': 'Mother', 'image_thumb_url': '', 'code': '8901000000016', 'ingredients_text': 'Full Cream Milk'}, {'name': 'Nestle a+ Slim Dahi', 'brand': 'Nestle', 'image_thumb_url': '', 'code': '8901000000017', 'ingredients_text': 'Double Toned Milk, Active Culture'}],
    'ice-creams': [{'name': "Kwality Wall's Vanilla", 'brand': 'Kwality', 'image_thumb_url': '', 'code': '8901000000016', 'ingredients_text': 'Water, Sugar, Milk Solids, Vegetable Protein, Emulsifier, Stabilizer, Vanilla Flavour'}, {'name': 'Amul Vanilla Gold Ice Cream', 'brand': 'Amul', 'image_thumb_url': '', 'code': '8901000000017', 'ingredients_text': 'Milk Solids, Sugar, Permitted Emulsifier (INS 471), Stabilizer (INS 407), Vanilla Flavour'}, {'name': 'Havmor Chocolate Ice Cream', 'brand': 'Havmor', 'image_thumb_url': '', 'code': '8901000000018', 'ingredients_text': 'Milk Solids, Sugar, Cocoa Solids, Cocoa Butter, Emulsifier, Stabilizer'}],
    'breads': [{'name': 'Britannia 100% Whole Wheat Bread', 'brand': 'Britannia', 'image_thumb_url': '', 'code': '8901000000017', 'ingredients_text': 'Whole Wheat Flour (Atta), Yeast, Sugar, Gluten, Iodised Salt, Preservative (INS 282)'}, {'name': 'Harvest Gold Brown Bread', 'brand': 'Harvest', 'image_thumb_url': '', 'code': '8901000000018', 'ingredients_text': 'Whole Wheat Flour, Sugar, Yeast, Soya Flour, Salt, Preservative (INS 282)'}, {'name': 'English Oven White Bread', 'brand': 'English', 'image_thumb_url': '', 'code': '8901000000019', 'ingredients_text': 'Refined Wheat Flour (Maida), Yeast, Sugar, Salt, Preservatives'}],
    'spices': [{'name': 'Everest Garam Masala', 'brand': 'Everest', 'image_thumb_url': '', 'code': '8901000000018', 'ingredients_text': 'Coriander, Cumin, Black Pepper, Ginger, Cinnamon, Cardamom, Cloves'}, {'name': 'MDH Deggi Mirch', 'brand': 'MDH', 'image_thumb_url': '', 'code': '8901000000019', 'ingredients_text': 'Red Chilli Powder'}, {'name': 'Catch Turmeric Powder', 'brand': 'Catch', 'image_thumb_url': '', 'code': '8901000000020', 'ingredients_text': 'Turmeric Powder'}],
    'pickles': [{'name': "Mother's Recipe Mango Pickle", 'brand': "Mother's", 'image_thumb_url': '', 'code': '8901000000019', 'ingredients_text': 'Mango Pieces, Salt, Mustard Oil, Fenugreek, Mustard, Chilli Powder, Turmeric, Acidity Regulator (INS 260)'}, {'name': 'Priya Lime Pickle', 'brand': 'Priya', 'image_thumb_url': '', 'code': '8901000000020', 'ingredients_text': 'Lime Pieces, Salt, Sesame Oil, Chilli Powder, Mustard Powder, Fenugreek Powder, Turmeric Powder'}, {'name': "Nilon's Mixed Pickle", 'brand': "Nilon's", 'image_thumb_url': '', 'code': '8901000000021', 'ingredients_text': 'Mango, Lime, Carrot, Green Chilli, Salt, Cottonseed Oil, Spices, Acidity Regulator (INS 260)'}],
    'prepared-meals': [{'name': 'MTR Ready to Eat Paneer Butter Masala', 'brand': 'MTR', 'image_thumb_url': '', 'code': '8901000000020', 'ingredients_text': 'Water, Paneer, Tomato, Onion, Butter, Cashew Nut, Ginger, Garlic, Spices'}, {'name': 'Gits Ready to Eat Dal Makhani', 'brand': 'Gits', 'image_thumb_url': '', 'code': '8901000000021', 'ingredients_text': 'Water, Black Gram, Red Kidney Beans, Tomato, Butter, Cream, Salt, Ginger, Garlic, Spices'}, {'name': "Haldiram's Ready to Eat Rajma Masala", 'brand': "Haldiram's", 'image_thumb_url': '', 'code': '8901000000022', 'ingredients_text': 'Rajma Beans, Water, Tomato, Onion, Edible Vegetable Oil, Ginger, Garlic, Spices'}],
    'energy-drinks': [{'name': 'Red Bull Energy Drink', 'brand': 'Red', 'image_thumb_url': '', 'code': '8901000000021', 'ingredients_text': 'Water, Sucrose, Glucose, Citric Acid, Taurine, Sodium Bicarbonate, Caffeine, Niacin, B-vitamins'}, {'name': 'Monster Energy', 'brand': 'Monster', 'image_thumb_url': '', 'code': '8901000000022', 'ingredients_text': 'Carbonated Water, Sugar, Glucose, Citric Acid, Natural Flavours, Taurine, Caffeine, Ginseng Extract'}, {'name': 'Sting Energy Drink', 'brand': 'Sting', 'image_thumb_url': '', 'code': '8901000000023', 'ingredients_text': 'Carbonated Water, Sugar, Citric Acid, Taurine, Caffeine, Inositol, Preservatives, Colour'}],
    'teas': [{'name': 'Tata Tea Premium', 'brand': 'Tata', 'image_thumb_url': '', 'code': '8901000000022', 'ingredients_text': 'Tea Leaves'}, {'name': 'Taj Mahal Tea', 'brand': 'Taj', 'image_thumb_url': '', 'code': '8901000000023', 'ingredients_text': 'Assam Tea Leaves'}, {'name': 'Red Label Tea', 'brand': 'Red', 'image_thumb_url': '', 'code': '8901000000024', 'ingredients_text': 'Black Tea Leaves'}],
    'coffees': [{'name': 'Nescafe Classic', 'brand': 'Nescafe', 'image_thumb_url': '', 'code': '8901000000023', 'ingredients_text': 'Coffee Beans'}, {'name': 'Bru Instant Coffee', 'brand': 'Bru', 'image_thumb_url': '', 'code': '8901000000024', 'ingredients_text': 'Coffee Beans, Chicory'}, {'name': 'Tata Coffee Grand', 'brand': 'Tata', 'image_thumb_url': '', 'code': '8901000000025', 'ingredients_text': 'Coffee Powder, Chicory'}],
    'chutneys': [{'name': 'Kissan Sweet & Spicy Chutney', 'brand': 'Kissan', 'image_thumb_url': '', 'code': '8901000000024', 'ingredients_text': 'Sugar, Tamarind Paste, Water, Salt, Spices, Acidity Regulator (INS 260)'}, {'name': "Mother's Recipe Tamarind Chutney", 'brand': "Mother's", 'image_thumb_url': '', 'code': '8901000000025', 'ingredients_text': 'Sugar, Tamarind, Dates, Salt, Spices, Acidity Regulator'}, {'name': 'Priya Gongura Chutney', 'brand': 'Priya', 'image_thumb_url': '', 'code': '8901000000026', 'ingredients_text': 'Gongura Leaves, Sesame Oil, Salt, Chilli Powder, Garlic, Fenugreek, Mustard'}],
    'jams': [{'name': 'Kissan Mixed Fruit Jam', 'brand': 'Kissan', 'image_thumb_url': '', 'code': '8901000000025', 'ingredients_text': 'Sugar, Mixed Fruit Pulp, Thickener (INS 440), Acidity Regulator (INS 330), Preservative (INS 211)'}, {'name': 'Mapro Strawberry Jam', 'brand': 'Mapro', 'image_thumb_url': '', 'code': '8901000000026', 'ingredients_text': 'Sugar, Strawberry Pulp, Pectin, Citric Acid, Preservatives'}, {'name': "Hershey's Chocolate Syrup", 'brand': "Hershey's", 'image_thumb_url': '', 'code': '8901000000027', 'ingredients_text': 'Sugar, Water, Cocoa Powder, High Fructose Corn Syrup, Preservative (INS 202), Xanthan Gum'}],
    'dietary-supplements': [{'name': 'Horlicks Classic Malt', 'brand': 'Horlicks', 'image_thumb_url': '', 'code': '8901000000026', 'ingredients_text': 'Malted Barley, Wheat Flour, Milk Solids, Sugar, Minerals, Vitamins, Salt'}, {'name': 'Bournvita Cadbury', 'brand': 'Bournvita', 'image_thumb_url': '', 'code': '8901000000027', 'ingredients_text': 'Cereal Extract, Sugar, Cocoa Powder, Milk Solids, Raising Agents, Emulsifiers, Vitamins, Minerals'}, {'name': 'Ensure Nutritional Powder', 'brand': 'Ensure', 'image_thumb_url': '', 'code': '8901000000028', 'ingredients_text': 'Maltodextrin, Calcium Caseinate, Sucrose, Vegetable Oils, Minerals, Vitamins'}],
    'baby-foods': [{'name': 'Cerelac Wheat Apple', 'brand': 'Cerelac', 'image_thumb_url': '', 'code': '8901000000027', 'ingredients_text': 'Wheat Flour, Milk Solids, Sucrose, Apple Juice Concentrate, Soyabean Oil, Minerals, Vitamins'}, {'name': 'Farex Baby Food', 'brand': 'Farex', 'image_thumb_url': '', 'code': '8901000000028', 'ingredients_text': 'Wheat Flour, Milk Solids, Sugar, Minerals, Vitamins'}, {'name': 'Nestum Rice Baby Cereal', 'brand': 'Nestum', 'image_thumb_url': '', 'code': '8901000000029', 'ingredients_text': 'Rice Flour, Milk Solids, Sucrose, Soyabean Oil, Vitamins, Minerals'}],
    'frozen-foods': [{'name': 'McCain French Fries', 'brand': 'McCain', 'image_thumb_url': '', 'code': '8901000000028', 'ingredients_text': 'Potato, Refined Palm Oil'}, {'name': 'Safal Frozen Green Peas', 'brand': 'Safal', 'image_thumb_url': '', 'code': '8901000000029', 'ingredients_text': 'Green Peas'}, {'name': "Venky's Chicken Nuggets", 'brand': "Venky's", 'image_thumb_url': '', 'code': '8901000000030', 'ingredients_text': 'Chicken Meat, Water, Bread Crumbs, Batter, Refined Palm Oil, Spices'}],
    'canned-foods': [{'name': 'Del Monte Canned Pineapple Slices', 'brand': 'Del', 'image_thumb_url': '', 'code': '8901000000029', 'ingredients_text': 'Pineapple Slices, Water, Sugar, Citric Acid'}, {'name': 'Golden Crown Canned Mushroom', 'brand': 'Golden', 'image_thumb_url': '', 'code': '8901000000030', 'ingredients_text': 'Mushrooms, Water, Salt, Citric Acid'}, {'name': 'American Garden Canned Sweet Corn', 'brand': 'American', 'image_thumb_url': '', 'code': '8901000000031', 'ingredients_text': 'Sweet Corn, Water, Salt'}],
    'carbonated-beverages': [{'name': 'Coca-Cola Classic', 'brand': 'Coca-Cola', 'image_thumb_url': '', 'code': '8901000000030', 'ingredients_text': 'Carbonated Water, Sugar, Acidity Regulator (INS 338), Caffeine, Natural Colour (INS 150d)'}, {'name': 'Pepsi Cola', 'brand': 'Pepsi', 'image_thumb_url': '', 'code': '8901000000031', 'ingredients_text': 'Carbonated Water, Sugar, Acidity Regulator, Caffeine, Colour'}, {'name': 'Sprite Lemon Lime', 'brand': 'Sprite', 'image_thumb_url': '', 'code': '8901000000032', 'ingredients_text': 'Carbonated Water, Sugar, Acidity Regulators (INS 330, INS 331), Preservative (INS 211)'}],
    'candies': [{'name': 'Pulse Candy Mango', 'brand': 'Pulse', 'image_thumb_url': '', 'code': '8901000000031', 'ingredients_text': 'Sugar, Liquid Glucose, Amchur Powder, Citric Acid, Food Colours'}, {'name': 'Alpenliebe Gold Candy', 'brand': 'Alpenliebe', 'image_thumb_url': '', 'code': '8901000000032', 'ingredients_text': 'Sugar, Liquid Glucose, Condensed Milk, Butter, Salt, Emulsifier, Flavours'}, {'name': 'Orbit Sugarfree Chewing Gum', 'brand': 'Orbit', 'image_thumb_url': '', 'code': '8901000000033', 'ingredients_text': 'Sorbitol, Gum Base, Xylitol, Mannitol, Humectant (INS 422), Sweeteners (Aspartame, Acesulfame K)'}],
    'papads': [{'name': 'Lijjat Udidad Papad', 'brand': 'Lijjat', 'image_thumb_url': '', 'code': '8901000000032', 'ingredients_text': 'Udad Dal Flour, Black Pepper, Salt, Raising Agent (INS 500ii), Cottonseed Oil'}, {'name': "Mother's Recipe Plain Papad", 'brand': "Mother's", 'image_thumb_url': '', 'code': '8901000000033', 'ingredients_text': 'Udad Dal Flour, Salt, Asafoetida, Papad Khar'}, {'name': "Nilon's Moong Papad", 'brand': "Nilon's", 'image_thumb_url': '', 'code': '8901000000034', 'ingredients_text': 'Moong Dal Flour, Udad Dal Flour, Black Pepper, Salt, Spices'}],
    'nuts': [{'name': 'Tata Sampann Almonds', 'brand': 'Tata', 'image_thumb_url': '', 'code': '8901000000033', 'ingredients_text': 'Almonds'}, {'name': 'Happilo California Walnuts', 'brand': 'Happilo', 'image_thumb_url': '', 'code': '8901000000034', 'ingredients_text': 'Walnuts'}, {'name': 'Delinut Cashews salted', 'brand': 'Delinut', 'image_thumb_url': '', 'code': '8901000000035', 'ingredients_text': 'Cashews, Salt, Gum Arabic'}],
    'honeys': [{'name': 'Dabur Honey', 'brand': 'Dabur', 'image_thumb_url': '', 'code': '8901000000034', 'ingredients_text': 'Honey'}, {'name': 'Patanjali Pure Honey', 'brand': 'Patanjali', 'image_thumb_url': '', 'code': '8901000000035', 'ingredients_text': 'Honey'}, {'name': 'Saffola Wild Forest Honey', 'brand': 'Saffola', 'image_thumb_url': '', 'code': '8901000000036', 'ingredients_text': 'Honey'}],
    'cooking-oils': [{'name': 'Fortune Mustard Oil', 'brand': 'Fortune', 'image_thumb_url': '', 'code': '8901000000035', 'ingredients_text': 'Mustard Oil'}, {'name': 'Saffola Gold Cooking Oil', 'brand': 'Saffola', 'image_thumb_url': '', 'code': '8901000000036', 'ingredients_text': 'Refined Rice Bran Oil, Refined Sunflower Oil, Antioxidant (INS 319)'}, {'name': 'Fortune Refined Sunflower Oil', 'brand': 'Fortune', 'image_thumb_url': '', 'code': '8901000000037', 'ingredients_text': 'Refined Sunflower Oil'}],
    'ghee': [{'name': 'Amul Pure Ghee', 'brand': 'Amul', 'image_thumb_url': '', 'code': '8901000000036', 'ingredients_text': 'Milk Fat'}, {'name': 'Patamjali Cow Ghee', 'brand': 'Patamjali', 'image_thumb_url': '', 'code': '8901000000037', 'ingredients_text': 'Cow Milk Fat'}, {'name': 'Gowardhan Pure Ghee', 'brand': 'Gowardhan', 'image_thumb_url': '', 'code': '8901000000038', 'ingredients_text': 'Milk Fat'}],
    'flours': [{'name': 'Aashirvaad Shudh Chakki Atta', 'brand': 'Aashirvaad', 'image_thumb_url': '', 'code': '8901000000037', 'ingredients_text': 'Whole Wheat'}, {'name': 'Pillsbury Chakki Fresh Atta', 'brand': 'Pillsbury', 'image_thumb_url': '', 'code': '8901000000038', 'ingredients_text': 'Whole Wheat'}, {'name': 'Fortune Chakki Fresh Atta', 'brand': 'Fortune', 'image_thumb_url': '', 'code': '8901000000039', 'ingredients_text': 'Whole Wheat'}],
    'pulses': [{'name': 'Tata Sampann Toor Dal', 'brand': 'Tata', 'image_thumb_url': '', 'code': '8901000000038', 'ingredients_text': 'Toor Dal'}, {'name': 'Organic Tattva Moong Dal', 'brand': 'Organic', 'image_thumb_url': '', 'code': '8901000000039', 'ingredients_text': 'Moong Dal'}, {'name': 'Fortune Chana Dal', 'brand': 'Fortune', 'image_thumb_url': '', 'code': '8901000000040', 'ingredients_text': 'Chana Dal'}],
    'rices': [{'name': 'India Gate Basmati Rice Premium', 'brand': 'India', 'image_thumb_url': '', 'code': '8901000000039', 'ingredients_text': 'Basmati Rice'}, {'name': 'Daawat Rozana Gold Basmati Rice', 'brand': 'Daawat', 'image_thumb_url': '', 'code': '8901000000040', 'ingredients_text': 'Basmati Rice'}, {'name': 'Lal Qilla Basmati Rice', 'brand': 'Lal', 'image_thumb_url': '', 'code': '8901000000041', 'ingredients_text': 'Basmati Rice'}],
    'salts': [{'name': 'Tata Salt Iodised', 'brand': 'Tata', 'image_thumb_url': '', 'code': '8901000000040', 'ingredients_text': 'Iodised Salt, Potassium Iodate, Anticaking Agent (INS 536)'}, {'name': 'Aashirvaad Iodised Salt', 'brand': 'Aashirvaad', 'image_thumb_url': '', 'code': '8901000000041', 'ingredients_text': 'Iodised Salt, Anticaking Agent'}, {'name': 'Catch Rock Salt', 'brand': 'Catch', 'image_thumb_url': '', 'code': '8901000000042', 'ingredients_text': 'Rock Salt'}],
    'sugars': [{'name': 'Mawana Double Refined Sugar', 'brand': 'Mawana', 'image_thumb_url': '', 'code': '8901000000041', 'ingredients_text': 'Sugar'}, {'name': 'Trust Classic Sulphurless Sugar', 'brand': 'Trust', 'image_thumb_url': '', 'code': '8901000000042', 'ingredients_text': 'Sugar'}, {'name': "Parry's White Sugar", 'brand': "Parry's", 'image_thumb_url': '', 'code': '8901000000043', 'ingredients_text': 'Sugar'}],
    'vinegars': [{'name': 'Del Monte Synthetic Vinegar', 'brand': 'Del', 'image_thumb_url': '', 'code': '8901000000042', 'ingredients_text': 'Water, Acetic Acid (INS 260)'}, {'name': 'Heinz Apple Cider Vinegar', 'brand': 'Heinz', 'image_thumb_url': '', 'code': '8901000000043', 'ingredients_text': 'Apple Cider Vinegar'}, {'name': 'DiSano Apple Cider Vinegar', 'brand': 'DiSano', 'image_thumb_url': '', 'code': '8901000000044', 'ingredients_text': 'Apple Cider Vinegar, Water'}],
    'dehydrated-soups': [{'name': 'Knorr Mixed Vegetable Soup', 'brand': 'Knorr', 'image_thumb_url': '', 'code': '8901000000043', 'ingredients_text': 'Refined Wheat Flour, Corn Flour, Sugar, Salt, Dehydrated Vegetables, Thickener (INS 415), Hydrolyzed Vegetable Protein'}, {'name': "Ching's Secret Tomato Soup", 'brand': "Ching's", 'image_thumb_url': '', 'code': '8901000000044', 'ingredients_text': 'Tomato Powder, Sugar, Potato Starch, Salt, Spices, Xanthan Gum'}, {'name': 'Knorr Hot & Sour Chicken Soup', 'brand': 'Knorr', 'image_thumb_url': '', 'code': '8901000000045', 'ingredients_text': 'Refined Wheat Flour, Sugar, Salt, Dehydrated Chicken, Soy Sauce Powder, Spices'}],
    'instant-coffees': [{'name': 'Nescafe Classic', 'brand': 'Nescafe', 'image_thumb_url': '', 'code': '8901000000044', 'ingredients_text': 'Coffee Beans'}, {'name': 'Bru Gold Coffee', 'brand': 'Bru', 'image_thumb_url': '', 'code': '8901000000045', 'ingredients_text': 'Coffee Beans'}, {'name': 'Davidoff Rich Aroma Instant Coffee', 'brand': 'Davidoff', 'image_thumb_url': '', 'code': '8901000000046', 'ingredients_text': 'Coffee Beans'}],
    'protein-powders': [{'name': 'Oziva Organic Plant Protein', 'brand': 'Oziva', 'image_thumb_url': '', 'code': '8901000000045', 'ingredients_text': 'Organic Pea Protein, Organic Brown Rice Protein, Organic Quinoa'}, {'name': 'MuscleBlaze Raw Whey Protein', 'brand': 'MuscleBlaze', 'image_thumb_url': '', 'code': '8901000000046', 'ingredients_text': 'Whey Protein Concentrate'}, {'name': 'Optimum Nutrition 100% Whey Gold Standard', 'brand': 'Optimum', 'image_thumb_url': '', 'code': '8901000000047', 'ingredients_text': 'Whey Protein Isolate, Whey Protein Concentrate, Whey Peptides, Cocoa, Soy Lecithin'}],
    'snack-bars': [{'name': 'Yogabar Multigrain Energy Bar', 'brand': 'Yogabar', 'image_thumb_url': '', 'code': '8901000000046', 'ingredients_text': 'Oats, Dates, Almonds, Cashews, Honey, Sunflower Seeds, Chia Seeds'}, {'name': 'RiteBite Max Protein Daily Bar', 'brand': 'RiteBite', 'image_thumb_url': '', 'code': '8901000000047', 'ingredients_text': 'Protein Blend, Oats, Cocoa Powder, Soy Nuggets, Glycerine, Prebiotics'}, {'name': 'The Whole Truth Protein Bar', 'brand': 'The', 'image_thumb_url': '', 'code': '8901000000048', 'ingredients_text': 'Dates, Whey Protein, Cashews, Almonds, Cocoa Powder'}],
    'wafers': [{'name': 'Dukes Waffy Strawberry', 'brand': 'Dukes', 'image_thumb_url': '', 'code': '8901000000047', 'ingredients_text': 'Refined Wheat Flour, Sugar, Palm Oil, Starch, Strawberry Powder, Emulsifier (INS 322)'}, {'name': 'Amul Choco Wafers', 'brand': 'Amul', 'image_thumb_url': '', 'code': '8901000000048', 'ingredients_text': 'Refined Wheat Flour, Sugar, Hydrogenated Vegetable Oil, Cocoa Powder'}, {'name': 'Tiffany Crunch N Cream Wafers', 'brand': 'Tiffany', 'image_thumb_url': '', 'code': '8901000000049', 'ingredients_text': 'Wheat Flour, Sugar, Vegetable Fat, Milk Solids, Cocoa Powder'}],
    'yogurt': [{'name': 'Epigamia Greek Yogurt Strawberry', 'brand': 'Epigamia', 'image_thumb_url': '', 'code': '8901000000048', 'ingredients_text': 'Double Toned Milk, Strawberry Prep, Active Culture'}, {'name': 'Mother Dairy Fruit Yogurt Blueberry', 'brand': 'Mother', 'image_thumb_url': '', 'code': '8901000000049', 'ingredients_text': 'Toned Milk, Sugar, Blueberry Prep, Pectin, Active Culture'}, {'name': 'Amul Masti Spiced Buttermilk', 'brand': 'Amul', 'image_thumb_url': '', 'code': '8901000000050', 'ingredients_text': 'Water, Milk Solids, Salt, Spices, Active Culture'}],
    'butters': [{'name': 'Amul Butter Salted', 'brand': 'Amul', 'image_thumb_url': '', 'code': '8901000000049', 'ingredients_text': 'Butter, Iodised Salt, Annatto Extract'}, {'name': 'Mother Dairy Salted Butter', 'brand': 'Mother', 'image_thumb_url': '', 'code': '8901000000050', 'ingredients_text': 'Butter, Iodised Salt'}, {'name': 'Nutralite Table Spread', 'brand': 'Nutralite', 'image_thumb_url': '', 'code': '8901000000051', 'ingredients_text': 'Refined Vegetable Oils, Water, Salt, Emulsifiers, Vitamin A, Vitamin D'}],
    'cheeses': [{'name': 'Amul Cheese Slices', 'brand': 'Amul', 'image_thumb_url': '', 'code': '8901000000050', 'ingredients_text': 'Cheese, Milk Solids, Emulsifying Salts (INS 331, INS 452), Iodised Salt, Preservative (INS 200)'}, {'name': 'Britannia Cheese Cubes', 'brand': 'Britannia', 'image_thumb_url': '', 'code': '8901000000051', 'ingredients_text': 'Cheese, Water, Milk Solids, Emulsifying Salts, Salt'}, {'name': 'Mother Dairy Cheese Spread', 'brand': 'Mother', 'image_thumb_url': '', 'code': '8901000000052', 'ingredients_text': 'Cheese, Water, Milk Solids, Emulsifiers, Salt'}],
    'sauces': [{'name': 'FunFoods Eggless Mayonnaise', 'brand': 'FunFoods', 'image_thumb_url': '', 'code': '8901000000051', 'ingredients_text': 'Refined Soyabean Oil, Water, Sugar, Lemon Juice, Thickeners (INS 1422, INS 415), Salt, Acidity Regulators (INS 270, INS 330)'}, {'name': 'Veeba Pasta & Pizza Sauce', 'brand': 'Veeba', 'image_thumb_url': '', 'code': '8901000000052', 'ingredients_text': 'Tomato Paste, Water, Onion, Sugar, Liquid Glucose, Garlic, Spices, Herbs'}, {'name': "Ching's Secret Red Chilli Sauce", 'brand': "Ching's", 'image_thumb_url': '', 'code': '8901000000053', 'ingredients_text': 'Water, Red Chilli, Ginger, Garlic, Salt, Starch, Acidity Regulator (INS 260)'}],
}

def fast_score_ingredients(db: Session, ingredients_text: str) -> tuple[int, str]:
    """
    Parses and matches ingredients ONLY against local DB.
    For any unrecognized ingredient, we check common patterns or default to 'safe'
    to avoid Gemini cost/delay for category rank comparison.
    """
    raw_list = clean_and_split_ingredients(ingredients_text)
    records = db.query(models.IngredientReference).all()
    
    results = []
    for name in raw_list:
        name_clean = name.strip()
        name_lower = name_clean.lower()
        query_e_codes = extract_e_numbers(name_lower)
        
        best_status = None
        for r in records:
            record_e_codes = extract_e_numbers(r.common_alias_names or "")
            if query_e_codes and query_e_codes.intersection(record_e_codes):
                best_status = r.safety_status
                break
            if name_lower == r.ingredient_name.lower():
                best_status = r.safety_status
                break
                
        if not best_status:
            # Check simple string rules
            if any(x in name_lower for x in ["sugar", "dextrose", "fructose", "syrup", "glucose"]):
                best_status = "moderate"
            elif any(x in name_lower for x in ["palm oil", "oil", "fat"]):
                best_status = "moderate"
            elif any(x in name_lower for x in ["hydrogenated", "trans fat"]):
                best_status = "unsafe"
            elif any(x in name_lower for x in ["salt", "sodium chloride"]):
                best_status = "safe"
            elif any(x in name_lower for x in ["msg", "glutamate"]):
                best_status = "moderate"
            else:
                best_status = "safe"
                
        results.append({"safety_status": best_status})
        
    score, label = calculate_health_score(results)
    return score, label


@app.get("/api/search-product")
async def search_product(query: str, user: models.User = Depends(get_current_user)):
    """
    Searches Open Food Facts by product or brand name.
    """
    if not query.strip():
        return {"products": []}
    
    encoded_query = urllib.parse.quote(query.strip())
    # Try querying the OFF search.pl CGI endpoint with 5-second timeout and browser-like user agent
    url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={encoded_query}&search_simple=1&action=process&json=1&page_size=20"
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LabelLensApp/1.0 (contact@labellens.org) Python-urllib/3.x",
            "Accept": "application/json"
        }
    )
    
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=5.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                products = data.get("products", [])
                
                results = []
                for p in products:
                    code = p.get("code")
                    name = p.get("product_name") or p.get("product_name_en") or p.get("product_name_fr")
                    brand = p.get("brands") or p.get("brand")
                    image = p.get("image_thumb_url") or p.get("image_small_url") or p.get("image_url")
                    
                    if not code or not name:
                        continue
                        
                    results.append({
                        "code": str(code),
                        "name": str(name),
                        "brand": str(brand) if brand else "Unknown Brand",
                        "image": str(image) if image else ""
                    })
                return {"products": results}
    except Exception as e:
        print(f"[SEARCH] OFF search query '{query}' failed: {e}")
        
    return {"products": []}


@app.get("/api/analyze-barcode")
async def analyze_barcode(code: str, language: str = "en", db = Depends(get_db), user: models.User = Depends(get_current_user)):
    """
    Retrieves ingredients for a barcode, runs the 3-layer analysis + scoring pipeline,
    and returns identical structure to /api/parse-ingredients.
    """
    if not code.strip():
        raise HTTPException(status_code=400, detail="Barcode is required.")
        
    url = f"https://world.openfoodfacts.org/api/v2/product/{urllib.parse.quote(code.strip())}.json"
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LabelLensApp/1.0 (contact@labellens.org) Python-urllib/3.x",
            "Accept": "application/json"
        }
    )
    
    ingredients_text = ""
    product_guess = "Packaged Food Product"
    product_name = ""
    image_url = ""
    
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=5.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                product = data.get("product", {})
                ingredients_text = product.get("ingredients_text") or product.get("ingredients_text_en") or ""
                product_name = product.get("product_name") or product.get("product_name_en") or ""
                image_url = product.get("image_url") or product.get("image_thumb_url") or ""
                if product_name:
                    product_guess = product_name
    except Exception as e:
        print(f"[BARCODE] Fetch for barcode '{code}' failed: {e}")
        raise HTTPException(
            status_code=502, 
            detail="Could not retrieve product from Open Food Facts. Please try again later."
        )
        
    if not ingredients_text.strip():
        # Search the mock data for matching code to support demo/mock fallback testing
        found_in_mock = False
        for category, mock_list in _CATEGORY_MOCK_DATA.items():
            for mp in mock_list:
                if mp["code"] == code.strip():
                    ingredients_text = mp["ingredients_text"]
                    product_guess = mp["name"]
                    image_url = mp["image_thumb_url"]
                    found_in_mock = True
                    break
            if found_in_mock:
                break
                
        if not found_in_mock:
            raise HTTPException(
                status_code=400,
                detail=f"This product ({product_name or code}) does not have an ingredients list in Open Food Facts."
            )
        
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        class MockClient:
            is_mock = True
        client = MockClient()
    else:
        client = genai.Client(api_key=gemini_api_key)
        
    try:
        raw_list = clean_and_split_ingredients(ingredients_text)
        results = []
        for name in raw_list:
            resolved = lookup_ingredient(db, client, name, language=language)
            results.append({
                "ingredient": resolved["ingredient"],
                "safety_status": resolved["safety_status"],
                "reason": resolved["reason"],
                "simple_explanation": resolved["simple_explanation"],
                "allergen": resolved["allergen"],
                "safe_frequency": resolved["safe_frequency"],
                "source": resolved["source"]
            })
            
        overall_score, score_label = calculate_health_score(results)
        
        if not product_guess or len(product_guess.strip()) < 3:
            product_guess = guess_product_from_ingredients(results)
            
        # Save to history (user is always authenticated now)
        try:
            hist = models.ScanHistory(
                user_id=user.id,
                product_name=product_guess,
                product_guess=product_guess,
                overall_score=overall_score,
                score_label=score_label,
                ingredients_data=results
            )
            db.add(hist)
            db.commit()
            print(f"[HISTORY] Successfully saved barcode scan for user {user.email}")
        except Exception as hist_err:
            db.rollback()
            print(f"[HISTORY] Error saving barcode scan: {hist_err}")
                
        return {
            "ingredients": results,
            "overall_score": overall_score,
            "score_label": score_label,
            "product_guess": product_guess,
            "image_url": image_url
        }
    except Exception as err:
        print(f"Error parsing barcode ingredients: {err}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze barcode ingredients: {str(err)}"
        )


@app.get("/api/category-best")
async def get_category_best(category: str, db = Depends(get_db), user: models.User = Depends(get_current_user)):
    """
    Ranked leaderboard of products in a category, sorted by health score.
    """
    cat_normalized = category.lower().strip()
    
    OFF_CATEGORY_MAP = {
        "biscuits": "biscuits",
        "ketchup": "ketchups",
        "juice": "fruit-juices",
        "chips": "potato-chips",
        "chocolates": "chocolates"
    }
    off_cat = OFF_CATEGORY_MAP.get(cat_normalized, cat_normalized)
    
    url = f"https://world.openfoodfacts.org/api/v2/search?categories_tags={urllib.parse.quote(off_cat)}&page_size=20&fields=code,product_name,brands,image_thumb_url,ingredients_text"
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LabelLensApp/1.0 (contact@labellens.org) Python-urllib/3.x",
            "Accept": "application/json"
        }
    )
    
    products = []
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=5.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                products = data.get("products", [])
    except Exception as e:
        print(f"[CATEGORY] OFF category search '{off_cat}' failed: {e}. Falling back to mock data.")
        
    if not products:
        mock_key = cat_normalized
        if mock_key.startswith("en:"):
            mock_key = mock_key[3:]
        products = _CATEGORY_MOCK_DATA.get(mock_key, [])
        
    ranked_results = []
    for p in products:
        code = p.get("code")
        name = p.get("product_name") or p.get("product_name_en") or p.get("product_name_fr")
        brand = p.get("brands") or p.get("brand") or "Unknown Brand"
        image = p.get("image_thumb_url") or p.get("image_small_url") or p.get("image_url") or ""
        ingredients_text = p.get("ingredients_text") or ""
        
        if not code or not name:
            continue
            
        if not ingredients_text.strip():
            continue
            
        score, label = fast_score_ingredients(db, ingredients_text)
        
        ranked_results.append({
            "code": str(code),
            "name": str(name),
            "brand": str(brand),
            "image": str(image),
            "score": score,
            "label": label
        })
        
    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    
    for idx, item in enumerate(ranked_results):
        item["rank"] = idx + 1
        
    return {"category": category, "products": ranked_results}


# ──────────────────────────────────────────────────────────────────────
# Phase 9: Signup, Login, Scan History, and Favorites Endpoints
# ──────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class FavoriteRequest(BaseModel):
    product_name: str
    product_guess: str
    overall_score: int | None = None
    ingredients_data: list[dict] = []


@app.post("/api/auth/signup")
async def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    """
    User registration endpoint. Validates unique email, hashes password, 
    and returns a JWT token.
    """
    email_clean = payload.email.strip().lower()
    if not payload.name.strip() or not email_clean or not payload.password:
        raise HTTPException(status_code=400, detail="Name, email, and password are required.")
        
    existing_user = db.query(models.User).filter(models.User.email == email_clean).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email is already registered.")
        
    hashed = hash_password(payload.password)
    user = models.User(
        name=payload.name.strip(),
        email=email_clean,
        hashed_password=hashed
    )
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error during user signup: {e}")
        
    token = create_access_token(data={"sub": user.email})
    return {"access_token": token, "token_type": "bearer", "name": user.name}


@app.post("/api/auth/login")
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """
    User login endpoint. Verifies credentials and returns a JWT token.
    """
    email_clean = payload.email.strip().lower()
    user = db.query(models.User).filter(models.User.email == email_clean).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password.")
        
    token = create_access_token(data={"sub": user.email})
    return {"access_token": token, "token_type": "bearer", "name": user.name}


@app.get("/api/auth/me")
async def get_me(user: models.User = Depends(get_current_user)):
    """
    Protected route that returns current user's profile info.
    """
    return {"name": user.name, "email": user.email}


@app.get("/api/history")
async def get_history(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Protected route that returns the user's past scans.
    """
    scans = db.query(models.ScanHistory).filter(models.ScanHistory.user_id == user.id).order_by(models.ScanHistory.scanned_at.desc()).all()
    results = []
    for s in scans:
        results.append({
            "id": s.id,
            "product_name": s.product_name,
            "product_guess": s.product_guess,
            "overall_score": s.overall_score,
            "score_label": s.score_label,
            "ingredients_data": s.ingredients_data,
            "scanned_at": s.scanned_at.isoformat() if s.scanned_at else None
        })
    return {"history": results}


@app.post("/api/favorites")
async def add_favorite(payload: FavoriteRequest, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Protected route to save a scan result to favorites.
    """
    # Check if already exists in favorites
    existing = db.query(models.Favorite).filter(
        models.Favorite.user_id == user.id,
        models.Favorite.product_name == payload.product_name
    ).first()
    if existing:
        return {"status": "already_favorite", "id": existing.id}
        
    fav = models.Favorite(
        user_id=user.id,
        product_name=payload.product_name,
        product_guess=payload.product_guess,
        overall_score=payload.overall_score,
        ingredients_data=payload.ingredients_data
    )
    db.add(fav)
    try:
        db.commit()
        db.refresh(fav)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error saving favorite: {e}")
        
    return {"status": "success", "id": fav.id}


@app.get("/api/favorites")
async def get_favorites(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Protected route to retrieve all favorited products of the user.
    """
    favs = db.query(models.Favorite).filter(models.Favorite.user_id == user.id).order_by(models.Favorite.added_at.desc()).all()
    results = []
    for f in favs:
        results.append({
            "id": f.id,
            "product_name": f.product_name,
            "product_guess": f.product_guess,
            "overall_score": f.overall_score,
            "ingredients_data": f.ingredients_data,
            "added_at": f.added_at.isoformat() if f.added_at else None
        })
    return {"favorites": results}


@app.delete("/api/favorites/{fav_id}")
async def delete_favorite(fav_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Protected route to delete a favorite by its ID.
    """
    fav = db.query(models.Favorite).filter(
        models.Favorite.id == fav_id,
        models.Favorite.user_id == user.id
    ).first()
    if not fav:
        raise HTTPException(status_code=404, detail="Favorite item not found.")
        
    db.delete(fav)
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error deleting favorite: {e}")
        
    return {"status": "success"}


# ──────────────────────────────────────────────────────────────────────
# Phase 11: Dynamic Categories & Standalone Ingredient Search Endpoints
# ──────────────────────────────────────────────────────────────────────

_categories_cache = {
    "data": None,
    "expiry": 0.0
}

@app.get("/api/categories")
async def get_categories(user: models.User = Depends(get_current_user)):
    """
    Fetches the full list of available food categories from the static Open Food Facts taxonomy,
    caching the result in-memory for 24 hours.
    Retries once after a 2-second delay if the request fails.
    Falls back to a comprehensive list of 46 mock categories if the API remains unavailable.
    """
    global _categories_cache
    now = time.time()
    
    # 1. Return from cache if valid
    if _categories_cache["data"] and now < _categories_cache["expiry"]:
        return {"categories": _categories_cache["data"]}
        
    # 2. Fetch from static Open Food Facts taxonomy endpoint
    url = "https://static.openfoodfacts.org/data/taxonomies/categories.json"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LabelLensApp/1.0 (contact@labellens.org) Python-urllib/3.x",
            "Accept": "application/json"
        }
    )
    
    data = None
    for attempt in range(1, 3):
        try:
            print(f"[CATEGORIES] Fetching categories taxonomy from static.openfoodfacts.org (Attempt {attempt})...")
            with urllib.request.urlopen(req, context=ctx, timeout=12.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode('utf-8'))
                    break
        except Exception as e:
            print(f"[CATEGORIES] Attempt {attempt} failed: {e}")
            if attempt == 1:
                print("[CATEGORIES] Retrying in 2 seconds...")
                time.sleep(2.0)
                
    if data:
        try:
            filtered = []
            for k, v in data.items():
                if k.startswith("en:") and isinstance(v, dict):
                    name_dict = v.get("name")
                    if isinstance(name_dict, dict):
                        eng_name = name_dict.get("en")
                        if eng_name:
                            filtered.append({
                                "id": k,
                                "name": eng_name,
                                "products": 100
                            })
            
            # Sort alphabetically
            filtered.sort(key=lambda x: x["name"].lower())
            
            # Cache for 24 hours
            _categories_cache["data"] = filtered
            _categories_cache["expiry"] = now + (24 * 3600)
            
            print(f"[CATEGORIES] Loaded and filtered {len(filtered)} categories from static taxonomy API.")
            return {"categories": filtered}
        except Exception as parse_err:
            print(f"[CATEGORIES] Parsing static taxonomy failed: {parse_err}")
            
    # 3. Fallback to mock categories (46 common Indian food categories) if the live call fails
    fallback_categories = [
        {"id": "en:biscuits", "name": "Biscuits", "products": 500},
        {"id": "en:cookies", "name": "Cookies", "products": 500},
        {"id": "en:potato-chips", "name": "Chips & Crisps", "products": 500},
        {"id": "en:chocolates", "name": "Chocolates", "products": 500},
        {"id": "en:ketchups", "name": "Ketchup & Sauces", "products": 500},
        {"id": "en:fruit-juices", "name": "Juices & Beverages", "products": 500},
        {"id": "en:instant-noodles", "name": "Instant Noodles", "products": 500},
        {"id": "en:pastas", "name": "Pasta", "products": 500},
        {"id": "en:breakfast-cereals", "name": "Breakfast Cereals", "products": 500},
        {"id": "en:milks", "name": "Dairy Products", "products": 500},
        {"id": "en:ice-creams", "name": "Ice Cream", "products": 500},
        {"id": "en:breads", "name": "Bread & Bakery", "products": 500},
        {"id": "en:spices", "name": "Spices & Masalas", "products": 500},
        {"id": "en:pickles", "name": "Pickles & Achaar", "products": 500},
        {"id": "en:prepared-meals", "name": "Ready-to-Eat Meals", "products": 500},
        {"id": "en:energy-drinks", "name": "Energy Drinks", "products": 500},
        {"id": "en:teas", "name": "Tea", "products": 500},
        {"id": "en:coffees", "name": "Coffee", "products": 500},
        {"id": "en:chutneys", "name": "Chutneys", "products": 500},
        {"id": "en:jams", "name": "Jams & Spreads", "products": 500},
        {"id": "en:dietary-supplements", "name": "Health Drinks & Supplements", "products": 500},
        {"id": "en:baby-foods", "name": "Baby Food", "products": 500},
        {"id": "en:frozen-foods", "name": "Frozen Foods", "products": 500},
        {"id": "en:canned-foods", "name": "Canned Foods", "products": 500},
        {"id": "en:carbonated-beverages", "name": "Soft Drinks", "products": 500},
        {"id": "en:candies", "name": "Candy & Chewing Gum", "products": 500},
        {"id": "en:papads", "name": "Papad", "products": 500},
        {"id": "en:nuts", "name": "Dry Fruits & Nuts", "products": 500},
        {"id": "en:honeys", "name": "Honey", "products": 500},
        {"id": "en:cooking-oils", "name": "Cooking Oil", "products": 500},
        {"id": "en:ghee", "name": "Ghee", "products": 500},
        {"id": "en:flours", "name": "Atta & Flour", "products": 500},
        {"id": "en:pulses", "name": "Pulses & Dal", "products": 500},
        {"id": "en:rices", "name": "Rice", "products": 500},
        {"id": "en:salts", "name": "Salt", "products": 500},
        {"id": "en:sugars", "name": "Sugar", "products": 500},
        {"id": "en:vinegars", "name": "Vinegar", "products": 500},
        {"id": "en:dehydrated-soups", "name": "Soup Mixes", "products": 500},
        {"id": "en:instant-coffees", "name": "Instant Coffee", "products": 500},
        {"id": "en:protein-powders", "name": "Protein Powders", "products": 500},
        {"id": "en:snack-bars", "name": "Energy Bars", "products": 500},
        {"id": "en:wafers", "name": "Wafers", "products": 500},
        {"id": "en:yogurt", "name": "Yogurt & Dahi", "products": 500},
        {"id": "en:butters", "name": "Butter", "products": 500},
        {"id": "en:cheeses", "name": "Cheese", "products": 500},
        {"id": "en:sauces", "name": "Sauces & Dressings", "products": 500}
    ]
    print(f"[CATEGORIES] Returning {len(fallback_categories)} fallback categories (OFF API unavailable).")
    return {"categories": fallback_categories}


@app.get("/api/search-ingredient")
async def search_ingredient(query: str, language: str = 'en', db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    """
    Looks up a specific ingredient by name using the 3-layer lookup system.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query parameter is required")
        
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        class MockClient:
            is_mock = True
        client = MockClient()
    else:
        client = genai.Client(api_key=gemini_api_key)
        
    try:
        resolved = lookup_ingredient(db, client, query, language=language)
        return resolved
    except Exception as e:
        print(f"[SEARCH-INGREDIENT] Error searching for '{query}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error looking up ingredient: {str(e)}"
        )


# ──────────────────────────────────────────────────────────────────────
# Phase 10: Serve React Frontend Static Assets
# ──────────────────────────────────────────────────────────────────────

# Calculate absolute path to frontend/dist directory
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

if os.path.exists(frontend_dist):
    assets_path = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
        print(f"[FRONTEND] Mounted static assets from {assets_path}")
    else:
        print(f"[FRONTEND] Warning: Assets folder not found at {assets_path}")
        
    @app.get("/{catchall:path}")
    async def serve_frontend(catchall: str):
        # Prevent intercepting /api/* calls (safety check)
        if catchall.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
            
        # Serve any static files directly inside dist (like favicon.ico) if requested
        file_path = os.path.join(frontend_dist, catchall)
        if catchall and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
            
        # Serve index.html as fallback for React client router
        index_path = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    print(f"[FRONTEND] Catch-all route mounted for index.html at {frontend_dist}")
else:
    print(f"[FRONTEND] Warning: dist folder not found at {frontend_dist}. Skipping static files mounting.")


# Read host and port from environment variables for deployment safety
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host=host, port=port, reload=True)
