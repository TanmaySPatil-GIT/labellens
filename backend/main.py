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

def detect_blur(image_bytes: bytes, threshold: float = 100.0) -> tuple[bool, float]:
    """
    Calculates the Laplacian variance of the image to determine if it is blurry.
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
    return variance < threshold, variance

@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
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
        is_blurry, variance = detect_blur(image_bytes, threshold=100.0)
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
            detail=f"Google Gemini Vision API error: {str(api_err)}"
        )

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


from pydantic import BaseModel

class ParseIngredientsRequest(BaseModel):
    raw_text: str
    language: str = 'en'  # 'en', 'hi', 'mr'

@app.post("/api/parse-ingredients")
async def parse_ingredients(payload: ParseIngredientsRequest, db = Depends(get_db), user_optional: models.User | None = Depends(get_current_user_optional)):
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
        
        # 5. Save to history if logged in
        if user_optional:
            try:
                hist = models.ScanHistory(
                    user_id=user_optional.id,
                    product_name=product_guess,
                    product_guess=product_guess,
                    overall_score=overall_score,
                    score_label=score_label,
                    ingredients_data=results
                )
                db.add(hist)
                db.commit()
                print(f"[HISTORY] Successfully saved scan for user {user_optional.email}")
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
async def suggest_alternative(payload: SuggestAlternativeRequest):
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
    "biscuits": [
        {
            "name": "Britannia Marie Gold",
            "brand": "Britannia",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/106/301/5395/front_en.12.400.jpg",
            "code": "8901063015395",
            "ingredients_text": "Wheat Flour, Sugar, Refined Palm Oil, Invert Sugar Syrup, Milk Solids, Iodised Salt, Raising Agents (INS 500ii, INS 503ii), Emulsifiers (Soy Lecithin, INS 471), Citric Acid"
        },
        {
            "name": "McVitie's Digestive",
            "brand": "McVitie's",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/500/016/800/1007/front_en.59.400.jpg",
            "code": "5000168001007",
            "ingredients_text": "Wheat Flour, Whole Wheat Flour, Vegetable Oil, Sugar, Malt Extract, Invert Sugar Syrup, Raising Agents, Salt, Malic Acid"
        },
        {
            "name": "Oreo Chocolate Creme Biscuits",
            "brand": "Oreo",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/762/221/044/9283/front_en.11.400.jpg",
            "code": "7622210449283",
            "ingredients_text": "Refined Wheat Flour, Sugar, Refined Palm Oil, Cocoa Powder, High Fructose Corn Syrup, Raising Agents, Salt, Soy Lecithin, Artificial Flavouring Substance"
        },
        {
            "name": "Sunfeast Dark Fantasy Choco Fills",
            "brand": "Sunfeast",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/172/518/1225/front_en.21.400.jpg",
            "code": "8901725181225",
            "ingredients_text": "Refined Wheat Flour, Choco Filling (Sugar, Refined Palm Oil, Cocoa Solids, Soy Lecithin), Refined Palm Oil, Sugar, Cocoa Solids, Invert Sugar Syrup, Liquid Glucose, Raising Agents, Salt, Soy Lecithin, Artificial Vanilla Flavour"
        }
    ],
    "ketchup": [
        {
            "name": "Heinz Tomato Ketchup",
            "brand": "Heinz",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/001/300/000/6403/front_en.35.400.jpg",
            "code": "0013000006403",
            "ingredients_text": "Tomato Concentrate from Red Ripe Tomatoes, Distilled Vinegar, High Fructose Corn Syrup, Salt, Spice, Onion Powder, Natural Flavoring"
        },
        {
            "name": "Maggi Tomato Ketchup",
            "brand": "Maggi",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/105/800/2287/front_en.15.400.jpg",
            "code": "8901058002287",
            "ingredients_text": "Tomato Paste, Sugar, Water, Salt, Acidity Regulator (INS 260), Thickener (INS 415), Preservative (INS 211), Onion, Garlic, Spices"
        },
        {
            "name": "Kissan Fresh Tomato Ketchup",
            "brand": "Kissan",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/103/034/1859/front_en.24.400.jpg",
            "code": "8901030341859",
            "ingredients_text": "Tomato Paste, Water, Sugar, Liquid Glucose, Salt, Acidity Regulator (INS 260), Thickeners (INS 1422, INS 415), Preservative (INS 211), Onion Powder, Garlic Powder, Spices"
        }
    ],
    "juice": [
        {
            "name": "Real Fruit Power Orange Juice",
            "brand": "Dabur",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/120/700/0394/front_en.14.400.jpg",
            "code": "8901207000394",
            "ingredients_text": "Water, Orange Juice Concentrate, Sugar, Acidity Regulator (INS 330), Stabilizer (INS 440), Vitamin C"
        },
        {
            "name": "Tropicana 100% Orange Juice",
            "brand": "Tropicana",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/208/300/1086/front_en.18.400.jpg",
            "code": "8902083001086",
            "ingredients_text": "Orange Juice Concentrate, Water"
        },
        {
            "name": "Raw Pressery Valencia Orange Juice",
            "brand": "Raw Pressery",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/606/703/0146/front_en.10.400.jpg",
            "code": "8906067030146",
            "ingredients_text": "Valencia Orange Juice"
        },
        {
            "name": "Minute Maid Pulpy Orange",
            "brand": "Minute Maid",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/176/406/1151/front_en.23.400.jpg",
            "code": "8901764061151",
            "ingredients_text": "Water, Sugar, Orange Juice Concentrate, Orange Pulp, Acidity Regulator (INS 330), Antioxidant (INS 300)"
        }
    ],
    "chips": [
        {
            "name": "Lay's Classic Salted",
            "brand": "Lay's",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/149/110/1838/front_en.28.400.jpg",
            "code": "8901491101838",
            "ingredients_text": "Potato, Refined Palm Oil, Salt"
        },
        {
            "name": "Pringles Sour Cream & Onion",
            "brand": "Pringles",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/389/000/015/9985/front_en.59.400.jpg",
            "code": "3890000159985",
            "ingredients_text": "Dried Potatoes, Vegetable Oil, Corn Flour, Wheat Starch, Maltodextrin, Emulsifier (INS 471), Salt, Whey, Dextrose, Monosodium Glutamate, Coconut Oil, Yeast Extract, Citric Acid"
        },
        {
            "name": "Too Yumm Karare Veggie Stix",
            "brand": "Too Yumm",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/609/057/0275/front_en.19.400.jpg",
            "code": "8906090570275",
            "ingredients_text": "Rice Flour, Corn Flour, Palm Oil, Spices, Salt, Acidity Regulator (INS 330), Stabilizer (INS 412)"
        }
    ],
    "chocolates": [
        {
            "name": "Amul Dark Chocolate",
            "brand": "Amul",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/126/207/0400/front_en.24.400.jpg",
            "code": "8901262070400",
            "ingredients_text": "Sugar, Cocoa Solids, Cocoa Butter, Permitted Emulsifiers (INS 322, INS 476), Ethyl Vanillin"
        },
        {
            "name": "Cadbury Dairy Milk",
            "brand": "Cadbury",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/762/220/114/0168/front_en.38.400.jpg",
            "code": "7622201140168",
            "ingredients_text": "Sugar, Milk Solids, Cocoa Butter, Cocoa Solids, Emulsifiers (INS 442, INS 476), Ethyl Vanillin"
        }
    ],
    "noodles": [
        {
            "name": "Maggi 2-Minute Masala Noodles",
            "brand": "Maggi",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/105/889/5476/front_en.11.400.jpg",
            "code": "8901058895476",
            "ingredients_text": "Wheat Flour, Palm Oil, Salt, Wheat Gluten, Potassium Chloride, Gelling Agent (INS 508), Raising Agent (INS 500ii), Humectant (INS 451i), Coriander, Chilli, Turmeric, Cumin, Aniseed, Fenugreek, Ginger, Black Pepper, Clove, Nutmeg, Cardamom, Monosodium Glutamate, Citric Acid"
        },
        {
            "name": "Yippee Magic Masala Noodles",
            "brand": "Sunfeast",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/172/519/8124/front_en.15.400.jpg",
            "code": "8901725198124",
            "ingredients_text": "Wheat Flour, Refined Palm Oil, Iodised Salt, Wheat Gluten, Calcium Carbonate, Gelling Agent (INS 508), Humectant (INS 451i), Spices, Sugar, Garlic Powder, Onion Powder, Monosodium Glutamate, Yeast Extract"
        }
    ],
    "instant-noodles": [
        {
            "name": "Maggi 2-Minute Masala Noodles",
            "brand": "Maggi",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/105/889/5476/front_en.11.400.jpg",
            "code": "8901058895476",
            "ingredients_text": "Wheat Flour, Palm Oil, Salt, Wheat Gluten, Potassium Chloride, Gelling Agent (INS 508), Raising Agent (INS 500ii), Humectant (INS 451i), Coriander, Chilli, Turmeric, Cumin, Aniseed, Fenugreek, Ginger, Black Pepper, Clove, Nutmeg, Cardamom, Monosodium Glutamate, Citric Acid"
        },
        {
            "name": "Yippee Magic Masala Noodles",
            "brand": "Sunfeast",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/172/519/8124/front_en.15.400.jpg",
            "code": "8901725198124",
            "ingredients_text": "Wheat Flour, Refined Palm Oil, Iodised Salt, Wheat Gluten, Calcium Carbonate, Gelling Agent (INS 508), Humectant (INS 451i), Spices, Sugar, Garlic Powder, Onion Powder, Monosodium Glutamate, Yeast Extract"
        }
    ],
    "pasta": [
        {
            "name": "Bambino Macaroni Pasta",
            "brand": "Bambino",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/124/810/0075/front_en.5.400.jpg",
            "code": "8901248100075",
            "ingredients_text": "Semolina, Durum Wheat Semolina"
        },
        {
            "name": "YiPPee Tricolor Pasta Masala",
            "brand": "Sunfeast",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/172/510/3463/front_en.12.400.jpg",
            "code": "8901725103463",
            "ingredients_text": "Semolina, Wheat Gluten, Spices, Sugar, Iodised Salt, Dehydrated Vegetables, Hydrolyzed Vegetable Protein, Citric Acid"
        }
    ],
    "pastas": [
        {
            "name": "Bambino Macaroni Pasta",
            "brand": "Bambino",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/124/810/0075/front_en.5.400.jpg",
            "code": "8901248100075",
            "ingredients_text": "Semolina, Durum Wheat Semolina"
        },
        {
            "name": "YiPPee Tricolor Pasta Masala",
            "brand": "Sunfeast",
            "image_thumb_url": "https://images.openfoodfacts.org/images/products/890/172/510/3463/front_en.12.400.jpg",
            "code": "8901725103463",
            "ingredients_text": "Semolina, Wheat Gluten, Spices, Sugar, Iodised Salt, Dehydrated Vegetables, Hydrolyzed Vegetable Protein, Citric Acid"
        }
    ]
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
async def search_product(query: str):
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
async def analyze_barcode(code: str, language: str = "en", db = Depends(get_db), user_optional: models.User | None = Depends(get_current_user_optional)):
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
            
        # Save to history if logged in
        if user_optional:
            try:
                hist = models.ScanHistory(
                    user_id=user_optional.id,
                    product_name=product_guess,
                    product_guess=product_guess,
                    overall_score=overall_score,
                    score_label=score_label,
                    ingredients_data=results
                )
                db.add(hist)
                db.commit()
                print(f"[HISTORY] Successfully saved barcode scan for user {user_optional.email}")
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
async def get_category_best(category: str, db = Depends(get_db)):
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
async def get_categories():
    """
    Fetches the full list of available food categories from Open Food Facts,
    filters for English tags with > 50 products, sorted by product count descending.
    Caches the list in-memory for 24 hours.
    Falls back to mock categories if the API call fails.
    """
    global _categories_cache
    now = time.time()
    
    # 1. Return from cache if valid
    if _categories_cache["data"] and now < _categories_cache["expiry"]:
        return {"categories": _categories_cache["data"]}
        
    # 2. Fetch from Open Food Facts categories API
    url = "https://world.openfoodfacts.org/categories.json"
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
        with urllib.request.urlopen(req, context=ctx, timeout=8.0) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                tags = data.get("tags", [])
                
                # Filter English categories and any matching noodle/pasta categories with >= 10 products
                filtered = []
                for t in tags:
                    tag_id = t.get("id") or ""
                    tag_name = t.get("name") or ""
                    products_count = t.get("products") or 0
                    
                    tag_id_lower = tag_id.lower()
                    tag_name_lower = tag_name.lower()
                    
                    is_food_candidate = False
                    if tag_id_lower.startswith("en:"):
                        is_food_candidate = True
                    elif any(term in tag_id_lower or term in tag_name_lower for term in ["noodle", "pasta", "spaghetti", "macaroni"]):
                        is_food_candidate = True
                        
                    if is_food_candidate and products_count >= 10 and tag_name:
                        filtered.append({
                            "id": tag_id,
                            "name": tag_name,
                            "products": products_count
                        })
                
                # Sort by count descending
                filtered.sort(key=lambda x: x["products"], reverse=True)
                
                # Cache for 24 hours
                _categories_cache["data"] = filtered
                _categories_cache["expiry"] = now + (24 * 3600)
                
                print(f"[CATEGORIES] Loaded and filtered {len(filtered)} categories from OFF API (Threshold >= 10).")
                return {"categories": filtered}
                
    except Exception as e:
        print(f"[CATEGORIES] Failed to fetch from OFF categories API: {e}. Using mock fallback.")
        
    # 3. Fallback to mock categories (including noodles, instant-noodles, pasta) if the call fails
    fallback_categories = [
        {"id": "en:biscuits", "name": "Biscuits", "products": 500},
        {"id": "en:ketchups", "name": "Ketchup", "products": 500},
        {"id": "en:fruit-juices", "name": "Juices", "products": 500},
        {"id": "en:potato-chips", "name": "Chips & Crisps", "products": 500},
        {"id": "en:chocolates", "name": "Chocolates", "products": 500},
        {"id": "en:noodles", "name": "Noodles", "products": 300},
        {"id": "en:instant-noodles", "name": "Instant Noodles", "products": 400},
        {"id": "en:pastas", "name": "Pasta", "products": 350}
    ]
    print(f"[CATEGORIES] Returning {len(fallback_categories)} fallback categories (OFF API unavailable).")
    return {"categories": fallback_categories}


@app.get("/api/search-ingredient")
async def search_ingredient(query: str, language: str = 'en', db: Session = Depends(get_db)):
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
