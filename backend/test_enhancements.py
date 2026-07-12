import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import os
import json

# Set environment variables for testing before importing anything
TEST_DB_FILE = "test_enh.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_FILE}"
os.environ["JWT_SECRET"] = "test_jwt_secret_key_1234567890_test_jwt_secret"
os.environ["GEMINI_API_KEY"] = ""

from database import Base, engine
from main import app
import models

class TestEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(bind=engine)
        if os.path.exists(TEST_DB_FILE):
            try:
                os.remove(TEST_DB_FILE)
            except Exception as e:
                print(f"Error removing test db file: {e}")

    def setUp(self):
        # All routes now require authentication. Create a test user and obtain a JWT token.
        from database import SessionLocal
        db = SessionLocal()
        db.query(models.User).delete()
        db.commit()
        db.close()

        signup_res = self.client.post(
            "/api/auth/signup",
            json={"name": "Test User", "email": "test@example.com", "password": "password"}
        )
        self.assertEqual(signup_res.status_code, 200)
        self.token = signup_res.json()["access_token"]
        self.auth_headers = {"Authorization": f"Bearer {self.token}"}

    @patch("urllib.request.urlopen")
    def test_get_categories_success(self, mock_urlopen):
        # 1. Test loading categories from static taxonomy API successfully
        mock_response = MagicMock()
        mock_response.status = 200
        mock_data = {
            "en:biscuits": {"name": {"en": "Biscuits"}},
            "en:beverages": {"name": {"en": "Beverages"}},
            "fr:chocolat": {"name": {"en": "Chocolat"}},
            "en:empty": {"name": {"en": "Empty Cat"}}
        }
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Clear cache first to force call
        import main
        main._categories_cache = {"data": None, "expiry": 0.0}

        response = self.client.get("/api/categories", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("categories", data)
        categories = data["categories"]
        
        self.assertEqual(len(categories), 3)
        self.assertEqual(categories[0]["id"], "en:beverages")
        self.assertEqual(categories[1]["id"], "en:biscuits")
        self.assertEqual(categories[2]["id"], "en:empty")

    @patch("urllib.request.urlopen")
    def test_get_categories_fallback(self, mock_urlopen):
        # 2. Test fallback to mock categories if OFF API fails
        mock_urlopen.side_effect = Exception("API connection timed out")

        # Clear cache first to force call
        import main
        main._categories_cache = {"data": None, "expiry": 0.0}

        response = self.client.get("/api/categories", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("categories", data)
        categories = data["categories"]
        
        self.assertEqual(len(categories), 46)
        self.assertEqual(categories[0]["id"], "en:biscuits")

    def test_search_ingredient_mock(self):
        # 3. Test standalone ingredient search route
        response = self.client.get("/api/search-ingredient?query=INS 471", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["ingredient"], "INS 471")
        self.assertEqual(data["safety_status"], "safe")
        self.assertIn("emulsifier", data["reason"].lower())

        response = self.client.get("/api/search-ingredient?query=unknowningredient", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["ingredient"], "unknowningredient")
        self.assertEqual(data["safety_status"], "unsafe")

    @patch("urllib.request.urlopen")
    def test_get_category_best_success(self, mock_urlopen):
        # Test successfully retrieving and filtering category products from OFF
        mock_response = MagicMock()
        mock_response.status = 200
        mock_data = {
            "products": [
                {
                    "code": "111",
                    "product_name": "Dynamic Biscuit A",
                    "brands": "Brand A",
                    "ingredients_text": "Wheat Flour, Sugar, Palm Oil"
                },
                {
                    "code": "222",
                    "product_name": "Dynamic Biscuit B",
                    "brands": "Brand B",
                    "ingredients_text": "Wheat Flour, Salt"
                }
            ]
        }
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        response = self.client.get("/api/category-best?category=en:biscuits", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["category"], "en:biscuits")
        self.assertEqual(len(data["products"]), 2)
        codes = [prod["code"] for prod in data["products"]]
        self.assertIn("111", codes)
        self.assertIn("222", codes)
        self.assertEqual(data["products"][0]["rank"], 1)

    @patch("urllib.request.urlopen")
    def test_get_category_best_fallback_similar(self, mock_urlopen):
        # Test OFF search API returning empty/error, fallback matching similar mock category "chips"
        mock_urlopen.side_effect = Exception("Service Temporarily Unavailable")

        response = self.client.get("/api/category-best?category=en:chips-and-fries", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["category"], "en:chips-and-fries")
        # Should match mock "chips" and return products
        self.assertTrue(len(data["products"]) > 0)
        self.assertEqual(data["products"][0]["brand"], "Lay's")

    @patch("urllib.request.urlopen")
    def test_get_category_best_empty_message(self, mock_urlopen):
        # Test fallback failing to find matching mock category, returns empty with friendly message
        mock_urlopen.side_effect = Exception("Service Temporarily Unavailable")

        response = self.client.get("/api/category-best?category=en:non-existent-category", headers=self.auth_headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["category"], "en:non-existent-category")
        self.assertEqual(len(data["products"]), 0)
        self.assertIn("No products with ingredient data found", data["message"])

if __name__ == "__main__":
    unittest.main()
