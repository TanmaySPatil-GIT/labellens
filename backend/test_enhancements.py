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

    @patch("urllib.request.urlopen")
    def test_get_categories_success(self, mock_urlopen):
        # 1. Test loading dynamic categories from OFF API successfully
        mock_response = MagicMock()
        mock_response.status = 200
        mock_data = {
            "tags": [
                {"id": "en:biscuits", "name": "Biscuits", "products": 500},
                {"id": "en:beverages", "name": "Beverages", "products": 800},
                {"id": "fr:chocolat", "name": "Chocolat", "products": 300},
                {"id": "en:empty", "name": "Empty Cat", "products": 10}
            ]
        }
        mock_response.read.return_value = json.dumps(mock_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Clear cache first to force call
        import main
        main._categories_cache = {"data": None, "expiry": 0.0}

        response = self.client.get("/api/categories")
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

        response = self.client.get("/api/categories")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("categories", data)
        categories = data["categories"]
        
        self.assertEqual(len(categories), 8)
        self.assertEqual(categories[0]["id"], "en:biscuits")

    def test_search_ingredient_mock(self):
        # 3. Test standalone ingredient search route
        response = self.client.get("/api/search-ingredient?query=INS 471")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["ingredient"], "INS 471")
        self.assertEqual(data["safety_status"], "safe")
        self.assertIn("emulsifier", data["reason"].lower())

        response = self.client.get("/api/search-ingredient?query=unknowningredient")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["ingredient"], "unknowningredient")
        self.assertEqual(data["safety_status"], "unsafe")

if __name__ == "__main__":
    unittest.main()
