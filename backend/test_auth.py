import os
import unittest
from fastapi.testclient import TestClient

# Set environment variables for testing before importing anything
TEST_DB_FILE = "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_FILE}"
os.environ["JWT_SECRET"] = "test_jwt_secret_key_1234567890_test_jwt_secret"
os.environ["GEMINI_API_KEY"] = ""

from database import Base, engine, SessionLocal
from main import app
import models

class TestAuthAndDataFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create tables on the SQLite test.db database
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        Base.metadata.drop_all(bind=engine)
        # Delete test.db file if it exists
        if os.path.exists(TEST_DB_FILE):
            try:
                os.remove(TEST_DB_FILE)
            except Exception as e:
                print(f"Error removing test db file: {e}")

    def setUp(self):
        # Clear tables before each test to guarantee isolation
        db = SessionLocal()
        db.query(models.Favorite).delete()
        db.query(models.ScanHistory).delete()
        db.query(models.User).delete()
        db.commit()
        db.close()

    def test_signup_success(self):
        response = self.client.post(
            "/api/auth/signup",
            json={"name": "Alice Tester", "email": "alice@example.com", "password": "securepassword"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["name"], "Alice Tester")

    def test_signup_duplicate_email(self):
        # Signup first user
        self.client.post(
            "/api/auth/signup",
            json={"name": "Alice", "email": "duplicate@example.com", "password": "pwd"}
        )
        # Attempt signup again with same email
        response = self.client.post(
            "/api/auth/signup",
            json={"name": "Alice 2", "email": "duplicate@example.com", "password": "pwd"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("already registered", response.json()["detail"])

    def test_login_success(self):
        # Sign up
        self.client.post(
            "/api/auth/signup",
            json={"name": "Bob", "email": "bob@example.com", "password": "bobpassword"}
        )
        # Log in
        response = self.client.post(
            "/api/auth/login",
            json={"email": "bob@example.com", "password": "bobpassword"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["name"], "Bob")

    def test_login_invalid_credentials(self):
        # Sign up
        self.client.post(
            "/api/auth/signup",
            json={"name": "Bob", "email": "bob2@example.com", "password": "bobpassword"}
        )
        # Try wrong password
        response = self.client.post(
            "/api/auth/login",
            json={"email": "bob2@example.com", "password": "wrongpassword"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid email or password", response.json()["detail"])

    def test_get_me_protected(self):
        # Try without token
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 403) # Forbidden/Unauthorized

        # Sign up & get token
        signup_res = self.client.post(
            "/api/auth/signup",
            json={"name": "Charlie", "email": "charlie@example.com", "password": "password"}
        )
        token = signup_res.json()["access_token"]

        # Try with valid token
        response = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["email"], "charlie@example.com")
        self.assertEqual(response.json()["name"], "Charlie")

    def test_scan_history_and_favorites(self):
        # Register user
        signup_res = self.client.post(
            "/api/auth/signup",
            json={"name": "User", "email": "user@example.com", "password": "password"}
        )
        token = signup_res.json()["access_token"]

        # 1. Call parse ingredients with authorization token
        response = self.client.post(
            "/api/parse-ingredients",
            json={"raw_text": "Water, Sugar, Citric Acid", "language": "en"},
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(response.status_code, 200)
        parse_data = response.json()

        # Check history endpoint to verify it saved automatically
        history_res = self.client.get(
            "/api/history",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(history_res.status_code, 200)
        history_data = history_res.json()["history"]
        self.assertEqual(len(history_data), 1)
        self.assertEqual(history_data[0]["product_name"], parse_data["product_guess"])

        # 2. Add to Favorites
        fav_res = self.client.post(
            "/api/favorites",
            json={
                "product_name": parse_data["product_guess"],
                "product_guess": parse_data["product_guess"],
                "overall_score": parse_data["overall_score"],
                "ingredients_data": parse_data["ingredients"]
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(fav_res.status_code, 200)
        self.assertEqual(fav_res.json()["status"], "success")
        fav_id = fav_res.json()["id"]

        # View Favorites
        get_favs_res = self.client.get(
            "/api/favorites",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(get_favs_res.status_code, 200)
        favs_list = get_favs_res.json()["favorites"]
        self.assertEqual(len(favs_list), 1)
        self.assertEqual(favs_list[0]["product_name"], parse_data["product_guess"])

        # 3. Delete Favorite
        del_res = self.client.delete(
            f"/api/favorites/{fav_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["status"], "success")

        # Verify favorites list is now empty
        get_favs_res2 = self.client.get(
            "/api/favorites",
            headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(len(get_favs_res2.json()["favorites"]), 0)

if __name__ == "__main__":
    unittest.main()
