from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base

class IngredientReference(Base):
    """
    SQLAlchemy model representing the local ingredient safety database reference.
    Contains ingredient names, E-numbers/INS code aliases, safety status,
    layman explanations, allergen flags, and source reference associations.
    """
    __tablename__ = "ingredients_reference"

    id = Column(Integer, primary_key=True, index=True)
    ingredient_name = Column(String(255), unique=True, nullable=False, index=True)
    common_alias_names = Column(Text, nullable=True)
    category = Column(String(50), nullable=True) # 'ingredient' or 'additive'
    safety_status = Column(String(50), nullable=False) # 'safe', 'moderate', or 'unsafe'
    reason = Column(Text, nullable=True)
    simple_explanation = Column(Text, nullable=True)
    allergen = Column(String(50), nullable=True) # 'nuts', 'dairy', 'gluten', 'soy', or 'none'
    safe_frequency_per_week = Column(String(100), nullable=True)
    source_reference = Column(String(100), nullable=True) # 'FSSAI', 'WHO', 'ICMR', etc.

    def __repr__(self):
        return f"<IngredientReference(name='{self.ingredient_name}', status='{self.safety_status}')>"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    scans = relationship("ScanHistory", back_populates="user", cascade="all, delete-orphan")
    favorites = relationship("Favorite", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(name='{self.name}', email='{self.email}')>"


class ScanHistory(Base):
    __tablename__ = "scan_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    product_name = Column(String(255), nullable=True)
    product_guess = Column(String(255), nullable=True)
    overall_score = Column(Integer, nullable=True)
    score_label = Column(String(50), nullable=True)
    ingredients_data = Column(JSON, nullable=True)
    scanned_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="scans")

    def __repr__(self):
        return f"<ScanHistory(product='{self.product_name}', score={self.overall_score})>"


class Favorite(Base):
    __tablename__ = "favorites"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    product_name = Column(String(255), nullable=True)
    product_guess = Column(String(255), nullable=True)
    overall_score = Column(Integer, nullable=True)
    ingredients_data = Column(JSON, nullable=True)
    added_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="favorites")

    def __repr__(self):
        return f"<Favorite(product='{self.product_name}')>"
