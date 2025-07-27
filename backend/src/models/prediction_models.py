"""
Prediction Models for Day Trading System
"""
from sqlalchemy import Column, Integer, String, DECIMAL, Text, TIMESTAMP, Boolean, Date, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import enum

Base = declarative_base()

class PredictionDirection(enum.Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

class UserPrediction(Base):
    __tablename__ = 'user_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    pair = Column(String(10), nullable=False)
    direction = Column(Enum(PredictionDirection), nullable=False)
    current_price = Column(DECIMAL(10, 5), nullable=False)
    target_price = Column(DECIMAL(10, 5), nullable=False)
    confidence = Column(DECIMAL(5, 2), nullable=False)
    timeframe = Column(String(10), default='15M')
    reasoning = Column(Text)
    brain_type = Column(String(20), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    expires_at = Column(TIMESTAMP)
    is_completed = Column(Boolean, default=False)
    actual_price_at_expiry = Column(DECIMAL(10, 5))
    prediction_success = Column(Boolean)
    success_percentage = Column(DECIMAL(5, 2))
    
    # Relationship
    user = relationship("User", back_populates="predictions")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set expiration time to 15 minutes from creation
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(minutes=15)
    
    def is_expired(self):
        """Check if prediction has expired"""
        return datetime.utcnow() > self.expires_at
    
    def calculate_success(self, actual_price):
        """Calculate if prediction was successful"""
        if self.direction == PredictionDirection.UP:
            return actual_price >= self.target_price
        elif self.direction == PredictionDirection.DOWN:
            return actual_price <= self.target_price
        else:  # SIDEWAYS
            threshold = float(self.current_price) * 0.001  # 0.1% threshold
            return abs(actual_price - float(self.current_price)) <= threshold
    
    def calculate_success_percentage(self, actual_price):
        """Calculate success percentage based on price movement"""
        if self.direction == PredictionDirection.UP:
            if actual_price >= self.target_price:
                return 100.0
            else:
                movement = (actual_price - float(self.current_price)) / (float(self.target_price) - float(self.current_price))
                return max(0, min(100, movement * 100))
        elif self.direction == PredictionDirection.DOWN:
            if actual_price <= self.target_price:
                return 100.0
            else:
                movement = (float(self.current_price) - actual_price) / (float(self.current_price) - float(self.target_price))
                return max(0, min(100, movement * 100))
        else:  # SIDEWAYS
            threshold = float(self.current_price) * 0.001
            deviation = abs(actual_price - float(self.current_price))
            if deviation <= threshold:
                return 100.0
            else:
                return max(0, 100 - (deviation / threshold) * 100)

class UserPredictionLimit(Base):
    __tablename__ = 'user_prediction_limits'
    
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    plan_type = Column(String(20), nullable=False)
    max_predictions_per_day = Column(Integer, nullable=False)
    predictions_used_today = Column(Integer, default=0)
    last_reset_date = Column(Date, default=datetime.utcnow().date)
    
    # Relationship
    user = relationship("User", back_populates="prediction_limits")
    
    def can_generate_prediction(self):
        """Check if user can generate a new prediction"""
        # Reset counter if it's a new day
        today = datetime.utcnow().date()
        if self.last_reset_date != today:
            self.predictions_used_today = 0
            self.last_reset_date = today
        
        return self.predictions_used_today < self.max_predictions_per_day
    
    def increment_usage(self):
        """Increment prediction usage counter"""
        self.predictions_used_today += 1
    
    def get_remaining_predictions(self):
        """Get remaining predictions for today"""
        return max(0, self.max_predictions_per_day - self.predictions_used_today) 