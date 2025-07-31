"""
Signal Models
============
Modelos para manejar señales de trading y sus límites por suscripción
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Date, ForeignKey, DECIMAL, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
import enum

Base = declarative_base()

class SignalType(enum.Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SignalStrength(enum.Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"

class UserSignal(Base):
    __tablename__ = 'user_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    pair = Column(String(10), nullable=False)
    signal_type = Column(Enum(SignalType), nullable=False)
    strength = Column(Enum(SignalStrength), nullable=False)
    confidence = Column(DECIMAL(5, 2), nullable=False)
    entry_price = Column(DECIMAL(10, 5), nullable=False)
    stop_loss = Column(DECIMAL(10, 5))
    take_profit = Column(DECIMAL(10, 5))
    brain_type = Column(String(20), nullable=False)
    style = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_completed = Column(Boolean, default=False)
    actual_price_at_expiry = Column(DECIMAL(10, 5))
    signal_success = Column(Boolean)
    success_percentage = Column(DECIMAL(5, 2))
    signal_quality = Column(DECIMAL(5, 2))
    
    # Relationship
    user = relationship("User", back_populates="signals")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set expiration time to 15 minutes from creation
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(minutes=15)
    
    def is_expired(self):
        """Check if signal has expired"""
        return datetime.utcnow() > self.expires_at
    
    def calculate_success(self, actual_price):
        """Calculate if signal was successful"""
        if self.signal_type == SignalType.BUY:
            return actual_price >= self.take_profit if self.take_profit else actual_price > self.entry_price
        elif self.signal_type == SignalType.SELL:
            return actual_price <= self.take_profit if self.take_profit else actual_price < self.entry_price
        else:  # HOLD
            return True  # HOLD signals are always considered successful
    
    def calculate_success_percentage(self, actual_price):
        """Calculate success percentage based on price movement"""
        if self.signal_type == SignalType.BUY:
            if self.take_profit:
                if actual_price >= self.take_profit:
                    return 100.0
                else:
                    movement = (actual_price - float(self.entry_price)) / (float(self.take_profit) - float(self.entry_price))
                    return max(0, min(100, movement * 100))
            else:
                if actual_price > float(self.entry_price):
                    return 100.0
                else:
                    return 0.0
        elif self.signal_type == SignalType.SELL:
            if self.take_profit:
                if actual_price <= self.take_profit:
                    return 100.0
                else:
                    movement = (float(self.entry_price) - actual_price) / (float(self.entry_price) - float(self.take_profit))
                    return max(0, min(100, movement * 100))
            else:
                if actual_price < float(self.entry_price):
                    return 100.0
                else:
                    return 0.0
        else:  # HOLD
            return 100.0  # HOLD signals are always 100% successful

class UserSignalLimit(Base):
    __tablename__ = 'user_signal_limits'
    
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    plan_type = Column(String(20), nullable=False)
    max_signals_per_day = Column(Integer, nullable=False)
    signals_used_today = Column(Integer, default=0)
    last_reset_date = Column(Date, default=datetime.utcnow().date)
    
    # Relationship
    user = relationship("User", back_populates="signal_limits")
    
    def can_generate_signal(self):
        """Check if user can generate a new signal"""
        # Reset counter if it's a new day
        today = datetime.utcnow().date()
        if self.last_reset_date != today:
            self.signals_used_today = 0
            self.last_reset_date = today
        
        return self.signals_used_today < self.max_signals_per_day
    
    def increment_usage(self):
        """Increment signal usage counter"""
        self.signals_used_today += 1
    
    def get_remaining_signals(self):
        """Get remaining signals for today"""
        return max(0, self.max_signals_per_day - self.signals_used_today) 