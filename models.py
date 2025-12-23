"""
Database models for the Portfolio Dashboard
Uses SQLAlchemy ORM with SQLite for local development
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///portfolio.db")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Portfolio(Base):
    """Portfolio metadata"""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="portfolio", cascade="all, delete-orphan")


class Holding(Base):
    """Current holdings in a portfolio"""
    __tablename__ = "holdings"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    purchase_date = Column(Date, nullable=False)
    purchase_price = Column(Float, nullable=False)
    asset_class = Column(String(50), default="Equity")
    sector = Column(String(50))
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")


class Transaction(Base):
    """Transaction history"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String(10), nullable=False)
    transaction_date = Column(Date, nullable=False)
    transaction_type = Column(String(10), nullable=False)  # BUY, SELL, DIVIDEND
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fees = Column(Float, default=0.0)
    notes = Column(String(500))
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")


class PriceHistory(Base):
    """Cached price history for securities"""
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)
    
    class Meta:
        unique_together = ('ticker', 'date')


class Benchmark(Base):
    """Benchmark indices for comparison"""
    __tablename__ = "benchmarks"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    ticker = Column(String(10), nullable=False)
    description = Column(String(500))


def init_db():
    """Initialize the database and create all tables"""
    Base.metadata.create_all(bind=engine)
    
    # Add default benchmarks if they don't exist
    session = SessionLocal()
    try:
        if session.query(Benchmark).count() == 0:
            default_benchmarks = [
                Benchmark(name="S&P 500", ticker="SPY", description="SPDR S&P 500 ETF Trust"),
                Benchmark(name="Nasdaq 100", ticker="QQQ", description="Invesco QQQ Trust"),
                Benchmark(name="Dow Jones", ticker="DIA", description="SPDR Dow Jones Industrial Average ETF"),
                Benchmark(name="Total Market", ticker="VTI", description="Vanguard Total Stock Market ETF"),
                Benchmark(name="Russell 2000", ticker="IWM", description="iShares Russell 2000 ETF"),
            ]
            session.add_all(default_benchmarks)
            session.commit()
    finally:
        session.close()


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")
