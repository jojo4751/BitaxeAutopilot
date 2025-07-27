import os
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from contextlib import contextmanager
import logging

from models.base import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database manager with connection pooling and session management"""
    
    def __init__(self, database_url: str = None, echo: bool = False):
        self.database_url = database_url or self._get_database_url()
        self.echo = echo
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or default to SQLite"""
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url
        
        # Default SQLite database
        db_path = os.getenv('DATABASE_PATH', 'data/bitaxe_data.db')
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f"sqlite:///{db_path}"
    
    def _initialize_engine(self):
        """Initialize database engine with appropriate settings"""
        if self.database_url.startswith('sqlite'):
            # SQLite specific settings
            self.engine = create_engine(
                self.database_url,
                echo=self.echo,
                poolclass=StaticPool,
                connect_args={
                    'check_same_thread': False,
                    'timeout': 30,
                    'isolation_level': None  # Use autocommit mode
                },
                pool_pre_ping=True,
                pool_recycle=3600  # Recycle connections every hour
            )
            
            # Enable WAL mode for better concurrency
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
                
        else:
            # PostgreSQL/MySQL settings
            self.engine = create_engine(
                self.database_url,
                echo=self.echo,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={'sslmode': 'prefer'} if 'postgresql' in self.database_url else {}
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database engine initialized: {self.database_url}")
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session_factory(self):
        """Get session factory for dependency injection"""
        return self.SessionLocal
    
    def health_check(self) -> bool:
        """Check database connection health"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get database connection information"""
        return {
            'url': self.database_url,
            'pool_size': getattr(self.engine.pool, 'size', None),
            'checked_in': getattr(self.engine.pool, 'checkedin', None),
            'checked_out': getattr(self.engine.pool, 'checkedout', None),
            'overflow': getattr(self.engine.pool, 'overflow', None),
        }
    
    def close(self):
        """Close database connections"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: DatabaseManager = None


def get_database_manager(database_url: str = None, echo: bool = False) -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url, echo)
    return _db_manager


def initialize_database(database_url: str = None, echo: bool = False, 
                       create_tables: bool = True) -> DatabaseManager:
    """Initialize database with tables"""
    global _db_manager
    _db_manager = DatabaseManager(database_url, echo)
    
    if create_tables:
        _db_manager.create_tables()
    
    return _db_manager


def get_session() -> Generator[Session, None, None]:
    """Get database session (dependency injection helper)"""
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session


def close_database():
    """Close database connections"""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None