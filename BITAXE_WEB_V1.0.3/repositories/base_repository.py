from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Type, TypeVar, Generic
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc, asc, and_, or_
from models.base import Base

T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T], ABC):
    """Abstract base repository with common CRUD operations"""
    
    def __init__(self, session: Session, model_class: Type[T]):
        self.session = session
        self.model_class = model_class
    
    def create(self, **kwargs) -> T:
        """Create a new record"""
        try:
            instance = self.model_class(**kwargs)
            self.session.add(instance)
            self.session.commit()
            self.session.refresh(instance)
            return instance
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e
    
    def get_by_id(self, record_id: int) -> Optional[T]:
        """Get record by ID"""
        return self.session.query(self.model_class).filter(
            self.model_class.id == record_id
        ).first()
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all records with pagination"""
        return self.session.query(self.model_class).offset(offset).limit(limit).all()
    
    def update(self, record_id: int, **kwargs) -> Optional[T]:
        """Update a record by ID"""
        try:
            instance = self.get_by_id(record_id)
            if instance:
                for key, value in kwargs.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                self.session.commit()
                self.session.refresh(instance)
            return instance
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e
    
    def delete(self, record_id: int) -> bool:
        """Delete a record by ID"""
        try:
            instance = self.get_by_id(record_id)
            if instance:
                self.session.delete(instance)
                self.session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e
    
    def count(self) -> int:
        """Count total records"""
        return self.session.query(self.model_class).count()
    
    def exists(self, **filters) -> bool:
        """Check if record exists with given filters"""
        query = self.session.query(self.model_class)
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                query = query.filter(getattr(self.model_class, key) == value)
        return query.first() is not None
    
    def find_by(self, limit: int = 100, offset: int = 0, order_by: str = None, 
                desc_order: bool = False, **filters) -> List[T]:
        """Find records by filters"""
        query = self.session.query(self.model_class)
        
        # Apply filters
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                attr = getattr(self.model_class, key)
                if isinstance(value, list):
                    query = query.filter(attr.in_(value))
                elif isinstance(value, dict) and 'operator' in value:
                    # Handle complex operators like >=, <=, etc.
                    op = value['operator']
                    val = value['value']
                    if op == 'gte':
                        query = query.filter(attr >= val)
                    elif op == 'lte':
                        query = query.filter(attr <= val)
                    elif op == 'gt':
                        query = query.filter(attr > val)
                    elif op == 'lt':
                        query = query.filter(attr < val)
                    elif op == 'like':
                        query = query.filter(attr.like(f'%{val}%'))
                    else:
                        query = query.filter(attr == val)
                else:
                    query = query.filter(attr == value)
        
        # Apply ordering
        if order_by and hasattr(self.model_class, order_by):
            order_attr = getattr(self.model_class, order_by)
            if desc_order:
                query = query.order_by(desc(order_attr))
            else:
                query = query.order_by(asc(order_attr))
        
        return query.offset(offset).limit(limit).all()
    
    def find_one_by(self, **filters) -> Optional[T]:
        """Find single record by filters"""
        results = self.find_by(limit=1, **filters)
        return results[0] if results else None
    
    def bulk_create(self, records_data: List[Dict[str, Any]]) -> List[T]:
        """Create multiple records in bulk"""
        try:
            instances = [self.model_class(**data) for data in records_data]
            self.session.bulk_save_objects(instances, return_defaults=True)
            self.session.commit()
            return instances
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e
    
    def bulk_update(self, updates: List[Dict[str, Any]]) -> bool:
        """Update multiple records in bulk"""
        try:
            self.session.bulk_update_mappings(self.model_class, updates)
            self.session.commit()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e
    
    def delete_by(self, **filters) -> int:
        """Delete records matching filters"""
        try:
            query = self.session.query(self.model_class)
            for key, value in filters.items():
                if hasattr(self.model_class, key):
                    query = query.filter(getattr(self.model_class, key) == value)
            
            count = query.count()
            query.delete()
            self.session.commit()
            return count
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e


class TimestampRepository(BaseRepository[T]):
    """Repository with timestamp-based queries"""
    
    def get_recent(self, hours: int = 24, limit: int = 100) -> List[T]:
        """Get recent records within specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return self.find_by(
            limit=limit,
            order_by='timestamp',
            desc_order=True,
            timestamp={'operator': 'gte', 'value': cutoff_time}
        )
    
    def get_by_time_range(self, start_time: datetime, end_time: datetime, 
                         limit: int = 1000, offset: int = 0) -> List[T]:
        """Get records within time range"""
        return self.session.query(self.model_class).filter(
            and_(
                self.model_class.timestamp >= start_time,
                self.model_class.timestamp <= end_time
            )
        ).order_by(self.model_class.timestamp).offset(offset).limit(limit).all()
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """Delete records older than specified days"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        return self.delete_by(
            timestamp={'operator': 'lt', 'value': cutoff_time}
        )