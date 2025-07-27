#!/usr/bin/env python3
"""
Migration script to convert existing SQLite data to new SQLAlchemy schema
"""
import os
import sys
import sqlite3
from datetime import datetime
from sqlalchemy.exc import IntegrityError

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import initialize_database
from repositories.repository_factory import RepositoryFactory
from services.service_container_v2 import initialize_services_v2


def migrate_existing_data(old_db_path: str, new_db_path: str = None):
    """Migrate data from old SQLite database to new schema"""
    
    print("Starting migration to SQLAlchemy schema...")
    
    # Check if old database exists
    if not os.path.exists(old_db_path):
        print(f"Old database not found: {old_db_path}")
        return False
    
    # Initialize new database
    container = initialize_services_v2(database_url=f"sqlite:///{new_db_path}" if new_db_path else None)
    
    try:
        # Connect to old database
        old_conn = sqlite3.connect(old_db_path)
        old_cursor = old_conn.cursor()
        
        # Get repository factory
        with container.get_database_service().get_repository_factory() as factory:
            
            # Migrate logs table
            print("Migrating logs table...")
            try:
                old_cursor.execute("SELECT * FROM logs ORDER BY timestamp")
                log_repo = factory.get_miner_log_repository()
                
                migrated_logs = 0
                for row in old_cursor.fetchall():
                    try:
                        # Map old schema to new schema
                        log_data = {
                            'timestamp': datetime.fromisoformat(row[0]) if row[0] else datetime.utcnow(),
                            'ip': row[1],
                            'hostname': row[2],
                            'temp': row[3],
                            'hashRate': row[4],
                            'power': row[5],
                            'voltage': row[6],
                            'frequency': row[7],
                            'coreVoltage': row[8],
                            'fanrpm': row[9],
                            'sharesAccepted': row[10],
                            'sharesRejected': row[11],
                            'uptime': row[12],
                            'version': row[13]
                        }
                        
                        log_repo.create(**log_data)
                        migrated_logs += 1
                        
                        if migrated_logs % 1000 == 0:
                            print(f"Migrated {migrated_logs} log entries...")
                            
                    except (IntegrityError, ValueError) as e:
                        print(f"Skipping invalid log entry: {e}")
                        continue
                
                print(f"Migrated {migrated_logs} log entries")
                
            except sqlite3.OperationalError as e:
                print(f"Logs table migration skipped: {e}")
            
            # Migrate protocol table
            print("Migrating protocol/events table...")
            try:
                old_cursor.execute("SELECT * FROM protocol ORDER BY timestamp")
                event_repo = factory.get_protocol_event_repository()
                
                migrated_events = 0
                for row in old_cursor.fetchall():
                    try:
                        event_repo.log_event(
                            ip=row[1],
                            event_type=row[2],
                            message=row[3] or "",
                            severity="INFO"
                        )
                        migrated_events += 1
                        
                    except (IntegrityError, ValueError) as e:
                        print(f"Skipping invalid event: {e}")
                        continue
                
                print(f"Migrated {migrated_events} events")
                
            except sqlite3.OperationalError as e:
                print(f"Protocol table migration skipped: {e}")
            
            # Migrate benchmark_results table
            print("Migrating benchmark results...")
            try:
                old_cursor.execute("SELECT * FROM benchmark_results ORDER BY timestamp")
                benchmark_repo = factory.get_benchmark_result_repository()
                
                migrated_benchmarks = 0
                for row in old_cursor.fetchall():
                    try:
                        benchmark_repo.save_result(
                            ip=row[1],
                            frequency=row[2] or 0,
                            core_voltage=row[3] or 0,
                            avg_hashrate=row[4] or 0,
                            avg_temp=row[5] or 0,
                            efficiency=row[6] or 0,
                            duration=row[8] or 600,
                            averageVRTemp=row[7] if len(row) > 7 else None
                        )
                        migrated_benchmarks += 1
                        
                    except (IntegrityError, ValueError) as e:
                        print(f"Skipping invalid benchmark: {e}")
                        continue
                
                print(f"Migrated {migrated_benchmarks} benchmark results")
                
            except sqlite3.OperationalError as e:
                print(f"Benchmark results migration skipped: {e}")
            
            # Migrate efficiency_markers table
            print("Migrating efficiency markers...")
            try:
                old_cursor.execute("SELECT * FROM efficiency_markers ORDER BY timestamp")
                efficiency_repo = factory.get_efficiency_marker_repository()
                
                migrated_markers = 0
                for row in old_cursor.fetchall():
                    try:
                        efficiency_repo.log_efficiency(
                            ip=row[1],
                            efficiency=row[2] or 0,
                            hashrate=row[3] or 0,
                            power=row[4] or 0,
                            temperature=row[5] or 0,
                            frequency=row[6] or 0,
                            core_voltage=row[7] or 0
                        )
                        migrated_markers += 1
                        
                    except (IntegrityError, ValueError) as e:
                        print(f"Skipping invalid efficiency marker: {e}")
                        continue
                
                print(f"Migrated {migrated_markers} efficiency markers")
                
            except sqlite3.OperationalError as e:
                print(f"Efficiency markers migration skipped: {e}")
        
        old_conn.close()
        
        # Log migration completion
        container.get_database_service().log_event(
            "SYSTEM", "MIGRATION_COMPLETED", 
            f"Successfully migrated from {old_db_path}", "INFO"
        )
        
        print("Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"Migration failed: {e}")
        container.get_database_service().log_event(
            "SYSTEM", "MIGRATION_FAILED", 
            f"Migration failed: {e}", "ERROR"
        )
        return False
        
    finally:
        container.shutdown()


def create_backup(db_path: str) -> str:
    """Create backup of existing database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return None


def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate BITAXE database to SQLAlchemy")
    parser.add_argument("--old-db", required=True, help="Path to existing SQLite database")
    parser.add_argument("--new-db", help="Path for new database (optional)")
    parser.add_argument("--backup", action="store_true", help="Create backup before migration")
    
    args = parser.parse_args()
    
    # Create backup if requested
    if args.backup:
        backup_path = create_backup(args.old_db)
        if not backup_path:
            print("Failed to create backup. Aborting migration.")
            return 1
    
    # Run migration
    success = migrate_existing_data(args.old_db, args.new_db)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())