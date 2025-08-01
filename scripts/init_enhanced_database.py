#!/usr/bin/env python3
"""
BitAxe V2.0.0 - Enhanced Database Initialization
Creates the complete analytics-ready database schema
"""

import os
import sys
from pathlib import Path
import logging

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bitaxe_app.core.config_manager import ConfigManager
from bitaxe_app.core.database_manager import DatabaseManager
from bitaxe_app.core.exceptions import DatabaseError, ConfigurationError


def main():
    """Initialize the enhanced analytics database."""
    print("🚀 BitAxe V2.0.0 Enhanced Database Initialization")
    print("=" * 60)
    
    try:
        # Initialize configuration manager
        print("📝 Loading configuration...")
        config_manager = ConfigManager()
        
        print(f"   Database path: {config_manager.database_path}")
        print(f"   Configured miners: {len(config_manager.ips)}")
        
        # Initialize database manager
        print("🗄️  Initializing database schema...")
        database_manager = DatabaseManager(config_manager)
        
        print("   ✓ Core tables created")
        print("   ✓ Analytics tables created")
        print("   ✓ Performance optimization applied")
        print("   ✓ Indexes created")
        
        # Verify database health
        print("🔍 Performing health check...")
        health_status = database_manager.health_check()
        
        if health_status['status'] == 'healthy':
            print("   ✓ Database health check passed")
        else:
            print("   ⚠️  Database health check shows issues:")
            for check, result in health_status['checks'].items():
                status = "✓" if result else "✗"
                print(f"     {status} {check}")
        
        # Get database statistics
        print("📊 Database statistics:")
        stats = database_manager.get_database_stats()
        
        print(f"   File size: {stats['file_size_mb']:.2f} MB")
        print(f"   Page size: {stats['page_size']} bytes")
        print(f"   Journal mode: {stats['journal_mode']}")
        
        # Print table information
        print("\n📋 Database schema summary:")
        tables = [
            'logs', 'benchmark_results', 'protocol', 'tuning_status',
            'efficiency_markers', 'system_metrics', 'config_history',
            'analytics_summary', 'performance_alerts', 'profitability_data',
            'miner_health', 'data_quality'
        ]
        
        for table in tables:
            count_key = f'{table}_count'
            count = stats.get(count_key, 0)
            print(f"   {table}: {count} records")
        
        print("\n🎉 Enhanced database initialization complete!")
        print(f"📍 Location: {config_manager.database_path}")
        print("\n💡 Next steps:")
        print("   1. Run enhanced_data_logger.py to start collecting data")
        print("   2. Start the web application with python app.py")
        print("   3. Access the analytics dashboard at http://localhost:5000")
        
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
        print("   Please check your config.json file")
        sys.exit(1)
        
    except DatabaseError as e:
        print(f"❌ Database error: {e}")
        print("   Please check file permissions and disk space")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.exception("Full error details:")
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging for detailed error reporting
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()