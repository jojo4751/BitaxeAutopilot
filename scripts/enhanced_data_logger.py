#!/usr/bin/env python3
"""
Enhanced Data Logger with Analytics
Collects comprehensive mining data with real-time analytics processing
"""

import time
import signal
import sys
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add parent directory to path for imports
sys.path.insert(0, '..')

from bitaxe_app.services import ServiceContainer

logger = logging.getLogger(__name__)


class EnhancedDataLogger:
    """Enhanced data logger with analytics processing."""
    
    def __init__(self):
        self.container = ServiceContainer()
        self.config_service = self.container.get_config_service()
        self.miner_service = self.container.get_miner_service()
        self.analytics_service = self.container.get_analytics_service()
        self.database_manager = self.container.get_database_service()
        
        self.running = True
        self.collection_interval = 30  # 30 seconds for high-frequency data
        self.summary_interval = 3600   # 1 hour for analytics summaries
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.last_summary_time = datetime.now()
        
    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received, stopping data logger...")
        self.running = False
        self.executor.shutdown(wait=True)
        self.container.shutdown()
        sys.exit(0)
    
    def collect_miner_data(self, ip: str) -> dict:
        """Collect data from a single miner with enhanced processing.
        
        Args:
            ip: Miner IP address
            
        Returns:
            Dictionary with collection results
        """
        try:
            start_time = time.time()
            
            # Fetch raw miner data
            raw_data = self.miner_service.fetch_miner_data(ip)
            
            if raw_data:
                # Process through analytics service
                self.analytics_service.process_miner_data(ip, raw_data)
                
                collection_time = time.time() - start_time
                
                return {
                    'ip': ip,
                    'success': True,
                    'data_points': len(raw_data),
                    'collection_time': collection_time,
                    'hashrate': raw_data.get('hashRate', 0),
                    'temp': raw_data.get('temp', 0),
                    'power': raw_data.get('power', 0)
                }
            else:
                return {
                    'ip': ip,
                    'success': False,
                    'error': 'No data received'
                }
                
        except Exception as e:
            logger.error(f"Error collecting data from {ip}: {e}")
            return {
                'ip': ip,
                'success': False,
                'error': str(e)
            }
    
    def collect_all_miners_data(self) -> dict:
        """Collect data from all configured miners concurrently.
        
        Returns:
            Dictionary with collection summary
        """
        ips = self.config_service.ips
        start_time = time.time()
        
        logger.info(f"Starting data collection from {len(ips)} miners...")
        
        # Submit all collection tasks
        future_to_ip = {
            self.executor.submit(self.collect_miner_data, ip): ip 
            for ip in ips
        }
        
        results = []
        successful_collections = 0
        total_data_points = 0
        
        # Collect results
        for future in as_completed(future_to_ip, timeout=30):
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    successful_collections += 1
                    total_data_points += result.get('data_points', 0)
                    
                    logger.debug(
                        f"✓ {result['ip']}: {result.get('hashrate', 0):.1f} GH/s, "
                        f"{result.get('temp', 0):.1f}°C, {result.get('power', 0):.1f}W "
                        f"({result['collection_time']:.2f}s)"
                    )
                else:
                    logger.warning(f"✗ {result['ip']}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                ip = future_to_ip[future]
                logger.error(f"Collection timeout/error for {ip}: {e}")
                results.append({
                    'ip': ip,
                    'success': False,
                    'error': f'Timeout/Exception: {e}'
                })
        
        total_time = time.time() - start_time
        
        summary = {
            'timestamp': datetime.now(),
            'total_miners': len(ips),
            'successful_collections': successful_collections,
            'failed_collections': len(ips) - successful_collections,
            'total_data_points': total_data_points,
            'collection_time': total_time,
            'results': results
        }
        
        logger.info(
            f"Collection complete: {successful_collections}/{len(ips)} miners, "
            f"{total_data_points} data points, {total_time:.2f}s"
        )
        
        return summary
    
    def generate_analytics_summaries(self):
        """Generate hourly analytics summaries for all miners."""
        try:
            current_time = datetime.now()
            
            # Generate hourly summaries
            end_time = current_time.replace(minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(hours=1)
            
            logger.info(f"Generating analytics summaries for {start_time} to {end_time}")
            
            for ip in self.config_service.ips:
                try:
                    summary = self.analytics_service.generate_performance_summary(
                        'hourly', start_time, end_time, ip
                    )
                    
                    if summary:
                        with self.database_manager.transaction() as conn:
                            self.database_manager.log_analytics_summary(
                                conn, 'hourly', start_time, end_time, ip, summary
                            )
                        
                        logger.debug(
                            f"Analytics summary for {ip}: "
                            f"score={summary.get('performance_score', 0):.1f}, "
                            f"uptime={summary.get('uptime_percentage', 0):.1f}%"
                        )
                    
                except Exception as e:
                    logger.error(f"Error generating summary for {ip}: {e}")
            
            self.last_summary_time = current_time
            
        except Exception as e:
            logger.error(f"Error generating analytics summaries: {e}")
    
    def cleanup_old_data(self):
        """Clean up old raw data to manage database size."""
        try:
            # Keep raw data for 7 days, summaries for much longer
            days_to_keep = 7
            deleted_counts = self.database_manager.cleanup_old_data(days_to_keep)
            
            if any(deleted_counts.values()):
                logger.info(f"Cleaned up old data: {deleted_counts}")
                
                # Run database vacuum occasionally
                if datetime.now().hour == 3:  # Run at 3 AM
                    logger.info("Running database vacuum...")
                    self.database_manager.vacuum_database()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def print_status_summary(self, collection_summary: dict):
        """Print a status summary to console."""
        timestamp = collection_summary['timestamp'].strftime('%H:%M:%S')
        successful = collection_summary['successful_collections']
        total = collection_summary['total_miners']
        collection_time = collection_summary['collection_time']
        
        # Calculate fleet stats
        active_results = [r for r in collection_summary['results'] if r['success']]
        
        if active_results:
            total_hashrate = sum(r.get('hashrate', 0) for r in active_results)
            avg_temp = sum(r.get('temp', 0) for r in active_results) / len(active_results)
            total_power = sum(r.get('power', 0) for r in active_results)
            
            print(f"[{timestamp}] Fleet Status: {successful}/{total} miners online | "
                  f"Total: {total_hashrate:.1f} GH/s, {avg_temp:.1f}°C, {total_power:.1f}W | "
                  f"Collection: {collection_time:.1f}s")
        else:
            print(f"[{timestamp}] Fleet Status: {successful}/{total} miners online | "
                  f"No active miners | Collection: {collection_time:.1f}s")
    
    def run(self):
        """Main data collection loop."""
        logger.info("Enhanced Data Logger started")
        logger.info(f"Collection interval: {self.collection_interval}s")
        logger.info(f"Summary generation interval: {self.summary_interval}s")
        logger.info(f"Monitoring {len(self.config_service.ips)} miners")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        collection_count = 0
        last_cleanup = datetime.now()
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Collect data from all miners
                collection_summary = self.collect_all_miners_data()
                self.print_status_summary(collection_summary)
                
                collection_count += 1
                
                # Generate analytics summaries every hour
                if (datetime.now() - self.last_summary_time).total_seconds() >= self.summary_interval:
                    logger.info("Generating periodic analytics summaries...")
                    self.generate_analytics_summaries()
                
                # Clean up old data daily
                if (datetime.now() - last_cleanup).total_seconds() >= 86400:  # 24 hours
                    logger.info("Running daily cleanup...")
                    self.cleanup_old_data()
                    last_cleanup = datetime.now()
                
                # Calculate sleep time to maintain consistent interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.collection_interval - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Collection cycle took {loop_duration:.1f}s, longer than {self.collection_interval}s interval")
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            logger.info(f"Data logger shutting down after {collection_count} collections")
            self.executor.shutdown(wait=True)
            self.container.shutdown()


def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('../logs/enhanced_data_logger.log')
        ]
    )
    
    logger.info("Starting Enhanced Data Logger...")
    
    try:
        data_logger = EnhancedDataLogger()
        data_logger.run()
    except Exception as e:
        logger.error(f"Failed to start data logger: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()