"""
Background Task Scheduler

Comprehensive task scheduling system with priorities, retries, and monitoring.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import heapq
from concurrent.futures import ThreadPoolExecutor

from logging.structured_logger import get_logger
from exceptions.custom_exceptions import ServiceError, ErrorCode

logger = get_logger("bitaxe.task_scheduler")


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    retry_count: int = 0


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    schedule_time: datetime = field(default_factory=datetime.now)
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    last_error: Optional[str] = None
    result: Optional[Any] = None
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.schedule_time < other.schedule_time


class CronSchedule:
    """Simple cron-like scheduling"""
    
    def __init__(self, minute: str = "*", hour: str = "*", day: str = "*", 
                 month: str = "*", weekday: str = "*"):
        self.minute = minute
        self.hour = hour
        self.day = day
        self.month = month
        self.weekday = weekday
    
    def next_run_time(self, from_time: datetime = None) -> datetime:
        """Calculate next run time based on cron pattern"""
        if from_time is None:
            from_time = datetime.now()
        
        # Simple implementation - for now just add intervals
        # In production, would implement proper cron parsing
        next_time = from_time.replace(second=0, microsecond=0)
        
        if self.minute == "*":
            next_time += timedelta(minutes=1)
        elif self.minute.isdigit():
            target_minute = int(self.minute)
            if next_time.minute >= target_minute:
                next_time = next_time.replace(hour=next_time.hour + 1, minute=target_minute)
            else:
                next_time = next_time.replace(minute=target_minute)
        
        return next_time


@dataclass
class RecurringTask:
    """Recurring task definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    func: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    schedule: Union[CronSchedule, timedelta] = field(default_factory=lambda: timedelta(minutes=5))
    enabled: bool = True
    max_instances: int = 1  # Max concurrent instances
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """
    Comprehensive task scheduler with async/await support
    
    Features:
    - Priority-based task queuing
    - Cron-like recurring tasks
    - Retry logic with exponential backoff
    - Task timeout handling
    - Concurrent worker pools
    - Metrics and monitoring
    - Persistent task state (optional)
    """
    
    def __init__(self, max_workers: int = 10, max_concurrent_tasks: int = 50):
        self.max_workers = max_workers
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task queues and storage
        self.task_queue: List[ScheduledTask] = []  # Priority heap
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.recurring_tasks: Dict[str, RecurringTask] = {}
        
        # Workers and control
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers // 2)
        
        # Metrics
        self.metrics = {
            'total_tasks_scheduled': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_tasks_cancelled': 0,
            'total_tasks_retried': 0,
            'avg_execution_time': 0.0,
            'current_queue_size': 0,
            'current_running_tasks': 0,
            'worker_utilization': 0.0
        }
        self.execution_times: List[float] = []
        
        # Task registry for named tasks
        self.task_registry: Dict[str, Callable] = {}
    
    async def start(self):
        """Start the task scheduler"""
        if self.is_running:
            return
        
        logger.info("Starting task scheduler")
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start scheduler task
        scheduler_task = asyncio.create_task(self._scheduler())
        self.workers.append(scheduler_task)
        
        # Start metrics collector
        metrics_task = asyncio.create_task(self._metrics_collector())
        self.workers.append(metrics_task)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_worker())
        self.workers.append(cleanup_task)
        
        logger.info(f"Started task scheduler with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the task scheduler"""
        if not self.is_running:
            return
        
        logger.info("Stopping task scheduler")
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Cancel running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.workers.clear()
        self.running_tasks.clear()
        
        logger.info("Task scheduler stopped")
    
    def register_task(self, name: str, func: Callable):
        """Register a named task function"""
        self.task_registry[name] = func
        logger.debug(f"Registered task function: {name}")
    
    def schedule_task(self, task: ScheduledTask) -> str:
        """Schedule a task for execution"""
        self.metrics['total_tasks_scheduled'] += 1
        
        # Add to priority queue
        heapq.heappush(self.task_queue, task)
        
        logger.debug(f"Scheduled task {task.id}: {task.name}",
                    priority=task.priority.name,
                    schedule_time=task.schedule_time.isoformat())
        
        return task.id
    
    def schedule_function(self, func: Callable, *args, name: str = "", 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         delay: float = 0, timeout: Optional[float] = None,
                         max_retries: int = 3, **kwargs) -> str:
        """Schedule a function for execution"""
        schedule_time = datetime.now()
        if delay > 0:
            schedule_time += timedelta(seconds=delay)
        
        task = ScheduledTask(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            schedule_time=schedule_time,
            timeout=timeout,
            max_retries=max_retries
        )
        
        return self.schedule_task(task)
    
    def schedule_named_task(self, task_name: str, *args, 
                           priority: TaskPriority = TaskPriority.NORMAL,
                           delay: float = 0, **kwargs) -> str:
        """Schedule a registered named task"""
        if task_name not in self.task_registry:
            raise ValueError(f"Task '{task_name}' not registered")
        
        func = self.task_registry[task_name]
        return self.schedule_function(func, *args, name=task_name, 
                                    priority=priority, delay=delay, **kwargs)
    
    def add_recurring_task(self, recurring_task: RecurringTask) -> str:
        """Add a recurring task"""
        # Calculate next run time
        if isinstance(recurring_task.schedule, timedelta):
            recurring_task.next_run = datetime.now() + recurring_task.schedule
        elif isinstance(recurring_task.schedule, CronSchedule):
            recurring_task.next_run = recurring_task.schedule.next_run_time()
        
        self.recurring_tasks[recurring_task.id] = recurring_task
        
        logger.info(f"Added recurring task: {recurring_task.name}",
                   next_run=recurring_task.next_run.isoformat() if recurring_task.next_run else None)
        
        return recurring_task.id
    
    def schedule_recurring_function(self, func: Callable, interval: Union[timedelta, CronSchedule],
                                  *args, name: str = "", enabled: bool = True, **kwargs) -> str:
        """Schedule a function to run repeatedly"""
        recurring_task = RecurringTask(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            schedule=interval,
            enabled=enabled
        )
        
        return self.add_recurring_task(recurring_task)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        # Check running tasks
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            logger.info(f"Cancelled running task {task_id}")
            return True
        
        # Check queue
        for i, task in enumerate(self.task_queue):
            if task.id == task_id:
                task.status = TaskStatus.CANCELLED
                self.task_queue.pop(i)
                heapq.heapify(self.task_queue)  # Re-heapify after removal
                self.metrics['total_tasks_cancelled'] += 1
                logger.info(f"Cancelled queued task {task_id}")
                return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task status and result"""
        # Check completed tasks
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        # Check running tasks
        if task_id in self.running_tasks:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.RUNNING
            )
        
        # Check queue
        for task in self.task_queue:
            if task.id == task_id:
                return TaskResult(
                    task_id=task_id,
                    status=task.status
                )
        
        return None
    
    async def _worker(self, worker_name: str):
        """Worker coroutine that processes tasks"""
        logger.debug(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get next task
                task = await self._get_next_task()
                if not task:
                    await asyncio.sleep(0.1)
                    continue
                
                logger.debug(f"Worker {worker_name} executing task {task.id}: {task.name}")
                
                # Execute task
                await self._execute_task(task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error", error=str(e))
                await asyncio.sleep(1)
        
        logger.debug(f"Worker {worker_name} stopped")
    
    async def _get_next_task(self) -> Optional[ScheduledTask]:
        """Get next task from queue"""
        if not self.task_queue:
            return None
        
        # Check if highest priority task is ready
        task = self.task_queue[0]
        if task.schedule_time <= datetime.now():
            return heapq.heappop(self.task_queue)
        
        return None
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a single task"""
        task_id = task.id
        
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Create asyncio task
            if asyncio.iscoroutinefunction(task.func):
                # Async function
                async_task = asyncio.create_task(task.func(*task.args, **task.kwargs))
            else:
                # Sync function - run in thread pool
                loop = asyncio.get_event_loop()
                async_task = loop.run_in_executor(
                    self.thread_pool, 
                    lambda: task.func(*task.args, **task.kwargs)
                )
            
            self.running_tasks[task_id] = async_task
            
            # Wait for completion with timeout
            if task.timeout:
                result = await asyncio.wait_for(async_task, timeout=task.timeout)
            else:
                result = await async_task
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self.execution_times.append(execution_time)
            
            # Store result
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                start_time=task.started_at,
                end_time=task.completed_at,
                execution_time=execution_time,
                retry_count=task.retry_count
            )
            
            self.completed_tasks[task_id] = task_result
            self.metrics['total_tasks_completed'] += 1
            
            logger.debug(f"Task {task_id} completed successfully",
                        execution_time=execution_time)
            
        except asyncio.TimeoutError:
            # Task timed out
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.last_error = "Task timed out"
            
            self._handle_task_failure(task, "Task timed out")
            
        except asyncio.CancelledError:
            # Task was cancelled
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.CANCELLED,
                start_time=task.started_at,
                end_time=task.completed_at
            )
            
            self.completed_tasks[task_id] = task_result
            self.metrics['total_tasks_cancelled'] += 1
            
            logger.debug(f"Task {task_id} was cancelled")
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.last_error = str(e)
            
            self._handle_task_failure(task, str(e))
            
        finally:
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
    
    def _handle_task_failure(self, task: ScheduledTask, error: str):
        """Handle task failure with retry logic"""
        task.retry_count += 1
        
        if task.retry_count <= task.max_retries:
            # Retry the task
            task.status = TaskStatus.RETRYING
            retry_delay = task.retry_delay * (2 ** (task.retry_count - 1))  # Exponential backoff
            task.schedule_time = datetime.now() + timedelta(seconds=retry_delay)
            
            # Re-queue the task
            heapq.heappush(self.task_queue, task)
            
            self.metrics['total_tasks_retried'] += 1
            
            logger.warning(f"Task {task.id} failed, retrying in {retry_delay}s",
                          error=error, retry_count=task.retry_count)
        else:
            # Max retries exceeded
            execution_time = None
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at).total_seconds()
            
            task_result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=error,
                start_time=task.started_at,
                end_time=task.completed_at,
                execution_time=execution_time,
                retry_count=task.retry_count
            )
            
            self.completed_tasks[task.id] = task_result
            self.metrics['total_tasks_failed'] += 1
            
            logger.error(f"Task {task.id} failed permanently after {task.retry_count} retries",
                        error=error)
    
    async def _scheduler(self):
        """Schedule recurring tasks"""
        logger.debug("Task scheduler started")
        
        while self.is_running:
            try:
                now = datetime.now()
                
                # Check recurring tasks
                for recurring_task in self.recurring_tasks.values():
                    if not recurring_task.enabled:
                        continue
                    
                    if recurring_task.next_run and now >= recurring_task.next_run:
                        # Check max instances
                        active_instances = sum(
                            1 for task_id, task in self.running_tasks.items()
                            if task_id.startswith(f"{recurring_task.id}_")
                        )
                        
                        if active_instances < recurring_task.max_instances:
                            # Create new task instance
                            task_instance = ScheduledTask(
                                id=f"{recurring_task.id}_{uuid.uuid4().hex[:8]}",
                                name=f"{recurring_task.name} (recurring)",
                                func=recurring_task.func,
                                args=recurring_task.args,
                                kwargs=recurring_task.kwargs,
                                priority=TaskPriority.NORMAL,
                                schedule_time=now,
                                metadata={**recurring_task.metadata, 'recurring_task_id': recurring_task.id}
                            )
                            
                            self.schedule_task(task_instance)
                            
                            # Update recurring task
                            recurring_task.last_run = now
                            recurring_task.run_count += 1
                            
                            # Calculate next run time
                            if isinstance(recurring_task.schedule, timedelta):
                                recurring_task.next_run = now + recurring_task.schedule
                            elif isinstance(recurring_task.schedule, CronSchedule):
                                recurring_task.next_run = recurring_task.schedule.next_run_time(now)
                        
                        else:
                            logger.warning(f"Skipping recurring task {recurring_task.name} - max instances reached")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error("Scheduler error", error=str(e))
                await asyncio.sleep(1)
        
        logger.debug("Task scheduler stopped")
    
    async def _metrics_collector(self):
        """Collect and log metrics"""
        logger.debug("Metrics collector started")
        
        while self.is_running:
            try:
                # Update metrics
                self.metrics['current_queue_size'] = len(self.task_queue)
                self.metrics['current_running_tasks'] = len(self.running_tasks)
                self.metrics['worker_utilization'] = len(self.running_tasks) / self.max_workers * 100
                
                if self.execution_times:
                    self.metrics['avg_execution_time'] = sum(self.execution_times) / len(self.execution_times)
                    
                    # Keep only recent execution times
                    if len(self.execution_times) > 1000:
                        self.execution_times = self.execution_times[-1000:]
                
                # Log metrics
                logger.info("Task scheduler metrics",
                           total_scheduled=self.metrics['total_tasks_scheduled'],
                           total_completed=self.metrics['total_tasks_completed'],
                           total_failed=self.metrics['total_tasks_failed'],
                           total_cancelled=self.metrics['total_tasks_cancelled'],
                           total_retried=self.metrics['total_tasks_retried'],
                           queue_size=self.metrics['current_queue_size'],
                           running_tasks=self.metrics['current_running_tasks'],
                           worker_utilization=self.metrics['worker_utilization'],
                           avg_execution_time=self.metrics['avg_execution_time'],
                           recurring_tasks=len(self.recurring_tasks))
                
                await asyncio.sleep(60)  # Log every minute
                
            except Exception as e:
                logger.error("Metrics collector error", error=str(e))
                await asyncio.sleep(5)
        
        logger.debug("Metrics collector stopped")
    
    async def _cleanup_worker(self):
        """Clean up old completed tasks"""
        logger.debug("Cleanup worker started")
        
        while self.is_running:
            try:
                # Clean up old completed tasks (older than 1 hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                old_tasks = []
                
                for task_id, result in self.completed_tasks.items():
                    if result.end_time and result.end_time < cutoff_time:
                        old_tasks.append(task_id)
                
                for task_id in old_tasks:
                    del self.completed_tasks[task_id]
                
                if old_tasks:
                    logger.debug(f"Cleaned up {len(old_tasks)} old completed tasks")
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error("Cleanup worker error", error=str(e))
                await asyncio.sleep(60)
        
        logger.debug("Cleanup worker stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current scheduler metrics"""
        return {
            **self.metrics,
            'recurring_tasks': len(self.recurring_tasks),
            'registered_tasks': len(self.task_registry),
            'completed_tasks_stored': len(self.completed_tasks)
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed queue status"""
        priority_counts = {}
        for task in self.task_queue:
            priority = task.priority.name
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            'total_queued': len(self.task_queue),
            'priority_breakdown': priority_counts,
            'running_tasks': len(self.running_tasks),
            'max_workers': self.max_workers,
            'worker_utilization': len(self.running_tasks) / self.max_workers * 100
        }
    
    def get_recurring_tasks_status(self) -> List[Dict[str, Any]]:
        """Get status of all recurring tasks"""
        status_list = []
        
        for task in self.recurring_tasks.values():
            status_list.append({
                'id': task.id,
                'name': task.name,
                'enabled': task.enabled,
                'run_count': task.run_count,
                'last_run': task.last_run.isoformat() if task.last_run else None,
                'next_run': task.next_run.isoformat() if task.next_run else None,
                'max_instances': task.max_instances
            })
        
        return status_list
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()