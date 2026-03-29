"""
Performance timing module for Hyper-Playground
Logs execution times for major operations to help identify bottlenecks.
"""
import time
import os
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Optional
import threading


class PerformanceTimer:
    """
    Thread-safe performance timer that logs execution times to a file.
    Each timing entry includes: timestamp, script name, line numbers, description, and duration.
    """
    
    def __init__(self, output_dir: str, map_name: str = "global"):
        """
        Initialize the performance timer.
        
        Args:
            output_dir: Directory where timing logs will be saved
            map_name: Name of the map being processed (for separate timing files)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique timing file for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timing_file = self.output_dir / f"performance_timing_{map_name}_{timestamp}.log"
        
        # Thread lock for safe concurrent writes
        self._lock = threading.Lock()
        
        # Initialize the file with header
        self._write_header()
    
    def _write_header(self):
        """Write header to the timing file."""
        with open(self.timing_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write(f"HYPER-PLAYGROUND PERFORMANCE TIMING LOG\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"{'Timestamp':<20} {'Script':<30} {'Lines':<15} {'Duration':<12} {'Description':<60}\n")
            f.write("-" * 100 + "\n")
    
    def log_timing(self, script_name: str, start_line: int, end_line: int, 
                   description: str, duration: float):
        """
        Log a timing entry to the file.
        
        Args:
            script_name: Name of the script/module
            start_line: Starting line number
            end_line: Ending line number
            description: Description of the operation
            duration: Duration in seconds
        """
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            line_range = f"{start_line}-{end_line}"
            duration_str = f"{duration:.4f}s"
            
            with open(self.timing_file, 'a') as f:
                f.write(f"{timestamp:<20} {script_name:<30} {line_range:<15} {duration_str:<12} {description:<60}\n")
    
    @contextmanager
    def measure(self, script_name: str, start_line: int, end_line: int, description: str):
        """
        Context manager for measuring execution time.
        
        Usage:
            with timer.measure("script.py", 100, 150, "Loading data"):
                # code to measure
                pass
        
        Args:
            script_name: Name of the script/module
            start_line: Starting line number
            end_line: Ending line number  
            description: Description of the operation
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.log_timing(script_name, start_line, end_line, description, duration)
    
    def get_summary(self) -> str:
        """
        Generate a summary of all timings.
        
        Returns:
            String containing timing summary
        """
        try:
            with open(self.timing_file, 'r') as f:
                content = f.read()
            
            # Parse timings (skip header lines)
            lines = content.split('\n')
            timings = []
            for line in lines:
                if line and not line.startswith(('=', '-', 'HYPER', 'Generated', 'Timestamp')):
                    try:
                        parts = line.split()
                        if len(parts) >= 4 and 's' in parts[3]:
                            duration = float(parts[3].rstrip('s'))
                            script = parts[1]
                            timings.append((script, duration))
                    except:
                        pass
            
            if not timings:
                return "No timings recorded yet."
            
            # Calculate totals by script
            script_totals = {}
            for script, duration in timings:
                script_totals[script] = script_totals.get(script, 0) + duration
            
            total_time = sum(script_totals.values())
            
            summary = "\n" + "=" * 80 + "\n"
            summary += "PERFORMANCE SUMMARY\n"
            summary += "=" * 80 + "\n"
            summary += f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n\n"
            summary += f"{'Script':<35} {'Time (s)':<12} {'% of Total':<12}\n"
            summary += "-" * 80 + "\n"
            
            for script, duration in sorted(script_totals.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                summary += f"{script:<35} {duration:>10.2f}s {percentage:>10.1f}%\n"
            
            return summary
            
        except Exception as e:
            return f"Error generating summary: {e}"
    
    def write_summary(self):
        """Write the summary to the timing file."""
        summary = self.get_summary()
        with self._lock:
            with open(self.timing_file, 'a') as f:
                f.write("\n\n" + summary)


# Global timer instance (will be initialized in each module)
_global_timer: Optional[PerformanceTimer] = None


def get_timer() -> Optional[PerformanceTimer]:
    """Get the global timer instance."""
    return _global_timer


def set_timer(timer: PerformanceTimer):
    """Set the global timer instance."""
    global _global_timer
    _global_timer = timer


def init_timer(output_dir: str, map_name: str = "global") -> PerformanceTimer:
    """
    Initialize the global timer.
    
    Args:
        output_dir: Directory where timing logs will be saved
        map_name: Name of the map being processed
    
    Returns:
        PerformanceTimer instance
    """
    timer = PerformanceTimer(output_dir, map_name)
    set_timer(timer)
    return timer
