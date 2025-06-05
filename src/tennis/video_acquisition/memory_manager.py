import psutil
import gc
import numpy as np
from typing import Any, List
import logging

class M1MemoryManager:
    """Memory management for 8GB RAM constraint on M1 Mac"""
    
    def __init__(self, memory_threshold: float = 0.8):
        self.memory_threshold = memory_threshold  # 80% of available RAM
        self.logger = logging.getLogger(__name__)
    
    def check_memory_usage(self) -> float:
        """Monitor current memory usage"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0
        
        if usage_percent > self.memory_threshold:
            self.logger.warning(f"Memory usage high: {usage_percent:.1%}")
            self._cleanup_memory()
        
        return usage_percent
    
    def _cleanup_memory(self) -> None:
        """Force garbage collection and memory cleanup"""
        gc.collect()
        
        # Additional cleanup for numpy arrays
        for obj in gc.get_objects():
            if isinstance(obj, np.ndarray):
                try:
                    del obj
                except:
                    pass
        
        gc.collect()
    
    def batch_process_frames(self, frames: List[np.ndarray], 
                           batch_size: int = 10) -> List[List[np.ndarray]]:
        """Split frames into memory-safe batches"""
        
        # Calculate optimal batch size based on available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Estimate frame size (assuming 720p RGB)
        frame_size_mb = (1280 * 720 * 3) / (1024**2)
        safe_batch_size = max(1, int((available_gb * 1024 * 0.5) // frame_size_mb))
        
        actual_batch_size = min(batch_size, safe_batch_size)
        self.logger.info(f"Using batch size: {actual_batch_size}")
        
        return [frames[i:i + actual_batch_size] 
                for i in range(0, len(frames), actual_batch_size)]
