import numpy as np
from typing import Dict, Any, Optional
import logging
import psutil

# Import from the ann-benchmarks base module
from ..base.module import BaseANN

logger = logging.getLogger(__name__)

class QuarkHNSW(BaseANN):
    """Quark HNSW implementation for ann-benchmarks."""
    
    def __init__(self, metric: str, method_param: Dict[str, Any]) -> None:
        self.metric = metric
        self.method_param = method_param
        self.data = None
        self.ef_search = method_param.get('ef_search', 100)
        self.M = method_param.get('M', 16)
        self.name = f"Quark-HNSW(M={self.M},ef={self.ef_search})"
        
    def fit(self, X: np.ndarray) -> None:
        """Build index from data."""
        logger.info(f"Building Quark-HNSW index with {X.shape[0]} vectors of dimension {X.shape[1]}")
        self.data = X.copy()
        
    def set_query_arguments(self, ef_search: int) -> None:
        """Set query-time parameters."""
        self.ef_search = ef_search
        self.name = f"Quark-HNSW(M={self.M},ef={ef_search})"
            
    def query(self, q: np.ndarray, n: int) -> np.ndarray:
        """Query for n nearest neighbors using brute force."""
        if self.data is None:
            raise RuntimeError("Index not built. Call fit() first.")
            
        # Brute force search for mock implementation
        if self.metric == 'euclidean':
            distances = np.sum((self.data - q.reshape(1, -1)) ** 2, axis=1)
        elif self.metric == 'angular':
            # Cosine distance
            norm_q = np.linalg.norm(q)
            norm_data = np.linalg.norm(self.data, axis=1)
            
            if norm_q > 0:
                cos_sim = np.dot(self.data, q) / (norm_data * norm_q)
                cos_sim = np.clip(cos_sim, -1, 1)  # Numerical stability
                distances = 1 - cos_sim
            else:
                distances = np.ones(len(self.data))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
            
        # Get top k indices
        indices = np.argpartition(distances, min(n, len(distances) - 1))[:n]
        # Sort by distance
        indices = indices[np.argsort(distances[indices])]
        
        return indices
        
    def get_memory_usage(self) -> Optional[float]:
        """Get memory usage in KB."""
        if self.data is not None:
            return self.data.nbytes / 1024
        return super().get_memory_usage()


class QuarkIVF(BaseANN):
    """Quark IVF implementation for ann-benchmarks."""
    
    def __init__(self, metric: str, method_param: Dict[str, Any]) -> None:
        self.metric = metric
        self.method_param = method_param
        self.data = None
        self.n_probe = method_param.get('n_probe', 10)
        self.n_lists = method_param.get('n_lists', 100)
        self.name = f"Quark-IVF(n_lists={self.n_lists},n_probe={self.n_probe})"
        
    def fit(self, X: np.ndarray) -> None:
        """Build IVF index from data."""
        logger.info(f"Building Quark-IVF index with {X.shape[0]} vectors of dimension {X.shape[1]}")
        self.data = X.copy()
        
    def set_query_arguments(self, n_probe: int) -> None:
        """Set query-time parameters."""
        self.n_probe = n_probe
        self.name = f"Quark-IVF(n_lists={self.n_lists},n_probe={n_probe})"
            
    def query(self, q: np.ndarray, n: int) -> np.ndarray:
        """Query for n nearest neighbors using brute force."""
        if self.data is None:
            raise RuntimeError("Index not built. Call fit() first.")
            
        # Mock IVF search (actually brute force for simplicity)
        if self.metric == 'euclidean':
            distances = np.sum((self.data - q.reshape(1, -1)) ** 2, axis=1)
        elif self.metric == 'angular':
            norm_q = np.linalg.norm(q)
            norm_data = np.linalg.norm(self.data, axis=1)
            
            if norm_q > 0:
                cos_sim = np.dot(self.data, q) / (norm_data * norm_q)
                cos_sim = np.clip(cos_sim, -1, 1)
                distances = 1 - cos_sim
            else:
                distances = np.ones(len(self.data))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
            
        # Get top k indices
        indices = np.argpartition(distances, min(n, len(distances) - 1))[:n]
        indices = indices[np.argsort(distances[indices])]
        
        return indices
        
    def get_memory_usage(self) -> Optional[float]:
        """Get memory usage in KB."""
        if self.data is not None:
            return self.data.nbytes / 1024
        return super().get_memory_usage()


class QuarkBinary(BaseANN):
    """Quark Binary implementation for ann-benchmarks."""
    
    def __init__(self, metric: str, method_param: Dict[str, Any]) -> None:
        self.metric = metric
        self.method_param = method_param
        self.data = None
        self.binary_data = None
        self.n_probe = method_param.get('n_probe', 10)
        self.n_lists = method_param.get('n_lists', 100)
        self.name = f"Quark-Binary(n_lists={self.n_lists},n_probe={self.n_probe})"
        
    def fit(self, X: np.ndarray) -> None:
        """Build Binary index from data."""
        logger.info(f"Building Quark-Binary index with {X.shape[0]} vectors of dimension {X.shape[1]}")
        self.data = X.copy()
        # Simple binary quantization (sign-based)
        self.binary_data = (X > 0).astype(np.uint8)
        
    def set_query_arguments(self, n_probe: int) -> None:
        """Set query-time parameters."""
        self.n_probe = n_probe
        self.name = f"Quark-Binary(n_lists={self.n_lists},n_probe={n_probe})"
            
    def query(self, q: np.ndarray, n: int) -> np.ndarray:
        """Query for n nearest neighbors using binary search."""
        if self.data is None:
            raise RuntimeError("Index not built. Call fit() first.")
            
        # Binary search using Hamming distance
        q_binary = (q > 0).astype(np.uint8)
        hamming_distances = np.sum(self.binary_data != q_binary.reshape(1, -1), axis=1)
        
        # Get top k indices
        indices = np.argpartition(hamming_distances, min(n, len(hamming_distances) - 1))[:n]
        indices = indices[np.argsort(hamming_distances[indices])]
        
        return indices
        
    def get_memory_usage(self) -> Optional[float]:
        """Get memory usage in KB."""
        if self.data is not None:
            binary_size = self.binary_data.nbytes if self.binary_data is not None else 0
            return (self.data.nbytes + binary_size) / 1024
        return super().get_memory_usage()