import numpy as np
from typing import List, Tuple, Optional

def unfold_tensor(tensor: np.ndarray, mode: int) -> np.ndarray:
    """
    Unfold a tensor along a specified mode.
    
    Args:
        tensor: Input tensor
        mode: Mode along which to unfold (0-based indexing)
        
    Returns:
        Unfolded matrix
    """
    shape = tensor.shape
    n = len(shape)
    return np.reshape(np.moveaxis(tensor, mode, 0), (shape[mode], -1))

def random_column_selection(tensor: np.ndarray, 
                          rank: int, 
                          num_samples: int,
                          seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Perform Random Column Selection (RCS) for tensor decomposition.
    
    Args:
        tensor: Input tensor
        rank: Target rank for decomposition
        num_samples: Number of columns to sample for each mode
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - List of factor matrices for each mode
        - List of sampled column indices for each mode
    """
    if seed is not None:
        np.random.seed(seed)
        
    n_modes = len(tensor.shape)
    factors = []
    sampled_indices = []
    
    for mode in range(n_modes):
        # Unfold tensor along current mode
        unfolded = unfold_tensor(tensor, mode)
        
        # Randomly sample columns
        n_cols = unfolded.shape[1]
        indices = np.random.choice(n_cols, size=min(num_samples, n_cols), replace=False)
        sampled_indices.append(indices)
        
        # Get sampled columns
        sampled_cols = unfolded[:, indices]
        
        # Compute SVD of sampled columns
        U, _, _ = np.linalg.svd(sampled_cols, full_matrices=False)
        
        # Take first 'rank' columns as factor matrix
        factor = U[:, :rank]
        factors.append(factor)
    
    return factors, sampled_indices

def reconstruct_tensor(factors: List[np.ndarray], 
                      shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reconstruct the tensor from factor matrices using the Tucker decomposition format.
    
    Args:
        factors: List of factor matrices for each mode
        shape: Original tensor shape
        
    Returns:
        Reconstructed tensor
    """
    n_modes = len(shape)
    core = np.ones(tuple(f.shape[1] for f in factors))
    
    # Initialize reconstructed tensor
    reconstructed = core
    
    # Perform tensor contractions
    for mode in range(n_modes):
        reconstructed = np.tensordot(reconstructed, factors[mode], axes=(mode, 1))
    
    return reconstructed

def rcs_decomposition(tensor: np.ndarray,
                     rank: int,
                     num_samples: int,
                     seed: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Perform complete RCS-based tensor decomposition.
    
    Args:
        tensor: Input tensor
        rank: Target rank for decomposition
        num_samples: Number of columns to sample for each mode
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
        - List of factor matrices
        - Reconstructed tensor
    """
    factors, _ = random_column_selection(tensor, rank, num_samples, seed)
    reconstructed = reconstruct_tensor(factors, tensor.shape)
    return factors, reconstructed
