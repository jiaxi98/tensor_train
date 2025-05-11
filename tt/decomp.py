import jax
import jax.numpy as jnp
from jax import lax

def tensor_train_decomposition(tensor, max_rank=None, eps=1e-10):
    """Tensor Train Decomposition
    
    参数:
        tensor: jnp.ndarray - 输入的高阶张量（至少2阶）
        max_rank: int - 最大允许的TT秩（默认自动选择）
        eps: float - 奇异值截断阈值
    
    返回:
        list of jnp.ndarray - TT核心列表，每个核心形状为(r_{i-1}, dim_i, r_i)
    """
    ndim = tensor.ndim
    shape = tensor.shape
    cores = []
    
    remaining_tensor = tensor.reshape(shape[0], -1)
    r_prev = 1
    
    for i in range(ndim-1):
        U, S, Vh = jnp.linalg.svd(remaining_tensor, full_matrices=False)
        
        if max_rank is not None:
            rank = min(max_rank, len(S))
        else:
            cum_energy = jnp.cumsum(S**2) / jnp.sum(S**2)
            rank = jnp.searchsorted(cum_energy, 1 - eps) + 1
        
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        core = U.reshape(r_prev, shape[i], rank)
        cores.append(core)
        
        remaining_tensor = (jnp.diag(S) @ Vh).reshape(rank * shape[i+1], -1)
        r_prev = rank
    
    final_core = remaining_tensor.reshape(r_prev, shape[-1], 1)
    cores.append(final_core)
    
    return cores

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    tensor = jax.random.normal(key, (2, 3, 4, 5))

    tt_cores = tensor_train_decomposition(tensor, max_rank=5)

    print("TT核心形状:")
    for i, core in enumerate(tt_cores):
        print(f"核心 {i}: {core.shape}")

    def reconstruct(cores):
        res = cores[0]
        for core in cores[1:]:
            res = jnp.tensordot(res, core, axes=(-1, 0))
        return res

    reconstructed = reconstruct(tt_cores)
    error = jnp.linalg.norm(tensor - reconstructed)
    print(f"\n重建误差: {error:.4e}")