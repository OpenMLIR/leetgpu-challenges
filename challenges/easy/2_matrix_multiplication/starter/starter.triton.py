import torch
import triton
import triton.language as tl

# Triton kernel for matrix multiplication
# Computes C[M, K] = A[M, N] * B[N, K]
@triton.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,         # pointers to matrices A, B, and C in GPU memory
    M, N, K,                     # dimensions: A[M,N], B[N,K], C[M,K]
    stride_am, stride_an,        # strides for A: row stride (am), column stride (an)
    stride_bn, stride_bk,        # strides for B: row stride (bn), column stride (bk)
    stride_cm, stride_ck,        # strides for C: row stride (cm), column stride (ck)
    BLOCK_SIZE_M: tl.constexpr,  # number of rows in a block of A and C
    BLOCK_SIZE_K: tl.constexpr,  # number of columns in a block of B and C
):
    # Program IDs define which block of work each instance of the kernel is responsible for.
    # axis=0 → K dimension (columns), axis=1 → M dimension (rows)
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # Compute the row and column indices for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # row indices
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)  # column indices

    # Compute pointers to A and B for the current block
    # A block: rows = offs_m, iterate N dimension later
    a_ptrs = a_ptr + offs_m[:, None] * stride_am
    # B block: columns = offs_k, iterate N dimension later
    b_ptrs = b_ptr + offs_k[None, :] * stride_bk

    # Initialize the accumulator for C with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # Loop over the shared dimension N
    # Each iteration processes one "slice" of A and B along the N axis
    for n in range(N):
        # Load A slice: shape (BLOCK_SIZE_M, 1)
        a = tl.load(a_ptrs + n * stride_an)
        # Load B slice: shape (1, BLOCK_SIZE_K)
        b = tl.load(b_ptrs + n * stride_bn)
        # Outer product update for C block
        accumulator += a * b

    # Compute the pointers for the output block in C
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck

    # Create a mask to avoid writing out-of-bounds elements
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    # Store the computed block back into C
    tl.store(c_ptrs, accumulator, mask=c_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = lambda META: (triton.cdiv(K, META['BLOCK_SIZE_K']), triton.cdiv(M, META['BLOCK_SIZE_M']), )
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_K=32,
    )
