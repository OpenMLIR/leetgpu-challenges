import torch
import triton
import triton.language as tl

@triton.jit
def fnv1a_hash(x):
    FNV_PRIME = 16777619
    OFFSET_BASIS = 2166136261
    
    hash_val = tl.full(x.shape, OFFSET_BASIS, tl.uint32)
    
    for byte_pos in range(4):
        byte = (x >> (byte_pos * 8)) & 0xFF
        hash_val = (hash_val ^ byte) * FNV_PRIME

    return hash_val

# Triton kernel to compute FNV-1a hash on an array of integers
# Each element is hashed independently using the FNV-1a algorithm
@triton.jit
def fnv1a_hash_kernel(
    input_ptr,                # Pointer to input array (GPU memory)
    output_ptr,               # Pointer to output array (GPU memory)
    n_elements,               # Number of elements in the input/output arrays
    n_rounds,                 # Number of times to apply the FNV-1a hash function
    BLOCK_SIZE: tl.constexpr  # Number of elements processed per Triton program
):
    # Program ID: identifies which block of elements this kernel instance processes
    pid = tl.program_id(axis=0)

    # Compute the indices for this block (BLOCK_SIZE elements per program)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask for valid indices to avoid out-of-bounds accesses
    mask = offsets < n_elements

    # Load the input values for this block (apply mask for safety)
    input_vals = tl.load(input_ptr + offsets, mask=mask)

    # Convert input values to 32-bit unsigned integers for hashing
    result = input_vals.to(tl.uint32)

    # Apply the FNV-1a hash function n_rounds times
    for _ in range(n_rounds):
        result = fnv1a_hash(result)

    # Store the hashed results back to output memory (masked)
    tl.store(output_ptr + offsets, result, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, R: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    fnv1a_hash_kernel[grid](
        input,
        output,
        N,
        R,
        BLOCK_SIZE
    )
