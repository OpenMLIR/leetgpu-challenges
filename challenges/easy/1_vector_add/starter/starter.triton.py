import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Multiple "programs" are processing different data chunks. Here we determine which one we are
    pid = tl.program_id(axis=0)  # We launch a 1D grid, so the axis is 0.
    # This program will handle input starting from a certain offset.
    # For example, if the vector length is 4096 and block size is 1024,
    # programs will access elements [0:1024), [1024:2048), [2048:3072), [3072:4096) respectively.
    block_start = pid * BLOCK_SIZE
    # Note: `offsets` is a list of pointers.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent out-of-bounds memory access
    mask = offsets < n_elements
    # Load a and b from DRAM; the mask ensures we avoid reading beyond the input size
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    # Write a + b back to DRAM
    tl.store(c_ptr + offsets, c, mask=mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
