import torch
import triton
import triton.language as tl

# Triton kernel to copy elements from one array to another
# This is a simple element-wise memory copy, fully parallelized over the GPU.
@triton.jit
def copy_kernel(
    input_ptr,          # Pointer to the input array in GPU memory
    output_ptr,         # Pointer to the output array in GPU memory
    n_elements,         # Total number of elements to copy
    BLOCK_SIZE: tl.constexpr  # Number of elements each kernel instance processes
):
    # Program ID along axis 0: identifies which block of elements this kernel handles
    pid = tl.program_id(0)

    # Compute offsets for this block
    # Each program handles BLOCK_SIZE consecutive elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask to avoid out-of-bounds access for the last block
    mask = offsets < n_elements

    # Load input values from global memory (masked)
    val = tl.load(input_ptr + offsets, mask=mask)

    # Store the values to the output array (masked)
    tl.store(output_ptr + offsets, val, mask=mask)

# a, b are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    n_elements = N * N
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    copy_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)
