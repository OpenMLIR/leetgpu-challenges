import torch
import triton
import triton.language as tl

# Triton kernel to reverse an array in-place
# Input: array of size N (float32 or any other type)
# Output: the same array, but reversed (input[i] ↔ input[N-1-i])
@triton.jit
def reverse_kernel(
    input_ptr,           # Pointer to the input array in GPU memory
    N,                   # Total number of elements in the array
    BLOCK_SIZE: tl.constexpr  # Number of elements processed per kernel instance
):
    # Program ID for axis 0: determines which block of elements this kernel handles
    pid = tl.program_id(axis=0)

    # Compute the offsets for this block: each program processes BLOCK_SIZE elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Only need to process the first half of the array because each swap covers two positions
    mask = offsets < (N // 2)

    # Compute the positions for the front and back elements to be swapped
    front_pos = offsets
    back_pos = N - 1 - offsets

    # Load the values at these positions (masked to avoid out-of-bounds access)
    front_val = tl.load(input_ptr + front_pos, mask=mask)
    back_val = tl.load(input_ptr + back_pos, mask=mask)

    # Swap the two values:
    #   front position gets the back value
    #   back position gets the front value
    tl.store(input_ptr + front_pos, back_val, mask=mask)
    tl.store(input_ptr + back_pos, front_val, mask=mask)

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)

    reverse_kernel[grid](
        input,
        N,
        BLOCK_SIZE
    )
