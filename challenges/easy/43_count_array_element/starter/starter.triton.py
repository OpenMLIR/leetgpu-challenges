import torch
import triton
import triton.language as tl

# Triton kernel to count the number of elements equal to a given value K
# This implementation processes the entire input array in chunks of BLOCK_SIZE
# using a single program instance (no parallelism across multiple program IDs).
@triton.jit
def count_equal_kernel(
    input_ptr,          # Pointer to the input array in GPU memory
    output_ptr,         # Pointer to the output location in GPU memory (stores the count)
    N,                  # Total number of elements in the input array
    K,                  # The value to compare against
    BLOCK_SIZE: tl.constexpr  # Number of elements processed per iteration
):
    # Initialize accumulator for the count
    sum = 0

    # Loop through the input array in chunks of BLOCK_SIZE
    for off in range(0, N, BLOCK_SIZE):
        # Compute offsets for the current chunk
        offsets = off + tl.arange(0, BLOCK_SIZE)

        # Load a block of elements from input_ptr (masked for out-of-bounds)
        x = tl.load(input_ptr + offsets, mask=offsets < N)

        # Compare each element with K:
        #   If x == K, return 1; otherwise return 0
        eq = tl.where(
            x == K,
            tl.full(x.shape, 1, x.dtype),
            tl.zeros(x.shape, x.dtype)
        )

        # Accumulate the sum of matches for this block
        sum += tl.sum(eq, axis=0)

    # Store the final count into output_ptr
    tl.store(output_ptr, sum)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, K: int):
    BLOCK_SIZE = 256
    grid = (1,)
    count_equal_kernel[grid](input, output, N, K, BLOCK_SIZE=BLOCK_SIZE)
