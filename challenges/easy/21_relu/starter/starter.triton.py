import torch
import triton
import triton.language as tl

# Triton kernel for ReLU activation function
# Computes: output[i] = max(0, input[i]) for i in [0, n_elements)
@triton.jit
def relu_kernel(
    input_ptr,          # Pointer to input array (GPU memory)
    output_ptr,         # Pointer to output array (GPU memory)
    n_elements,         # Total number of elements to process
    BLOCK_SIZE: tl.constexpr  # Number of elements processed per program instance
):
    # Program ID along axis 0: each program handles a block of elements
    pid = tl.program_id(axis=0)

    # Compute the starting indices for this program block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we only process valid indices (avoid out-of-bounds access)
    mask = offsets < n_elements

    # Load input values from global memory (masked for out-of-range)
    x = tl.load(input_ptr + offsets, mask=mask)

    # Create a tensor of zeros with the same shape and dtype as x
    zero = tl.zeros(x.shape, x.dtype)

    # Apply ReLU: relu_val = max(0, x)
    relu_val = tl.maximum(zero, x)

    # Store the result back into the output array
    tl.store(output_ptr + offsets, relu_val, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)
