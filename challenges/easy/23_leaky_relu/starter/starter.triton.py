import torch
import triton
import triton.language as tl

# Triton kernel for Leaky ReLU activation
# Computes: output[i] = x if x > 0 else 0.01 * x
@triton.jit
def leaky_relu_kernel(
    input_ptr,               # Pointer to input tensor in GPU memory
    output_ptr,              # Pointer to output tensor in GPU memory
    n_elements,              # Total number of elements in the input/output tensors
    BLOCK_SIZE: tl.constexpr # Number of elements processed per program instance
):
    # Each Triton program processes BLOCK_SIZE elements
    pid = tl.program_id(axis=0)

    # Compute the offsets for the elements handled by this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure valid access (avoid out-of-bounds for the last block)
    mask = offsets < n_elements

    # Load input values from global memory (only valid positions)
    x = tl.load(input_ptr + offsets, mask=mask)

    # Apply Leaky ReLU:
    #   If x > 0 → x
    #   Else → 0.01 * x
    relu_val = tl.maximum(x * 0.01, x)

    # Store the computed result back to output tensor (masked)
    tl.store(output_ptr + offsets, relu_val, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    leaky_relu_kernel[grid](
        input,
        output,
        N,
        BLOCK_SIZE
    )
