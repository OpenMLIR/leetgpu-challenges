import torch
import triton
import triton.language as tl

# Triton kernel for 1D convolution
# Computes output[i] = sum_{k=0}^{kernel_size-1} input[i+k] * kernel[k]
# for i in range(input_size - kernel_size + 1)
@triton.jit
def conv1d_kernel(
    input_ptr,  # Pointer to input array
    kernel_ptr, # Pointer to kernel weights
    output_ptr, # Pointer to output array
    input_size, # Length of input array
    kernel_size,# Length of kernel (filter)
    BLOCK_SIZE: tl.constexpr # Number of output elements computed per program
):
    # Program ID for axis 0: determines which block of output this program computes
    pid = tl.program_id(axis=0)

    # Compute the starting index for this program in the output array
    output_start = pid * BLOCK_SIZE

    # Compute the indices of the output elements this program will handle
    offsets = output_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't compute beyond the valid output range
    # Valid outputs: indices < (input_size - kernel_size + 1)
    mask = offsets < (input_size - kernel_size + 1)

    # Initialize accumulator for the convolution result
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over the kernel elements (convolution sum)
    for k in range(kernel_size):
        # For each output position, the input position is offset by k
        input_pos = offsets + k

        # Load input values (masked to avoid out-of-bounds access)
        input_val = tl.load(input_ptr + input_pos,
                            mask=mask & (input_pos < input_size),
                            other=0.0)

        # Load the current kernel value (same for all threads in this block)
        kernel_val = tl.load(kernel_ptr + k)

        # Accumulate the product into the result
        accumulator += input_val * kernel_val

    # Store the computed output values back to memory
    tl.store(output_ptr + offsets, accumulator, mask=mask)

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks, )

    conv1d_kernel[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )
