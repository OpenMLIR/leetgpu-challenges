import torch
import triton
import triton.language as tl

# Triton kernel to invert image colors (except alpha channel)
# The image is assumed to be stored in RGBA format (4 channels per pixel).
# For each pixel:
#   R' = 255 - R
#   G' = 255 - G
#   B' = 255 - B
#   A remains unchanged
@triton.jit
def invert_kernel(
    image_ptr,               # Pointer to the image data in GPU memory (uint8 values)
    width, height,           # Image dimensions
    BLOCK_SIZE: tl.constexpr # Number of pixels processed per kernel instance (tile size)
):
    # Get the program ID along the first axis.
    # Each program processes a block of pixels.
    pid = tl.program_id(axis=0)

    # Compute the starting index in the flattened array for this block
    # Each pixel has 4 channels (RGBA), so multiply by 4
    block_start = pid * BLOCK_SIZE * 4

    # Compute the offsets for all elements in this block (BLOCK_SIZE pixels * 4 channels)
    offsets = block_start + tl.arange(0, BLOCK_SIZE * 4)

    # Create a mask:
    #  1. (offsets < width*height*4): avoid out-of-bounds access at the end
    #  2. (offsets % 4 != 3): skip the alpha channel (do not invert it)
    mask = (offsets < width * height * 4) & (offsets % 4 != 3)

    # Load the RGBA values for this block (masked load for valid indices only)
    input = tl.load(image_ptr + offsets, mask=mask)

    # Invert color values: output = 255 - input
    # Only applied to R, G, B channels because alpha was masked out
    output = 255 - input

    # Store the inverted values back into the image
    tl.store(image_ptr + offsets, output, mask=mask)

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](
        image,
        width, height,
        BLOCK_SIZE
    )
