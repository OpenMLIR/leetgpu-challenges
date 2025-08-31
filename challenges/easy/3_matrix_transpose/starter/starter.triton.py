import torch
import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_ir, stride_ic,
    stride_or, stride_oc,
    BLOCK_SIZE_ROW : tl.constexpr, BLOCK_SIZE_COL : tl.constexpr,
):
    # 1. determine the input tile coordinates this thread block is responsible for
    pid_m = tl.program_id(0)  # block index in row dimension
    pid_n = tl.program_id(1)  # block index in col dimension

    # 2. compute element-wise offsets within the tile
    offs_m = pid_m * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offs_n = pid_n * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    # 3. define global memory pointers for input tile (row-major)
    input_ptrs = input_ptr + offs_m[:, None] * stride_ir + offs_n[None, :] * stride_ic

    # 4. load input tile from global memory with boundary check
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    block = tl.load(input_ptrs, mask=mask, other=0)

    # 5. transpose the tile (swap rows and columns)
    transposed_block = tl.trans(block)  # Triton built-in transpose function

    # 6. compute global memory pointers for output tile (column-major)
    output_ptrs = output_ptr + offs_n[:, None] * M + offs_m[None, :]  # M is row stride after transpose

    # 7. store the transposed tile to global memory
    tl.store(output_ptrs, transposed_block, mask=mask.T)  # transpose mask as well

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    grid = lambda META: (triton.cdiv(rows, META['BLOCK_SIZE_ROW']), triton.cdiv(cols, META['BLOCK_SIZE_COL']))
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc,
        BLOCK_SIZE_ROW=32, BLOCK_SIZE_COL=64,
    )
