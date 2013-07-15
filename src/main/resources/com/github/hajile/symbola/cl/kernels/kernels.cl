__kernel void ew_cos(__global const float* a, __global float* out)
{
    int i = get_global_id(0);
    out[i] = cos(a[i]);
}

__kernel void transpose(__global float* output, __global const float* input, int rows, int cols)
{
    int gx = get_group_id(0);
    int gy = get_group_id(1);

    // (32, 2) is emulated by (64, 1) to ensure correct layout
    int lx = get_local_id(0) % 32;
    int ly = get_local_id(0) / 32;

    __local float tile[32 * 33]; // + the padding element

    int li = mad24(ly, 33, lx);
    int lo = mad24(lx, 33, ly);

    int in_x = mad24(gx, 32, lx);
    int in_y = mad24(gy, 32, ly);

    int input_index = mad24(in_y, cols, in_x);

    int out_x = mad24(gy, 32, lx);
    int out_y = mad24(gx, 32, ly);

    int output_index = mad24(out_y, rows + 32, out_x);

    int gi_stride  = cols * 2;
    int go_stride = (rows + 32) * 2;

    int li_stride  = 2 * (32 + 1);
    int lo_stride = 2;

    // load
    for (int i = 0; i < 16; i++) {
        tile[li] = input[input_index]; li += li_stride; input_index += gi_stride;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 16; i++) {
        output[output_index] = tile[lo]; lo += lo_stride; output_index += go_stride;
    }
}

__kernel void mmul(__global const float* a, __global const float* b, int dot_dim, __global float* out)
{
    int i = get_global_id(0); // row
    int j = get_global_id(1); // column
    int rows = get_global_size(0);
    int cols = get_global_size(1);

    float temp = 0;

    for (int k = 0; k < dot_dim; k++) {
        // a[i][j] = a[i + j*rows]
        // b[i][j] = b[i + j*dot_dim]

        temp += a[i + k*rows] * b[j*dot_dim + k];
    }
    out[dot_dim*j + i] = temp;
}

/*
  matrices are in column-major format, B is transposed
*/
#define TILE_WIDTH 2
__kernel void mmultopt(__global const float* restrict a, __global const float* restrict bt, int n, int m, int p, __global float* restrict out)
{
    int i = get_global_id(0) % TILE_WIDTH; // column (B row)
    int j = get_global_id(0) / TILE_WIDTH; // row (A row)

    __local float achunk[TILE_WIDTH*TILE_WIDTH];
    __local float bchunk[TILE_WIDTH*TILE_WIDTH];

    int li = get_local_id(0) % TILE_WIDTH;
    int lj = get_local_id(0) / TILE_WIDTH;

    int gi = get_group_id(0) % TILE_WIDTH;
    int gj = get_group_id(0) / TILE_WIDTH;

    float temp = 0;

    int ptiles = p / TILE_WIDTH;

    int ai = i;
    int bi = li + gj*TILE_WIDTH;

    for (int tile = 0; tile < ptiles; tile++) {
        int abj = TILE_WIDTH*tile+lj;

        achunk[lj*TILE_WIDTH + li] = a[abj*n + ai];
        bchunk[lj*TILE_WIDTH + li] = bt[abj*m + bi];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_WIDTH; k++) {
           temp = mad(achunk[k*TILE_WIDTH + li], bchunk[k*TILE_WIDTH + lj], temp);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out[j*n + i] = temp;
}
