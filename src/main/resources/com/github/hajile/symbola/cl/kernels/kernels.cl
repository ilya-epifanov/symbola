__kernel void ew_cos(__global const float* a, __global float* out)
{
    int i = get_global_id(0);
    out[i] = cos(a[i]);
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

#define TILE_WIDTH 16
__kernel void mmultopt(__global const float* a, __global const float* bt, int ntiles, int mtiles, int ptiles, __global float* out)
{
    int i = get_global_id(0); // row
    int j = get_global_id(1); // column

    __local float achunk[TILE_WIDTH][TILE_WIDTH];
    __local float bchunk[TILE_WIDTH][TILE_WIDTH];

    int li = get_local_id(0);
    int lj = get_local_id(1);

    float temp = 0;

    for (int mm = 0; mm < mtiles; mm++) {
        achunk[li][lj] = a[i + j*get_global_size(1)];
        bchunk[li][lj] = bt[i + j*get_global_size(1)];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_WIDTH; k++) {
           temp = mad(achunk[k][lj], bchunk[k][lj], temp);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out[i + j*get_global_size(1)] = temp;
}
