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


/*
  matrices are in column-major format, B is transposed
*/
#define TILE_WIDTH 16
__kernel void mmultopt(__global const float* a, __global const float* bt, int ntiles, int mtiles, int ptiles, __global float* out)
{
    int i = get_global_id(0); // row (A row)
    int j = get_global_id(1); // column (B row)

    __local float achunk[TILE_WIDTH][TILE_WIDTH];
    __local float bchunk[TILE_WIDTH][TILE_WIDTH];

    int li = get_local_id(0);
    int lj = get_local_id(1);

    int gi = get_group_id(0);
    int gj = get_group_id(1);

    float temp = 0;

    int n = ntiles * TILE_WIDTH;
    int m = mtiles * TILE_WIDTH;
    int p = ptiles * TILE_WIDTH;


    for (int mm = 0; mm < ptiles; mm++) {
        //     C   R
        // a[lj + mm*TW][i]
        // b[li + mm*TW][j]
        achunk[lj][li] = a[(TILE_WIDTH*mm+lj)*n + i];
        bchunk[lj][li] = bt[(TILE_WIDTH*mm+lj)*m + (li + gj*TILE_WIDTH)];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_WIDTH; k++) {
           temp += achunk[k][li] * bchunk[k][lj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    out[i + j*n] = temp;
}
