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

__kernel void mmult(__global const float* at, __global const float* b, int dot_dim, __global float* out)
{
    int i = get_global_id(0); // row
    int j = get_global_id(1); // column
    int rows = get_global_size(0);
    int cols = get_global_size(1);

    int lid = get_local_id(0);

    float temp = 0;

    __local float arow[32];
    __local float brow[32];
    __local float crow[32];

    arow[lid] = at[i*dot_dim + lid];
    brow[lid] = b[j*dot_dim + lid];
    crow[lid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    crow[lid] = arow[lid] * brow[lid];

    for(int offset = 1;
        offset < get_local_size(0);
        offset <<= 1) {
        int mask = (offset << 1) - 1;
        if ((local_index & mask) == 0) {
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
            scratch[local_index] = mine + other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_index == 0) {
        out[dot_dim*j + i] = scratch[0];
    }

    out[dot_dim*j + i] = temp;
}