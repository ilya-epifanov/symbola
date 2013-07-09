__kernel void add_floats(__global const float* a, __global const float* b, __global float* out)
{
    int i = get_global_id(0);
    out[i] = a[i] + b[i];
}

__kernel void kernel2(__global const float* a, __global const float* b, __global float* out)
{
    int i = get_global_id(0);
    out[i] = a[i] / (1 + fabs(sin(a[i]) * b[i] * b[i]));
}

