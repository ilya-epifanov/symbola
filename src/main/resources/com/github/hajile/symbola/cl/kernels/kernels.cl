__kernel void cos(__global const float* a, __global float* out)
{
    int i = get_global_id(0);
    out[i] = fcos(a);
}
