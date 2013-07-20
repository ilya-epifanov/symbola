#define TILE_SIDE 32
#define TILE_AREA (TILE_SIDE*TILE_SIDE)
#define SUB_STRIDE (TILE_AREA/4)

#define ROWMAJOR(i, j, side) (side*i + j)
#define COLMAJOR(i, j, side) (side*j + i)
#define TILEOFFSET(ti, tj, side) (ROWMAJOR(ti, tj, side) * TILE_AREA)

#define PRECISION 0

#if PRECISION == 2
#define acc(x, y, z) fma(x, y, z)
#elif (PRECISION == 1) && FP_FAST_FMAF
#define acc(x, y, z) fma(x, y, z)
#elif PRECISION == 1
#define acc(x, y, z) ((x)*(y) + (z))
#else
#define acc(x, y, z) mad(x, y, z)
#endif

/*
 TILE_SIDE x TILE_SIDE threads process 2x2 matrix each

 a: n*p
 b: m*p
 o: n*m
*/
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mmul(__global float* restrict out, __global float const* restrict a, __global float const* restrict bt,
          int n, int m, int p)
{
  int tn = n / TILE_SIDE;
  int tm = m / TILE_SIDE;
  int tp = p / TILE_SIDE;

  int oti = get_group_id(0) / tm; // TSxTS block row
  int otj = get_group_id(0) % tm; // TSxTS block col

  int osi = get_local_id(0) / (TILE_SIDE / 2); // 2x2 matrix row inside a TSxTS block
  int osj = get_local_id(0) % (TILE_SIDE / 2); // ----""---- col -""-

  __global float* otile = out + TILEOFFSET(oti, otj, tm);
  __global float* osubm = otile + ROWMAJOR(osi, osj, TILE_SIDE/2);

  __local float atile[TILE_AREA];
  __local float btile[TILE_AREA];

  float temp00 = 0.0, temp01 = 0.0, temp10 = 0.0, temp11 = 0.0;

  // enumerating tiles
  for (int pp = 0; pp < tp; pp++) {
    __global const float* atile_g = a + TILEOFFSET(oti, pp, tp);
    __global const float* btile_g = bt + TILEOFFSET(otj, pp, tp);

    for (int i = 0; i < 4; i++) {
      atile[get_local_id(0) + i*SUB_STRIDE] = atile_g[get_local_id(0) + i*SUB_STRIDE];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < 4; i++) {
      btile[get_local_id(0) + i*SUB_STRIDE] = btile_g[get_local_id(0) + i*SUB_STRIDE];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // enumerating 2x2 submatrices
    for (int ppp = 0; ppp < TILE_SIDE/2; ppp++) {
      __local const float* asubm = atile + ROWMAJOR(osi, ppp, TILE_SIDE/2);
      __local const float* bsubm = btile + ROWMAJOR(osj, ppp, TILE_SIDE/2);

      float a00 = asubm[SUB_STRIDE*0];
      float a01 = asubm[SUB_STRIDE*2];
      float a10 = asubm[SUB_STRIDE*1];
      float a11 = asubm[SUB_STRIDE*3];

      float b00 = bsubm[SUB_STRIDE*0];
      float b10 = bsubm[SUB_STRIDE*2]; // N.B. B matrix is transposed!
      float b01 = bsubm[SUB_STRIDE*1]; // -""-
      float b11 = bsubm[SUB_STRIDE*3];

      temp00 = acc(a00, b00, temp00); temp00 = acc(a01, b10, temp00);
      temp01 = acc(a00, b01, temp01); temp01 = acc(a01, b11, temp01);
      temp10 = acc(a10, b00, temp10); temp10 = acc(a11, b10, temp10);
      temp11 = acc(a10, b01, temp11); temp11 = acc(a11, b11, temp11);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  osubm[SUB_STRIDE*0] = temp00;
  osubm[SUB_STRIDE*2] = temp01;
  osubm[SUB_STRIDE*1] = temp10;
  osubm[SUB_STRIDE*3] = temp11;
}
