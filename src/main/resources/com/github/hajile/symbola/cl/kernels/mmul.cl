#define TILE_SIDE 32
#define TILE_AREA (TILE_SIDE*TILE_SIDE)
#define SUB_STRIDE (TILE_AREA/4)

//#define ROWMAJOR(i, j, side) mad24(side, i, j)
//#define COLMAJOR(i, j, side) mad24(side, j, i)
//
//#define TILEOFFSET(ti, tj, side) mul24(ROWMAJOR(ti, tj, side), TILE_AREA)

#define ROWMAJOR(i, j, side) (side*i + j)
#define COLMAJOR(i, j, side) (side*j + i)

#define TILEOFFSET(ti, tj, side) (ROWMAJOR(ti, tj, side) * TILE_AREA)

/*
 TILE_SIDE x TILE_SIDE threads process 2x2 matrix each

 a: n*p
 b: m*p
 o: n*m
*/
__kernel void mmul(__global float* out, __global float const* a, __global float const* bt, int n, int m, int p)
{
  int tn = n / TILE_SIDE;
  int tm = m / TILE_SIDE;
  int tp = p / TILE_SIDE;

  /*
    val ti = i / tile
    val tj = j / tile
    val tileOffset = (tc * ti + tj) * tileArea
    val si = i % tile
    val sj = j % tile
    val part = sj & 0x01 | ((si & 0x01) << 1) // (0, 0), (0, 1), (1, 0), (1, 1)
    val partIx = (si / 2) * (tile / 2) + (sj / 2)
    tileOffset + part * partArea + partIx
  */
  // blocks are laid out row-major
  // global_id(0) oti otj | TILE_SIDE==4
  // 0            0   0
  // 1            0   0
  // 2            0   0
//  int osgi = get_global_id(0) % (n / 2);
//  int osgj = get_global_id(0) / (n / 2);

  int oti = get_group_id(0) / tm; // TSxTS block row
  int otj = get_group_id(0) % tm; // TSxTS block col

  int osi = get_local_id(0) / (TILE_SIDE / 2); // 2x2 matrix row inside a TSxTS block
  int osj = get_local_id(0) % (TILE_SIDE / 2); // ----""---- col -""-

//  printf("oti: %d, otj: %d, osi: %d, osj: %d\n", oti, otj, osi, osj);

  __global float* otile = out + TILEOFFSET(oti, otj, tm);
  __global float* osubm = otile + ROWMAJOR(osi, osj, TILE_SIDE/2);

  __local float atile[TILE_AREA];
  __local float btile[TILE_AREA];

  float temp00 = 0.0, temp01 = 0.0, temp10 = 0.0, temp11 = 0.0;

  // enumerating tiles
  for (int pp = 0; pp < tp; pp++) {
//    __global const float* atile = a + TILEOFFSET(oti, pp, tp);
//    __global const float* btile = bt + TILEOFFSET(otj, pp, tp);
    __global const float* atile_g = a + TILEOFFSET(oti, pp, tp);
    __global const float* btile_g = bt + TILEOFFSET(otj, pp, tp);

    for (int i = 0; i < 4; i++) {
      atile[get_local_id(0) + i*SUB_STRIDE] = atile_g[get_local_id(0) + i*SUB_STRIDE];
      barrier(CLK_LOCAL_MEM_FENCE); // this actually speeds things up
    }
    for (int i = 0; i < 4; i++) {
      btile[get_local_id(0) + i*SUB_STRIDE] = btile_g[get_local_id(0) + i*SUB_STRIDE];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // enumerating 2x2 submatrices
    for (int ppp = 0; ppp < TILE_SIDE/2; ppp++) {
//      __global const float* asubm = atile + ROWMAJOR(osi, ppp, TILE_SIDE/2);
//      __global const float* bsubm = btile + ROWMAJOR(osj, ppp, TILE_SIDE/2);
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

      temp00 = mad(a00, b00, temp00); temp00 = mad(a01, b10, temp00);
      temp01 = mad(a00, b01, temp01); temp01 = mad(a01, b11, temp01);
      temp10 = mad(a10, b00, temp10); temp10 = mad(a11, b10, temp10);
      temp11 = mad(a10, b01, temp11); temp11 = mad(a11, b11, temp11);

//      temp00 += a00*b00 + a01*b10;
//      temp01 += a00*b01 + a01*b11;
//      temp10 += a10*b00 + a11*b10;
//      temp11 += a10*b01 + a11*b11;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  osubm[SUB_STRIDE*0] = temp00;
  osubm[SUB_STRIDE*2] = temp01;
  osubm[SUB_STRIDE*1] = temp10;
  osubm[SUB_STRIDE*3] = temp11;
}
