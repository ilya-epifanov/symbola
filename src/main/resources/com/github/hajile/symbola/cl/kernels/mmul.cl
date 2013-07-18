#define TILE_SIDE 4
#define TILE_AREA (TILE_SIDE*TILE_SIDE)
#define SUB_STRIDE (TILE_AREA/4)

#define ROWMAJOR(i, j, side) (side*i + j)
#define COLMAJOR(i, j, side) (i + side*j)

#define TILEOFFSET(ti, tj, side) (ROWMAJOR(ti, tj, side) * TILE_AREA)

/*
 TILE_SIDE x TILE_SIDE threads process 2x2 matrix each

 a: n*p
 b: m*p
 o: n*m
*/
__kernel void mmul(__global float* out, __global const float* restrict a, __global const float* restrict bt, int n, int m, int p)
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
  int osgi = get_global_id(0) % (n / 2);
  int osgj = get_global_id(0) / (n / 2);

  int oti = osgi / (TILE_SIDE / 2); // TSxTS block row
  int otj = osgj / (TILE_SIDE / 2); // TSxTS block col

  int osi = osgi % (TILE_SIDE / 2); // 2x2 matrix row inside a TSxTS block
  int osj = osgj % (TILE_SIDE / 2); // ----""---- col -""-

  __global float* otile = out + TILEOFFSET(oti, otj, tm);
  __global float* osubm = otile + ROWMAJOR(osi, osj, TILE_SIDE/2);

  float temp00 = 0.0, temp01 = 0.0, temp10 = 0.0, temp11 = 0.0;
  // enumerating tiles

  for (int pp = 0; pp < tp; pp++) {
    __global const float* atile = a + TILEOFFSET(oti, pp, tp);
    __global const float* btile = bt + TILEOFFSET(otj, pp, tp);

    // enumerating 2x2 submatrices
    for (int ppp = 0; ppp < TILE_SIDE/2; ppp++) {
      __global const float* asubm = atile + ROWMAJOR(osi, ppp, TILE_SIDE/2);
      __global const float* bsubm = btile + ROWMAJOR(osj, ppp, TILE_SIDE/2);

      float a00 = asubm[SUB_STRIDE*0];
      float a01 = asubm[SUB_STRIDE*2];
      float a10 = asubm[SUB_STRIDE*1];
      float a11 = asubm[SUB_STRIDE*3];

      float b00 = bsubm[SUB_STRIDE*0];
      float b10 = bsubm[SUB_STRIDE*2]; // N.B. B matrix is transposed!
      float b01 = bsubm[SUB_STRIDE*1]; // -""-
      float b11 = bsubm[SUB_STRIDE*3];

      temp00 += a00*b00 + a01*b10;
      temp01 += a00*b01 + a01*b11;
      temp10 += a10*b00 + a11*b10;
      temp11 += a10*b01 + a11*b11;
    }
  }

//  if (true) {
//    float marker = NAN;
//    osubm[0] = marker;
//    osubm[SUB_STRIDE] = marker;
//    osubm[SUB_STRIDE*2] = marker;
//    osubm[SUB_STRIDE*3] = marker;
//  }

//  float baset = 100;
//  float bases = 2;
//  osubm[SUB_STRIDE*0] = osi + osj * 0.1;
//  osubm[SUB_STRIDE*1] = baset + oti + otj * 0.1;
//  osubm[SUB_STRIDE*2] = NAN;
//  osubm[SUB_STRIDE*3] = osgi + osgj * 0.1;

  osubm[SUB_STRIDE*0] = temp00;
  osubm[SUB_STRIDE*2] = temp01;
  osubm[SUB_STRIDE*1] = temp10;
  osubm[SUB_STRIDE*3] = temp11;
}
