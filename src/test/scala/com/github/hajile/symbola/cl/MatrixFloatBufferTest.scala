package com.github.hajile.symbola.cl

import org.scalatest.matchers.ShouldMatchers
import org.scalatest.FlatSpec
import java.nio.FloatBuffer
import com.github.hajile.symbola.fn.ExprGraph.RealizedMatrix
import breeze.linalg.DenseMatrix

class MatrixFloatBufferTest extends FlatSpec with ShouldMatchers {
  "single block matrix" should "be written correctly" in {
    val buf = FloatBuffer.allocate(16)
    val mx = new MatrixFloatBuffer(buf, RealizedMatrix(4, 4))
    for (i <- 0 until 4; j <- 0 until 4)
      mx.put(i, j, i*10 + j)

    buf.array() should equal(Array[Float](
      0, 2, 20, 22,
      1, 3, 21, 23,
      10, 12, 30, 32,
      11, 13, 31, 33
    ))
  }

  it should "be read correctly" in {
    val buf = FloatBuffer.wrap(Array[Float](
          0, 2, 20, 22,
          1, 3, 21, 23,
          10, 12, 30, 32,
          11, 13, 31, 33
        ))

    val mx = new MatrixFloatBuffer(buf, RealizedMatrix(4, 4))
    mx.toDenseMatrix should equal(DenseMatrix(
      (0, 1, 2, 3), (10, 11, 12, 13), (20, 21, 22, 23), (30, 31, 32, 33)
    ).map(_.toFloat))
  }

  "2x2 block matrix" should "be written correctly" in {
    val buf = FloatBuffer.allocate(64)
    val mx = new MatrixFloatBuffer(buf, RealizedMatrix(8, 8))
    for (i <- 0 until 8; j <- 0 until 8)
      mx.put(i, j, i*10 + j)

    buf.array() should equal(Array[Float](
      0, 2, 20, 22,
      1, 3, 21, 23,
      10, 12, 30, 32,
      11, 13, 31, 33,

      4, 6, 24, 26,
      5, 7, 25, 27,
      14, 16, 34, 36,
      15, 17, 35, 37,

      40, 42, 60, 62,
      41, 43, 61, 63,
      50, 52, 70, 72,
      51, 53, 71, 73,

      44, 46, 64, 66,
      45, 47, 65, 67,
      54, 56, 74, 76,
      55, 57, 75, 77
    ))
  }

  it should "be read correctly" in {
    val buf = FloatBuffer.wrap(Array[Float](
          0, 2, 20, 22,
          1, 3, 21, 23,
          10, 12, 30, 32,
          11, 13, 31, 33,

          4, 6, 24, 26,
          5, 7, 25, 27,
          14, 16, 34, 36,
          15, 17, 35, 37,

          40, 42, 60, 62,
          41, 43, 61, 63,
          50, 52, 70, 72,
          51, 53, 71, 73,

          44, 46, 64, 66,
          45, 47, 65, 67,
          54, 56, 74, 76,
          55, 57, 75, 77
        ))

    val mx = new MatrixFloatBuffer(buf, RealizedMatrix(8, 8))
    mx.toDenseMatrix should equal(DenseMatrix(
      (0, 1, 2, 3, 4, 5, 6, 7),
      (10, 11, 12, 13, 14, 15, 16, 17),
      (20, 21, 22, 23, 24, 25, 26, 27),
      (30, 31, 32, 33, 34, 35, 36, 37),
      (40, 41, 42, 43, 44, 45, 46, 47),
      (50, 51, 52, 53, 54, 55, 56, 57),
      (60, 61, 62, 63, 64, 65, 66, 67),
      (70, 71, 72, 73, 74, 75, 76, 77)
    ).map(_.toFloat))
  }
}
