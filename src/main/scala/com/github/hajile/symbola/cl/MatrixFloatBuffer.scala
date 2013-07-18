package com.github.hajile.symbola.cl

import breeze.linalg.DenseMatrix
import com.github.hajile.symbola.fn.ExprGraph.RealizedMatrix
import java.nio.FloatBuffer

final class MatrixFloatBuffer(buf: FloatBuffer, shape: RealizedMatrix) {
  private val tile = 4
  private val tc = shape.cols / tile
  private val tileArea = tile * tile
  private val partArea = tileArea / 4

  def put(x: Int, y: Int, v: Float): MatrixFloatBuffer = {
    buf.put(offset(x, y), v)
    this
  }

  def toDenseMatrix: DenseMatrix[Float] = {
    val ret = new DenseMatrix[Float](shape.rows, shape.cols)
    for (i <- 0 until shape.rows; j <- 0 until shape.cols) {
      ret.update(i, j, buf.get(offset(i, j)))
    }
    ret
  }

  def offset(i: Int, j: Int): Int = {
    val ti = i / tile
    val tj = j / tile
    val tileOffset = (tc * ti + tj) * tileArea
    val si = i % tile
    val sj = j % tile
    val part = (si & 0x01) | ((sj & 0x01) << 1) // (0, 0), (0, 1), (1, 0), (1, 1)
    val partIx = (si / 2) * (tile / 2) + (sj / 2)
    tileOffset + part * partArea + partIx
  }
}
