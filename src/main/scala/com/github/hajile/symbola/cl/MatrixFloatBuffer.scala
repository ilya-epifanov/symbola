package com.github.hajile.symbola.cl

import breeze.linalg.DenseMatrix
import com.github.hajile.symbola.fn.ExprGraph.RealizedMatrix
import java.nio.FloatBuffer
import scalaxy.loops._

final class MatrixFloatBuffer(buf: FloatBuffer, shape: RealizedMatrix) {
  private val tile = 32
  private val tc = shape.cols / tile
  private val tileArea = tile * tile
  private val partArea = tileArea / 4

  def put(x: Int, y: Int, v: Float): MatrixFloatBuffer = {
    buf.put(offset(x, y), v)
    this
  }

  def toDenseMatrix: DenseMatrix[Float] = {
    val ret = new DenseMatrix[Float](shape.rows, shape.cols)
    for (i <- 0 until shape.rows optimized; j <- 0 until shape.cols optimized) {
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

  def checkApproxEquals(mx: DenseMatrix[Float]): Unit = {
    for (i <- 0 until shape.rows optimized; j <- 0 until shape.cols optimized) {
      val ref = mx(i, j)
      val act = buf.get(offset(i, j))

      require((math.abs(ref - act) / (1 + math.abs(ref) + math.abs(act))) <= 0.05, f"got $act, expected $ref")
    }
  }
}
