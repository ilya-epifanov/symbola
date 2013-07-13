package com.github.hajile.symbola.cl

import com.github.hajile.symbola.fn.ExprGraph.RealizedMatrix
import java.nio.FloatBuffer

final class MatrixFloatBuffer(buf: FloatBuffer, shape: RealizedMatrix) {
  def put(x: Int, y: Int, v: Float): MatrixFloatBuffer = {
    buf.put(x + y * shape.rowsP, v)
    this
  }
}
