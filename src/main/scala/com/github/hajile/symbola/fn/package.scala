package com.github.hajile.symbola

import breeze.linalg.DenseMatrix

package object fn {
  val M = MatrixExpr
  val S = ScalarExpr

  type SimpleMatrix = DenseMatrix[Float]
}
