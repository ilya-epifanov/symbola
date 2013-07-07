package com.github.hajile.symbola

import com.github.hajile.symbola.ad.BackwardGradient
import BackwardGradient.BackwardContext
import breeze.linalg.DenseMatrix
import breeze.numerics

package object fn {
  type SimpleMatrix = DenseMatrix[Double]

  object Fn {
    def input(name: String, v: Double) = Input(name, DenseMatrix(v))

    def input(name: String, v: SimpleMatrix) = Input(name, v)

    def value(v: Double): Expr = new Value(DenseMatrix(v))

    def sin(e: Expr): Expr = Sin(e)

    def cos(e: Expr): Expr = Cos(e)

    def sum(e1: Expr, e2: Expr): Expr = Sum(Seq(e1, e2))

    def prod(e1: Expr, e2: Expr): Expr = Prod(e1, e2)

    def neg(e: Expr): Expr = Neg(e)

    //    def pow(b: Expr, e: Expr): Expr = new Expr {
    //      def apply() = math.pow(b(), e())
    //    }
    //
    //    def pow(b: Expr, e: Double): Expr = new Expr {
    //      def apply() = math.pow(b(), e)
    //    }
  }

}
