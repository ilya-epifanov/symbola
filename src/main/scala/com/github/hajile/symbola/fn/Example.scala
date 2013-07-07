package com.github.hajile.symbola.fn

import breeze.linalg.DenseMatrix
import breeze.numerics
import com.github.hajile.symbola.ad.{SymbolicBackwardGradient, NumericGradient, BackwardGradient}

object Example extends App {
  import Fn._

  val in1 = input("x₁", DenseMatrix(Math.PI/2, Math.PI).t)
  val in2 = input("x₂", DenseMatrix(1d, 1d))
  val in3 = input("x₃", DenseMatrix.eye[Double](2))

  println("in1 = " + in1.get)
  println("in2 = " + in2.get)
  println("in3 = " + in3.get)

  println("f = " + numerics.sin(in1.get) * (in3.get * in2.get))

  val expr = prod(sin(in1), prod(in3, in2))

  println(s"f = ${expr()}")
  println(s"numeric: ${new NumericGradient(1.0e-10, expr, in1, in2, in3)()}")
  println(s"backward: ${new BackwardGradient(expr, in1, in2, in3)()}")

  val gradExprs: Map[Input, Expr] = new SymbolicBackwardGradient().adjointTreesFor(expr, Set(in1, in2, in3))
  DotVisualizer.visualize("Unoptimized", gradExprs.values.toSeq ++ Seq(expr) : _*)

  println(s"symbolic: $gradExprs")
  println(s"symbolic (evaluated): ${gradExprs.mapValues(_())}")

  val optimizedGradExprs: Map[Input, Expr] = GraphOptimizer.optimize(gradExprs)
  DotVisualizer.visualize("Optimized", optimizedGradExprs.values.toSeq ++ Seq(expr) : _*)

  println(s"symbolic optimized: $optimizedGradExprs")
  println(s"symbolic optimized (evaluated): ${optimizedGradExprs.mapValues(_())}")
}
