package com.github.hajile.symbola.fn

import breeze.linalg.DenseMatrix

object Example extends App {
  //  import Fn._

  val m1 = DenseMatrix(Math.PI / 2, Math.PI).t
  val m2 = DenseMatrix(1d, 1d)
  val m3 = DenseMatrix.eye[Double](2)

  val in1 = M.InputCell("x₁")
  val in2 = M.InputCell("x₂")
  val in3 = M.InputCell("x₃")

//  println("in1 = " + in1.get)
//  println("in2 = " + in2.get)
//  println("in3 = " + in3.get)

//  println("f = " + numerics.sin(in1.get) * (in3.get * in2.get))

  val s1 = S.InputCell("x")
  val s2 = S.InputCell("y")
  val sin = S.Div(s1, s2)

  val sgrad = S.Grad(sin, Set(s1, s2))
  DotVisualizer.visualize("Scalar", Seq(sin, sgrad(s1), sgrad(s2)): _*)
  DotVisualizer.visualize("Scalar-Optimized", GraphOptimizer.optimizeScalar(sin, sgrad(s1), sgrad(s2)): _*)


  val expr = M.Prod(M.Dotwise1(in1, sin, s1), M.Prod(in3, in2))

  val grad = M.Grad(expr, Set(in1, in2, in3))


//  println(s"f = ${expr()}")
//  println(s"numeric: ${new NumericGradient(1.0e-10, expr, in1, in2, in3)()}")
//  println(s"backward: ${new BackwardGradient(expr, in1, in2, in3)()}")

  DotVisualizer.visualize("Unoptimized", Seq(grad(in1), grad(in2), grad(in3), expr): _*)

  val exprs = Map(in1.name -> grad(in1), in2.name -> grad(in2), in3.name -> grad(in3), "ƒ" -> expr)

  val optimized = GraphOptimizer.optimizeMatrix(exprs)
  DotVisualizer.visualize("Optimized", optimized.values.toSeq ++ Seq(expr): _*)

  println(s"symbolic optimized: $optimized")
//  println(s"symbolic optimized (evaluated): ${optimized.mapValues(_())}")
}
