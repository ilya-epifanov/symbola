package com.github.hajile.symbola.fn

import com.github.hajile.symbola.fn.ExprGraph.RealizedMatrix
import com.jogamp.opencl.{CLPlatform, CLContext}
import scala.util.Random
import com.jogamp.opencl.CLDevice.Type
import com.jogamp.opencl.util.Filter
import com.github.hajile.symbola.cl.OpenCLApp

object Example2 extends OpenCLApp {
  val expr = new ExprGraph(ctx)

  val x1 = expr.in("x1")
  //  val x2 = expr.in("x2", Matrix)
  //  val x3 = expr.in("x3", Matrix)

  val d1 = S.InputCell("x")

  val e1 = M.Prod(M.Dotwise1(x1, S.Cos(d1), d1), x1)

  expr.out("f", e1)

  val dim = 1024
  val rows = dim
  val cols = dim

  val rnd = new Random()

  val buf1 = expr.getIn("x1", RealizedMatrix(rows, cols))
  for (i <- 0 until rows; j <- 0 until cols) {
    buf1.put(i, j, rnd.nextGaussian().toFloat)
  }
  expr.writeIn("x1")

  expr.realize()

  for (i <- 0 until 4) {
    val began = System.nanoTime()
    expr.run()
    val elapsed = System.nanoTime() - began
    println(f"${elapsed / 1000000.0}%.3fms")
  }

  //  val buf2 = Pointer.allocateFloats(rows * cols)
  //  expr.get("f", buf2)

  //  for (i <- 0 until 6) {
  //    println(f"${buf1.get(i).toDouble}%.3f \t${buf2.get(i).toDouble}%.3f \t${math.cos(buf1.get(i).toDouble)}%.3f")
  //    require(buf2.get(i) == math.cos(buf1.get(i).toDouble))
  //  }
}
