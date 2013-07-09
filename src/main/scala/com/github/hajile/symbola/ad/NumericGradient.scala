package com.github.hajile.symbola.ad

import com.github.hajile.symbola.fn._
//import com.github.hajile.symbola.fn.Input

//class NumericGradient(val diff: Double, protected val expr: Expr, protected val inputs: Input*) extends Gradient {
//  private val diffRecip = 1.0 / diff
//
//  def apply(): Map[Input, SimpleMatrix] = {
//    val ret = Map.newBuilder[Input, SimpleMatrix]
//    for (i <- inputs) {
//      val in = i.get
//      val r = new SimpleMatrix(in.rows, in.cols)
//      for (j <- 0 until in.rows) {
//        for (k <- 0 until in.cols) {
//          val oldV = in(j, k)
//          val atLeft = expr()
//          in.update(j, k, oldV + diff)
//          val atRight = expr()
//          in.update(j, k, oldV)
//          r.update(j, k, (atRight - atLeft).apply(0, 0) * diffRecip)
//        }
//      }
//      ret += (i -> r)
//    }
//    ret.result()
//  }
//}
