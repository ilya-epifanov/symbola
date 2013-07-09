package com.github.hajile.symbola.ad

import scala.collection.mutable
import breeze.linalg.DenseMatrix
import com.github.hajile.symbola.fn._
//import com.github.hajile.symbola.ad.BackwardGradient.BackwardContext
import scala.collection.immutable.SortedMap

//class BackwardGradient(protected val expr: Expr, protected val inputs: Input*) extends Gradient {
//  def apply() = {
//    val ctx = new BackwardContext(mutable.OpenHashMap[Input, SimpleMatrix]() ++=
//            (inputs.map((i: Input) => i -> DenseMatrix.zeros[Double](i.get.rows, i.get.cols))))
//    expr.backwards(DenseMatrix.ones(1, 1))(ctx)
//    SortedMap()(Input.InputOrderingByName) ++ ctx.pds
//  }
//}
//
//object BackwardGradient {
//  class BackwardContext(val pds: mutable.Map[Input, SimpleMatrix]) {
//    def contribute(input: Input, sensitivity: SimpleMatrix) {
//      if (pds.contains(input))
//        pds.update(input, sensitivity + pds(input))
//    }
//  }
//}