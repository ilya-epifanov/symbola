package com.github.hajile.symbola.ad

import com.github.hajile.symbola.fn.{Eye, Expr, Input, Sum}
import scala.collection.immutable
import com.github.hajile.symbola.fn.ExprHasInputs._
import scala.collection.immutable.SortedMap

class SymbolicBackwardGradient {
  def adjointTreesFor(gradientOf: Expr, wrt: Set[Input]): Map[Input, Expr] = {
    SortedMap()(Input.InputOrderingByName) ++
            SymbolicBackwardGradient.getAdjoint(gradientOf, wrt, Eye(1)).mapValues {
              es =>
                Sum(es.toSeq)
            }
  }
}

object SymbolicBackwardGradient {
  private def getAdjoint(node: Expr, wrt: Set[Input], seed: Expr): immutable.HashMap[Input, Set[Expr]] = {
    val in = inputs(node)
    val adjoints: Seq[immutable.HashMap[Input, Set[Expr]]] = in.collect {
      case i: Input if wrt.contains(i) =>
        immutable.HashMap[Input, Set[Expr]](i -> Set(node.backwardsExpr(seed, i)))
      case i: Input =>
        immutable.HashMap[Input, Set[Expr]]()
      case e =>
        getAdjoint(e, wrt, node.backwardsExpr(seed, e))
    }
    if (adjoints.isEmpty)
      immutable.HashMap[Input, Set[Expr]]()
    else
      adjoints.reduce {
        (a, b) =>
          a.merged(b) {
            case ((k1, v1), (_, v2)) =>
              k1 -> (v1 ++ v2)
          }
      }
  }

}